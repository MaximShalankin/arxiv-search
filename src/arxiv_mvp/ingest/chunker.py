"""Markdown-aware chunker for Docling output.

Contract (ticket-002.1 M1.1):
- Input:  markdown string + arxiv_id
- Output: list[Chunk] with
    * text              — chunk body
    * heading_path      — "Methods > 3.2 Training Setup" (ATX heading chain)
    * chunk_index       — 0-based, document-sequential
    * token_count       — via tiktoken cl100k_base (fallback: word*1.3)
    * arxiv_id
    * content_hash_sha256

Rules:
- ATX headings (`# Foo`) define section boundaries; each new heading updates
  heading_path and flushes current buffer.
- Paragraph-group: collect text until ~target_tokens, then flush. Never split
  a paragraph mid-way (one paragraph may exceed target).
- Tables (markdown pipe-tables) are atomic — STC (Small Table Chunk). Each
  table becomes its own chunk with prefix `[TABLE]\\n` so retrieval can down-
  rank or specialize.
- Math markers (`<!-- formula-not-decoded -->`) pass through as text.
- Empty markdown → [] (caller logs warning).

HARD SIZE CAP (added 2026-06-17 after ticket-002.1 Phase E bug):
- Any unit exceeding `_HARD_MAX_CHARS` (24K chars ≈ 6K tokens) is split into
  multiple sub-units BEFORE Phase 2 grouping. This guards against pathological
  cases discovered in Phase E: Docling can emit single 96K-char tables from
  paper appendices, and oversized paragraphs. Without the cap, these chunks
  blow vLLM BGE-M3's 8192-token context (reject at 131K chars hard limit, or
  silently truncate below it).
- Tables: split by rows preserving header (each sub-table keeps header + N data rows).
- Text: split by sentence boundaries; hard-split sentence if itself oversized.
"""
import re
from dataclasses import dataclass

from ..utils.hashing import sha256_hex

# Tokenizer: tiktoken cl100k_base (matches GPT-4/claude-haiku economic LLMs)
# Falls back to word-based estimate if tiktoken unavailable (offline).
_ENC = None
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def count_tokens(text: str) -> int:
    """Token count. tiktoken if available; else word_count * 1.3 estimate."""
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return max(1, int(len(text.split()) * 1.3))


# Pre-compiled regex for ATX headings (lines starting with 1-6 '#' followed by space)
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$')

# Hard cap on chunk size. vLLM BGE-M3 has 8192-token context, tokenizer is
# XLM-RoBERTa (NOT tiktoken cl100k_base). XLM-R density observed on math-heavy
# arxiv papers up to 0.72 tokens/char (e.g., 0904.0322 from 2009). At cap=9000,
# worst-case XLM-R token count = 9000 * 0.72 + [TABLE]\n prefix ≈ 6500 tokens.
# Safe margin under 8192. Iteration history:
#   v1: cap=24000  → 3 papers still failed (chunks of 18-24K chars → 8-13K XLM-R tokens)
#   v2: cap=10000  → table prefix [TABLE]\n (+9 chars) pushed to 10009 chars, edge case
#   v3: cap=9000   → conservative, accounts for prefix + worst-case XLM-R density
_HARD_MAX_CHARS = 9000

# Sentence boundary: . ! ? followed by whitespace. Crude but adequate for
# chunking purposes (we're not doing NLP-grade sentence splitting).
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')


def _split_long_unit(path: str, kind: str, text: str) -> list[tuple[str, str, str]]:
    """Split a unit exceeding _HARD_MAX_CHARS into multiple sub-units.

    Tables: split by rows preserving header.
    Text: split by sentence boundaries; hard-split oversized single sentence.
    Returns at least one (path, kind, text) tuple.
    """
    if len(text) <= _HARD_MAX_CHARS:
        return [(path, kind, text)]
    if kind == "table":
        return _split_table_by_rows(path, text)
    return _split_text_by_sentences(path, text)


def _split_table_by_rows(path: str, text: str) -> list[tuple[str, str, str]]:
    """Split markdown table into sub-tables, repeating the header block in each.

    Header = first line (column names) + optional second line (divider with ---).
    Data rows are distributed to keep each sub-table ≤ _HARD_MAX_CHARS.
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return [(path, "table", text)]

    header_lines = [lines[0]]
    data_start = 1
    if (len(lines) > 1
            and lines[1].lstrip().startswith("|")
            and "---" in lines[1]):
        header_lines.append(lines[1])
        data_start = 2
    header_block = "\n".join(header_lines)

    data_rows = lines[data_start:]
    sub_units: list[tuple[str, str, str]] = []
    cur_rows: list[str] = []

    for row in data_rows:
        # Project exact size if we append this row
        candidate = header_block + "\n" + "\n".join(cur_rows + [row])
        if cur_rows and len(candidate) > _HARD_MAX_CHARS:
            sub_units.append((path, "table", header_block + "\n" + "\n".join(cur_rows)))
            cur_rows = []
        cur_rows.append(row)

    if cur_rows:
        sub_units.append((path, "table", header_block + "\n" + "\n".join(cur_rows)))
    final: list[tuple[str, str, str]] = []
    for path_, kind, txt in sub_units:
        if len(txt) <= _HARD_MAX_CHARS:
            final.append((path_, kind, txt))
        else:
            # Last-resort: split by char window with header prepended.
            cap = max(1, _HARD_MAX_CHARS - len(header_block) - 2)
            body = txt[len(header_block) + 1:]
            for i in range(0, len(body), cap):
                final.append((path_, kind, header_block + "\n" + body[i:i + cap]))
    return final


def _split_text_by_sentences(path: str, text: str) -> list[tuple[str, str, str]]:
    """Split long text by sentence boundaries.

    Single sentence larger than _HARD_MAX_CHARS gets hard-split by char window.
    """
    sentences = _SENTENCE_END_RE.split(text)
    cap = _HARD_MAX_CHARS
    sub_units: list[tuple[str, str, str]] = []
    cur: list[str] = []
    cur_bytes = 0

    for sent in sentences:
        sent_bytes = len(sent) + 1
        # Hard-split oversized single sentence
        if sent_bytes > cap:
            if cur:
                sub_units.append((path, "text", " ".join(cur)))
                cur = []
                cur_bytes = 0
            for i in range(0, len(sent), cap - 100):
                sub_units.append((path, "text", sent[i:i + cap - 100]))
            continue
        if cur and cur_bytes + sent_bytes > cap:
            sub_units.append((path, "text", " ".join(cur)))
            cur = []
            cur_bytes = 0
        cur.append(sent)
        cur_bytes += sent_bytes

    if cur:
        sub_units.append((path, "text", " ".join(cur)))
    return sub_units


@dataclass
class Chunk:
    text: str
    heading_path: str          # "Methods > 3.2 Training Setup", or "" for top-level
    chunk_index: int
    token_count: int
    arxiv_id: str
    content_hash_sha256: str

    def __post_init__(self):
        # Invariant: text is non-empty stripped
        assert self.text.strip(), "Chunk text must be non-empty"


def _sanitize_text(text: str) -> str:
    """Remove characters that break downstream storage.

    PostgreSQL TEXT rejects \\x00 (null byte). Some Docling outputs contain
    null bytes from corrupt PDF text layers (~4% of arxiv PDFs). Strip them
    before chunking so they never reach PG / embeddings / Qdrant.
    2026-08-27 (reviewer CTA): extended to all C0 controls (except \\n\\t\\r),
    DEL and C1 (0x80-0x9F) — strict JSON clients reject them in responses.
    """
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)


def _emit_paragraph(buf: list[str], heading_path: str) -> str | None:
    """Join buffered lines into a paragraph string; None if empty."""
    if not buf:
        return None
    text = "\n".join(buf).strip()
    return text or None


def chunk_markdown(
    markdown: str,
    arxiv_id: str,
    target_tokens: int = 300,
) -> list[Chunk]:
    """Split markdown into Chunks per ticket-002.1 contract."""
    if not markdown or not markdown.strip():
        return []

    # Scrub null bytes from PDF text layer (otherwise PG TEXT rejects)
    markdown = _sanitize_text(markdown)

    lines = markdown.split("\n")
    # Phase 1: walk lines, build ordered list of (heading_path, kind, text) units.
    # kind is 'text' or 'table'. Each unit is atomic for chunking purposes.
    units: list[tuple[str, str, str]] = []
    heading_stack: list[tuple[int, str]] = []  # [(level, text), ...]
    para_buf: list[str] = []

    def _flush_para():
        if para_buf:
            text = "\n".join(para_buf).strip()
            if text:
                path = " > ".join(h[1] for h in heading_stack)
                units.append((path, "text", text))
            para_buf.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            _flush_para()
            level = len(m.group(1))
            text = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            i += 1
            continue

        # Blank line: paragraph boundary
        if not line.strip():
            _flush_para()
            i += 1
            continue

        # Table block: line starts with '|' and the next line is a divider ('|---|')
        if line.lstrip().startswith("|"):
            # collect contiguous lines starting with '|' or empty cells
            tbl = [line]
            j = i + 1
            while j < n and lines[j].lstrip().startswith("|"):
                tbl.append(lines[j])
                j += 1
            # Heuristic: only treat as table if at least 2 rows (header + divider)
            if len(tbl) >= 2:
                _flush_para()
                path = " > ".join(h[1] for h in heading_stack)
                units.append((path, "table", "\n".join(tbl)))
                i = j
                continue
            # else: falls through to regular line handling

        # Regular line — accumulate into paragraph buffer
        para_buf.append(line)
        i += 1

    _flush_para()

    # Phase 1.5: enforce hard size cap by splitting any oversized units.
    # Without this, Docling's giant appendix tables (up to 96K chars observed
    # in Phase E) and pathological single paragraphs produce chunks that blow
    # vLLM BGE-M3's 8192-token context.
    normalized_units: list[tuple[str, str, str]] = []
    for path_u, kind_u, text_u in units:
        normalized_units.extend(_split_long_unit(path_u, kind_u, text_u))
    units = normalized_units

    # Phase 2: group units into chunks.
    # Text units: group up to target_tokens. New heading_path breaks the group
    # (keeps each chunk under one section for cleaner retrieval context).
    # Table units: always their own chunk (STC).
    chunks: list[Chunk] = []
    cur_text: list[str] = []
    cur_tokens = 0
    cur_path: str | None = None
    chunk_idx = 0

    def _flush_chunk():
        nonlocal cur_text, cur_tokens, cur_path, chunk_idx
        if not cur_text:
            return
        text = "\n\n".join(cur_text)
        chunks.append(Chunk(
            text=text,
            heading_path=cur_path or "",
            chunk_index=chunk_idx,
            token_count=count_tokens(text),
            arxiv_id=arxiv_id,
            content_hash_sha256=sha256_hex(text),
        ))
        chunk_idx += 1
        cur_text = []
        cur_tokens = 0
        # cur_path retained until next unit arrives (path-change also flushes)

    for path, kind, text in units:
        if kind == "table":
            _flush_chunk()
            body = f"[TABLE]\n{text}"
            chunks.append(Chunk(
                text=body,
                heading_path=path,
                chunk_index=chunk_idx,
                token_count=count_tokens(body),
                arxiv_id=arxiv_id,
                content_hash_sha256=sha256_hex(body),
            ))
            chunk_idx += 1
            cur_path = None  # force flush before next text unit
            continue

        # Text unit
        if cur_path is not None and path != cur_path:
            _flush_chunk()
        if not cur_text:
            cur_path = path

        unit_tokens = count_tokens(text)
        # If adding this unit overshoots target AND we already have content,
        # flush first.
        if cur_text and cur_tokens + unit_tokens > target_tokens:
            _flush_chunk()
            cur_path = path

        # If single unit > target (rare: long paragraph), still emit as chunk.
        cur_text.append(text)
        cur_tokens += unit_tokens

    _flush_chunk()
    return chunks
