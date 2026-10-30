"""FastAPI app: core search and RAG endpoints.

Kept endpoints:
  GET  /                      JSON service description
  GET  /health                liveness + storage connectivity
  GET  /stats                 corpus counts (cached, background refresh)
  POST /search                sparse | dense | hybrid | rerank
  POST /ask                   retrieve -> rerank -> LLM with [N] citations
  GET  /paper/{arxiv_id}      paper metadata + chunk ids
  POST /paper/{arxiv_id}/ask  scoped RAG for one paper
  POST /warmup                preload the reranker

Deliberately not part of this distilled service: SEO/impact analysis, claim
matrix, conflicting evidence, table extraction, citation graph, promotion,
multi-turn sessions, export, and any frontend.
"""
import asyncio
import math
import re
import time
from typing import Annotated

import asyncpg
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from ..utils.filters import FilterError, apply_recency, build_qdrant_filter
from ..utils.logging import get_logger
from . import schemas
from .dependencies import lifespan

log = get_logger("api.routes")

# Matches [1], [2, 4], [1,2,3] citation markers.
_CITE_RE = re.compile(r"\[([\d,\s]+)\]")


# ─── Boilerplate chunk handling ──────────────────────────────────────────────
# References / bibliography / term-list front-matter chunks are indexed as
# normal chunks and pollute content search. Drop them before rerank and demote
# them to the end for non-rerank /search paths.
_HEADING_NUMBERING_RE = re.compile(
    r"^(?:(?:\d+|[ivxl]+)[.\s]+|[•■]\s*)+", re.IGNORECASE)
_FORMAT_BOILERPLATE_RE = re.compile(r"reference format", re.IGNORECASE)


def _is_boilerplate_chunk(heading_path: str | None) -> bool:
    hp = (heading_path or "").strip()
    if not hp:
        return False
    if _FORMAT_BOILERPLATE_RE.search(hp):
        return True
    seg = hp.split(">")[-1].strip()
    seg = _HEADING_NUMBERING_RE.sub("", seg)
    low = seg.lower()
    if low.startswith(("reference", "bibliograph")) or low.endswith(
            ("references", "bibliography")):
        return True
    if low.startswith(("index terms", "ccs concepts", "keyword:")):
        return True
    return low == "keywords" or low.startswith(
        ("keywords:", "keywords :", "keywords and"))


def _drop_boilerplate_hits(hits: list) -> list:
    """Remove boilerplate chunks from candidate lists (pre-rerank paths)."""
    return [h for h in hits if not _is_boilerplate_chunk(h.heading_path)]


def _demote_boilerplate_hits(hits: list) -> list:
    """Content chunks first, boilerplate last (stable) — non-rerank /search."""
    return sorted(hits, key=lambda h: _is_boilerplate_chunk(h.heading_path))


# ─── Text hydration from PostgreSQL ─────────────────────────────────────────
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _scrub_control(s: str) -> str:
    """Replace C1 control chars from PDF text layers; keep \\n \\t \\r."""
    return _CONTROL_CHARS_RE.sub(" ", s) if s else s


async def _enrich_hits(state, hits: list) -> None:
    """Hydrate hit text+title from PG by chunk_id.

    PostgreSQL ``chunks.text`` is the single source of truth for chunk text.
    Qdrant payloads intentionally omit text; hydration happens before rerank
    and before the response is returned.
    """
    ids = [h.chunk_id for h in hits if getattr(h, "chunk_id", None)]
    if not ids:
        return
    async with state.postgres.pool.acquire() as conn:
        # Compare as uuid, not text: casting the column defeats chunks_pkey and
        # turns every lookup into a full scan on a large corpus.
        rows = await conn.fetch(
            "SELECT c.chunk_id::text AS chunk_id, c.text, p.title "
            "FROM chunks c JOIN papers p ON p.arxiv_id = c.arxiv_id "
            "WHERE c.chunk_id = ANY($1::uuid[])",
            list(dict.fromkeys(ids)),
        )
    m = {r["chunk_id"]: r for r in rows}
    for h in hits:
        r = m.get(getattr(h, "chunk_id", ""))
        if r is None:
            continue
        if r["text"]:
            h.text = _scrub_control(r["text"])
        if r["title"] and not h.title:
            h.title = r["title"]


# Backward-compatible alias used by older call sites.
_enrich_titles = _enrich_hits


# ─── Heavy-request protection ────────────────────────────────────────────────
# One heavy request at a time. A second concurrent request gets HTTP 429,
# never silent queueing. Heavy = /ask, /paper/{id}/ask, /search rerank.
_HEAVY_SEM = asyncio.Semaphore(1)
_HEAVY_BUSY_DETAIL = (
    "heavy request already running — retry after it completes "
    "(/ask, /paper/{id}/ask and /search rerank are serialized)")


class _HeavyLane:
    """Context manager: one heavy request at a time; 429 instead of queuing."""

    async def __aenter__(self):
        try:
            await asyncio.wait_for(_HEAVY_SEM.acquire(), timeout=1.0)
        except TimeoutError:
            raise HTTPException(429, _HEAVY_BUSY_DETAIL)

    async def __aexit__(self, *exc):
        _HEAVY_SEM.release()


_HEAVY_LANE = _HeavyLane()


async def _heavy_lane_dep():
    """FastAPI dependency wrapper for the heavy lane."""
    async with _HEAVY_LANE:
        yield


async def _preflight_stores(state) -> None:
    """Cheap storage ping before heavy pipelines (503 + reason)."""
    try:
        await asyncio.wait_for(
            state.qdrant.client.get_collection(state.qdrant.collection_name),
            timeout=15.0)
    except Exception as e:
        raise HTTPException(503, f"qdrant unavailable: {type(e).__name__}")
    try:
        async with state.postgres.pool.acquire() as conn:
            await asyncio.wait_for(conn.fetchval("SELECT 1"), timeout=15.0)
    except Exception as e:
        raise HTTPException(503, f"postgres unavailable: {type(e).__name__}")


async def _ensure_reranker(state) -> None:
    """Load the reranker on first use; 503 with an actionable hint on failure."""
    if getattr(state, "reranker_ready", False):
        return
    try:
        await state.reranker.ensure_loaded()
        state.reranker_ready = True
    except Exception as e:
        raise HTTPException(
            503,
            "reranker unavailable: "
            f"{type(e).__name__}: {str(e)[:200]}. Install the [rerank] extra "
            "or set RERANKER_HTTP_URL to an HTTP reranker.",
        )


# ─── App ──────────────────────────────────────────────────────────────────────
_ROOT_ENDPOINTS = [
    ("POST", "/search", "Retrieval search (bm25, dense, hybrid, rerank)"),
    ("POST", "/ask", "RAG: retrieve + rerank + cited answer"),
    ("GET", "/paper/{arxiv_id}", "Paper metadata + chunk listing"),
    ("POST", "/paper/{arxiv_id}/ask", "Scoped Q&A against one paper"),
    ("GET", "/stats", "Corpus statistics"),
    ("GET", "/health", "Service health check"),
]

app = FastAPI(
    title="arxiv-search",
    description=(
        "Distilled arxiv-search core: BGE-M3 dense + native sparse retrieval, "
        "RRF fusion, optional cross-encoder rerank, and RAG answers with "
        "inline [N] citations.\n\n"
        "Limitations: no authentication, binds to localhost by default, one "
        "heavy request at a time (429), and storage failures surface as 503. "
        "Do not expose this service to the public internet. See README.md and "
        "docs/LIMITATIONS.md."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root(request: Request):
    """JSON service description for API clients."""
    stats = getattr(request.app.state, "corpus_stats", None) or {}
    return {
        "name": app.title,
        "version": app.version,
        "endpoints": [
            {"method": m, "path": p, "summary": s} for m, p, s in _ROOT_ENDPOINTS
        ],
        "docs_url": "/docs",
        "curl_example": (
            'curl -X POST http://127.0.0.1:8080/search '
            '-H "Content-Type: application/json" '
            '-d \'{"query": "retrieval augmented generation", '
            '"mode": "hybrid", "top_k": 5}\''
        ),
        "n_papers": stats.get("n_papers"),
        "n_chunks": stats.get("n_chunks"),
        "year_range": stats.get("year_range"),
        "top_categories": stats.get("top_categories"),
    }


# ─── Corpus statistics ───────────────────────────────────────────────────────
def _arxiv_id_to_year(arxiv_id: str) -> int:
    """arxiv_id -> full year. '0701.1234' -> 2007, '2606.12345' -> 2026."""
    yy = int(arxiv_id[:2])
    return yy + (1900 if yy >= 91 else 2000)


_STATS_REFRESH_TTL_S = 300.0
_STATS_REFRESH_RUNNING = False


async def _recompute_corpus_stats(state) -> dict:
    """Full stats recompute from PG (counts + top categories)."""
    async with state.postgres.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT count(*) AS n_papers FROM papers")
        row_c = await conn.fetchrow("SELECT count(*) AS n_chunks FROM chunks")
        row_y = await conn.fetchrow(
            "SELECT min(arxiv_id) FILTER "
            "(WHERE arxiv_id ~ '^[0-9]{4}\\.[0-9]{4,5}$') AS mn, "
            "max(arxiv_id) FILTER "
            "(WHERE arxiv_id ~ '^[0-9]{4}\\.[0-9]{4,5}$') AS mx "
            "FROM papers")
        top = await conn.fetch(
            "SELECT category, count(*) AS n_papers FROM papers, "
            "unnest(categories) AS category GROUP BY category "
            "ORDER BY n_papers DESC LIMIT 10")
    return {
        "n_papers": row["n_papers"],
        "n_chunks": row_c["n_chunks"],
        "year_range": [
            _arxiv_id_to_year(row_y["mn"]) if row_y["mn"] else None,
            _arxiv_id_to_year(row_y["mx"]) if row_y["mx"] else None,
        ],
        "top_categories": [
            {"category": t["category"], "n_papers": t["n_papers"]} for t in top
        ],
    }


async def _refresh_corpus_stats_bg(state) -> None:
    """Recompute stats off the request path, gated on the heavy lane."""
    global _STATS_REFRESH_RUNNING
    try:
        async with _HEAVY_SEM:
            t0 = time.monotonic()
            state.corpus_stats = await _recompute_corpus_stats(state)
            state.corpus_stats_ts = time.monotonic()
            log.info("corpus_stats_refreshed",
                     elapsed_s=round(time.monotonic() - t0, 1),
                     n_papers=state.corpus_stats["n_papers"])
    except Exception as e:
        log.warning("corpus_stats_recompute_failed",
                    error=f"{type(e).__name__}: {str(e)[:120]}")
    finally:
        _STATS_REFRESH_RUNNING = False


@app.get("/stats", response_model=schemas.CorpusStatsResponse)
async def corpus_stats(request: Request):
    """Corpus statistics: cached, with a background refresh every 5 minutes."""
    global _STATS_REFRESH_RUNNING
    state = request.app.state
    stats = getattr(state, "corpus_stats", None)
    if stats is None:
        raise HTTPException(503, "corpus stats not yet loaded")
    ts = getattr(state, "corpus_stats_ts", 0.0)
    if time.monotonic() - ts > _STATS_REFRESH_TTL_S and not _STATS_REFRESH_RUNNING:
        _STATS_REFRESH_RUNNING = True
        asyncio.create_task(_refresh_corpus_stats_bg(state))
    return schemas.CorpusStatsResponse(
        n_papers=stats["n_papers"],
        n_chunks=stats["n_chunks"],
        year_range=stats["year_range"],
        top_categories=stats["top_categories"],
    )


# ─── Response helpers ───────────────────────────────────────────────────────
def _hit_to_schema(hit) -> schemas.SearchHit:
    score_norm = None
    if hit.source == "reranked" and hit.score is not None:
        try:
            score_norm = round(1.0 / (1.0 + math.exp(-hit.score)), 4)
        except OverflowError:
            score_norm = 0.0 if hit.score < 0 else 1.0
    tier = getattr(hit, "tier", "fulltext") or "fulltext"
    return schemas.SearchHit(
        chunk_id=hit.chunk_id,
        score=hit.score,
        source=hit.source,
        text=hit.text,
        arxiv_id=hit.arxiv_id,
        heading_path=hit.heading_path,
        title=hit.title,
        chunk_index=hit.chunk_index,
        year=hit.year,
        categories=hit.categories,
        raw_score=hit.raw_score,
        score_norm=score_norm,
        tier=tier,
    )


def _dedupe_by_paper(hits: list) -> list:
    """Keep the best (highest-ranked) chunk per arxiv_id."""
    seen: set[str] = set()
    out = []
    for h in hits:
        key = h.arxiv_id or h.chunk_id
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


# ─── Health and warmup ──────────────────────────────────────────────────────
@app.get("/health", response_model=schemas.HealthResponse)
async def health(request: Request):
    """Liveness + storage connectivity probe."""
    state = request.app.state
    pg_ok = "ok"
    qd_ok = "ok"
    try:
        async with state.postgres.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as e:
        pg_ok = f"error: {type(e).__name__}"
    try:
        await state.qdrant.client.get_collection(state.qdrant.collection_name)
    except Exception as e:
        qd_ok = f"error: {type(e).__name__}"
    emb_ok = "ok" if state.embedder._client is not None else "not-initialized"
    return schemas.HealthResponse(
        postgres=pg_ok,
        qdrant=qd_ok,
        embedder=emb_ok,
        reranker_loaded=getattr(state, "reranker_ready", False),
    )


@app.post("/warmup")
async def warmup(request: Request):
    """Force-load the reranker (useful before the first rerank query)."""
    state = request.app.state
    t0 = time.monotonic()
    await _ensure_reranker(state)
    return {"reranker_loaded": True,
            "elapsed_ms": round((time.monotonic() - t0) * 1000, 1)}


# ─── Search ─────────────────────────────────────────────────────────────────
@app.post("/search", response_model=schemas.SearchResponse)
async def search(req: schemas.SearchRequest, request: Request):
    """Retrieval-only search: sparse, dense, hybrid, or rerank."""
    state = request.app.state
    t_start = time.monotonic()
    breakdown: dict[str, float] = {}

    try:
        filters = apply_recency(req.filters, req.recency)
        if filters is not None:
            build_qdrant_filter(filters)
    except FilterError as e:
        raise HTTPException(400, f"invalid filters: {e}")

    if req.mode == "bm25":
        t0 = time.monotonic()
        hits = await state.sparse.search(
            req.query, top_k=min(req.top_k * 4, 500), filters=filters)
        hits = _demote_boilerplate_hits(hits)[:req.top_k]
        breakdown["sparse_ms"] = round((time.monotonic() - t0) * 1000, 1)
    elif req.mode == "dense":
        t0 = time.monotonic()
        hits = await state.dense.search(
            req.query, top_k=min(req.top_k * 4, 500), filters=filters)
        hits = _demote_boilerplate_hits(hits)[:req.top_k]
        breakdown["dense_ms"] = round((time.monotonic() - t0) * 1000, 1)
    elif req.mode in ("hybrid", "rerank"):
        fetch_k = min(req.top_k_hybrid * 3, 500)
        t0 = time.monotonic()
        d_hits, s_hits = await _gather(
            state.dense.search(req.query, top_k=fetch_k, filters=filters),
            state.sparse.search(req.query, top_k=fetch_k, filters=filters),
        )
        breakdown["parallel_dense_sparse_ms"] = round(
            (time.monotonic() - t0) * 1000, 1)
        from ..retrieval import rrf_fuse
        t0 = time.monotonic()
        fused = rrf_fuse(d_hits, s_hits, k=60, weights=(0.5, 0.5),
                         top_k=fetch_k)
        breakdown["rrf_ms"] = round((time.monotonic() - t0) * 1000, 2)
        if req.mode == "rerank":
            async with _HEAVY_LANE:
                fused = _drop_boilerplate_hits(fused)[:req.top_k_hybrid]
                # The reranker needs chunk text; Qdrant payload omits it.
                await _enrich_hits(state, fused)
                await _ensure_reranker(state)
                t0 = time.monotonic()
                hits = await state.reranker.rerank(
                    req.query, fused, top_k=req.top_k, candidate_window=2000)
                if req.min_score is not None:
                    hits = [h for h in hits if h.score >= req.min_score]
                breakdown["rerank_ms"] = round(
                    (time.monotonic() - t0) * 1000, 1)
        else:
            hits = _demote_boilerplate_hits(fused)[:req.top_k]
    else:
        raise HTTPException(
            400, f"unknown mode: {req.mode}. Use bm25|dense|hybrid|rerank")

    breakdown["total_ms"] = round((time.monotonic() - t_start) * 1000, 1)
    if req.dedupe_by_paper:
        hits = _dedupe_by_paper(hits)
    await _enrich_titles(state, hits)
    return schemas.SearchResponse(
        query=req.query,
        mode=req.mode,
        top_k=req.top_k,
        elapsed_ms=breakdown["total_ms"],
        hits=[_hit_to_schema(h) for h in hits],
        breakdown_ms=breakdown,
    )


# ─── RAG ────────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=schemas.AskResponse)
async def ask(req: schemas.AskRequest, request: Request,
              _lane: Annotated[None, Depends(_heavy_lane_dep)]):
    """Full RAG: retrieve -> rerank -> LLM answer with [N] citations."""
    state = request.app.state
    await _preflight_stores(state)
    try:
        filters = apply_recency(None, req.recency)
    except FilterError as e:
        raise HTTPException(400, f"invalid recency: {e}")

    answer, reranked, citations, total_ms, grounding = await _do_rag(
        state, req.query, req.top_k, filters=filters,
        min_score=req.min_score, dedupe_by_paper=req.dedupe_by_paper,
    )
    return schemas.AskResponse(
        query=req.query,
        answer=answer,
        context=[_hit_to_schema(h) for h in reranked],
        citations=citations,
        elapsed_ms=total_ms,
        grounding=grounding,
    )


@app.get("/paper/{arxiv_id}", response_model=schemas.PaperResponse)
async def get_paper(arxiv_id: str, request: Request):
    """Fetch paper metadata + list of chunk_ids."""
    state = request.app.state
    async with state.postgres.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT arxiv_id, title, authors, categories, abstract "
            "FROM papers WHERE arxiv_id = $1", arxiv_id)
        if not row:
            raise HTTPException(404, f"paper not found: {arxiv_id}")
        chunk_rows = await conn.fetch(
            "SELECT chunk_id::text, heading_path FROM chunks "
            "WHERE arxiv_id = $1 ORDER BY chunk_index", arxiv_id)
    abstract_only = (
        len(chunk_rows) == 1
        and (chunk_rows[0]["heading_path"] or "") == "Abstract")
    return schemas.PaperResponse(
        arxiv_id=row["arxiv_id"],
        title=row["title"] or "",
        authors=row["authors"] or [],
        categories=row["categories"] or [],
        abstract=row["abstract"] or "",
        n_chunks=len(chunk_rows),
        chunk_ids=[r["chunk_id"] for r in chunk_rows],
        tier="abstract" if abstract_only else "fulltext",
    )


@app.post("/paper/{arxiv_id}/ask", response_model=schemas.AskResponse)
async def paper_ask(arxiv_id: str, req: schemas.PaperAskRequest,
                    request: Request,
                    _lane: Annotated[None, Depends(_heavy_lane_dep)]):
    """Scoped RAG: answer ONLY from one paper's chunks."""
    state = request.app.state
    await _preflight_stores(state)
    async with state.postgres.pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE arxiv_id = $1", arxiv_id)
    if not n:
        raise HTTPException(404, f"no chunks for arxiv_id: {arxiv_id}")
    answer, reranked, citations, total_ms, grounding = await _do_rag(
        state, req.query, req.top_k, filters={"arxiv_id": arxiv_id},
        min_score=req.min_score, dedupe_by_paper=req.dedupe_by_paper,
    )
    return schemas.AskResponse(
        query=req.query,
        answer=answer,
        context=[_hit_to_schema(h) for h in reranked],
        citations=citations,
        elapsed_ms=total_ms,
        grounding=grounding,
    )


# ─── Retrieval/RAG helpers ─────────────────────────────────────────────────
async def _gather(*aws):
    """asyncio.gather wrapper (kept for testability/mocking)."""
    return await asyncio.gather(*aws)


class _CtxItem:
    """Context item for generation + citation mapping."""
    __slots__ = ("arxiv_id", "chunk_id", "heading_path", "score", "text")

    def __init__(self, text, arxiv_id, chunk_id="", heading_path="", score=None):
        self.text = text
        self.arxiv_id = arxiv_id
        self.chunk_id = chunk_id
        self.heading_path = heading_path
        self.score = score


def _ctx_from_hit(h) -> _CtxItem:
    return _CtxItem(
        text=h.text or "", arxiv_id=h.arxiv_id, chunk_id=h.chunk_id,
        heading_path=h.heading_path or "", score=h.score,
    )


# Retrieval candidate window before rerank.
_RETRIEVAL_WINDOW = 100


async def _single_query_retrieve(state, query, window, filters, top_k):
    """dense+sparse@window -> RRF -> drop boilerplate -> rerank -> top_k."""
    from ..retrieval import rrf_fuse
    fetch = min(window * 3, 500)
    d_hits, s_hits = await _gather(
        state.dense.search(query, top_k=fetch, filters=filters),
        state.sparse.search(query, top_k=fetch, filters=filters),
    )
    fused = rrf_fuse(d_hits, s_hits, k=60, weights=(0.5, 0.5), top_k=fetch)
    fused = _drop_boilerplate_hits(fused)[:window]
    if not fused:
        return []
    # Qdrant payloads omit text; the reranker needs it before scoring.
    await _enrich_hits(state, fused)
    await _ensure_reranker(state)
    return await state.reranker.rerank(
        query, fused, top_k=top_k, candidate_window=2000)


async def _assemble_ctx(state, query: str, top_k: int,
                        filters: dict | None = None,
                        min_score: float | None = None,
                        dedupe_by_paper: bool = False):
    """Retrieve -> rerank -> filter -> context items (None on empty retrieval)."""
    reranked = await _single_query_retrieve(
        state, query, _RETRIEVAL_WINDOW, filters, top_k)
    if min_score is not None:
        reranked = [h for h in reranked if h.score >= min_score]
    if dedupe_by_paper:
        reranked = _dedupe_by_paper(reranked)
    if not reranked:
        return None, []
    await _enrich_titles(state, reranked)
    return [_ctx_from_hit(h) for h in reranked], reranked


async def _do_rag(state, query: str, top_k: int,
                  filters: dict | None = None,
                  min_score: float | None = None,
                  dedupe_by_paper: bool = False):
    """Shared retrieve -> rerank -> generate path.

    Returns (answer, reranked_hits, citations, elapsed_ms, grounding). On empty
    retrieval returns a refusal answer with empty context/citations.
    """
    t_start = time.monotonic()
    ctx_items, reranked = await _assemble_ctx(
        state, query, top_k, filters=filters, min_score=min_score,
        dedupe_by_paper=dedupe_by_paper)
    if ctx_items is None:
        answer = (
            "[no passages retrieved for this query — try rephrasing or "
            "broadening the search]"
        )
        gr = schemas.Grounding(
            confidence="refusal", context_strength=0.0, n_cited=0,
            n_context=0, refusal=True, warnings=["no passages retrieved"])
        return answer, [], [], (time.monotonic() - t_start) * 1000, gr
    try:
        gen = await state.generator.generate(
            query, context_chunks=[it.text for it in ctx_items])
        answer = gen.answer
    except Exception as e:
        answer = (
            f"[generation failed: {type(e).__name__}: {str(e)[:200]}] "
            f"Retrieved {len(reranked)} passages — see context field."
        )
        log.warning("ask_generation_failed", error=str(e)[:200])
    citations = _extract_citations(answer, ctx_items)
    grounding = _compute_grounding(answer, ctx_items, citations)
    return (answer, reranked, citations,
            (time.monotonic() - t_start) * 1000, grounding)


def _extract_citations(answer: str, context_items: list) -> list[schemas.Citation]:
    """Parse [N] / [N, M] markers from an answer into Citation objects."""
    if not answer or not context_items:
        return []
    out: list[schemas.Citation] = []
    seen: set[int] = set()
    for m in _CITE_RE.finditer(answer):
        for tok in re.split(r"[,\s]+", m.group(1)):
            if not tok.isdigit():
                continue
            idx = int(tok)
            if idx in seen or not (1 <= idx <= len(context_items)):
                continue
            seen.add(idx)
            it = context_items[idx - 1]
            aid = it.arxiv_id or ""
            out.append(schemas.Citation(
                index=idx,
                chunk_id=it.chunk_id or "",
                arxiv_id=aid,
                quote=(it.text or "").strip()[:60],
                section=it.heading_path or "",
                abs_url=f"https://arxiv.org/abs/{aid}" if aid else "",
                pdf_url=f"https://arxiv.org/pdf/{aid}" if aid else "",
            ))
    return out


# ─── Grounding heuristic ───────────────────────────────────────────────────
_GROUND_TAU_HI = 0.7
_GROUND_TAU_LO = 0.4
_REFUSAL_MARKERS = (
    "passages do not", "do not specify", "not specified in the",
    "not provided in the", "not contain enough", "does not contain",
    "[no passages retrieved", "no information in the", "cannot answer",
)
_CITE_OF_CTX_RATIO_WARN = 3


def _is_refusal(answer: str) -> bool:
    a = (answer or "").lower().strip()
    return bool(a) and any(m in a for m in _REFUSAL_MARKERS)


def _compute_grounding(answer: str, ctx_items: list,
                       citations: list) -> schemas.Grounding:
    """Composite grounding heuristic (no extra LLM)."""
    n_context = len(ctx_items)
    n_cited = len(citations)
    refusal = _is_refusal(answer)
    cited_items = [ctx_items[c.index - 1] for c in citations
                   if 1 <= c.index <= n_context]
    pool = cited_items if cited_items else ctx_items
    scores = sorted((getattr(it, "score", None) for it in pool),
                    key=lambda s: (s is not None, s), reverse=True)
    top = [s for s in scores if s is not None][:3]
    if top:
        strength = round(
            sum(1.0 / (1.0 + math.exp(-s)) for s in top) / len(top), 3)
    else:
        strength = 0.0
    warnings: list[str] = []
    if refusal:
        confidence = "refusal"
        warnings.append("refusal — retrieved context insufficient for the question")
    elif n_cited == 0:
        confidence = "low"
        warnings.append("answer cites none of the retrieved passages — verify against sources")
    elif strength < _GROUND_TAU_LO:
        confidence = "low"
        warnings.append("low retrieval strength — key evidence may be missing")
    elif strength >= _GROUND_TAU_HI:
        confidence = "high"
    else:
        confidence = "medium"
    if (not refusal and n_cited >= 1 and n_context >= _CITE_OF_CTX_RATIO_WARN
            and n_cited < max(2, n_context // _CITE_OF_CTX_RATIO_WARN)):
        warnings.append(f"answer cites {n_cited} of {n_context} passages")
    return schemas.Grounding(
        confidence=confidence, context_strength=strength,
        n_cited=n_cited, n_context=n_context, refusal=refusal,
        warnings=warnings,
    )


# ─── Exception handlers: JSON 503/504 instead of empty bodies ───────────────
@app.exception_handler(ResponseHandlingException)
async def _qdrant_response_error(request: Request, exc: ResponseHandlingException):
    return JSONResponse(status_code=503,
                        content={"detail": f"qdrant error: {type(exc).__name__}"})


@app.exception_handler(UnexpectedResponse)
async def _qdrant_unexpected(request: Request, exc: UnexpectedResponse):
    return JSONResponse(status_code=503,
                        content={"detail": f"qdrant error: {type(exc).__name__}"})


@app.exception_handler(asyncpg.PostgresError)
async def _pg_error(request: Request, exc: asyncpg.PostgresError):
    return JSONResponse(status_code=503,
                        content={"detail": f"postgres error: {type(exc).__name__}"})


@app.exception_handler(asyncpg.exceptions.InterfaceError)
async def _pg_interface_error(request: Request, exc: asyncpg.exceptions.InterfaceError):
    return JSONResponse(status_code=503,
                        content={"detail": f"postgres error: {type(exc).__name__}"})


@app.exception_handler(httpx.TimeoutException)
async def _http_timeout(request: Request, exc: httpx.TimeoutException):
    return JSONResponse(status_code=504,
                        content={"detail": f"upstream timeout: {type(exc).__name__}"})


@app.exception_handler(httpx.HTTPError)
async def _http_error(request: Request, exc: httpx.HTTPError):
    return JSONResponse(status_code=503,
                        content={"detail": f"upstream error: {type(exc).__name__}"})
