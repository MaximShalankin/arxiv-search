#!/usr/bin/env python3
"""Full-text ingest runner: parse → chunk → embed → write (PG + Qdrant).

Usage:
    python scripts/run_fulltext_ingest.py \\
        --config configs/example.yaml [--limit N] [--retry-failed] \\
        [--reset-in-progress]

Behavior:
- Loads a YAML config (paths and endpoints are explicit; no machine defaults).
- Reads the ID file ``subsample_path`` (one arXiv ID per line).
- Resolves each ID to ``{pdf_root}/{YYMM}/{id}.pdf`` (or ``{id}v*.pdf``).
- Filters out already-done papers (stage='indexed' AND status='done').
- Runs the pipeline: parse → chunk → embed → write.
- ``--retry-failed`` re-enqueues papers with status='failed'.
- ``--reset-in-progress`` clears status='in_progress' rows before starting.

Checkpoint behavior:
- SIGINT/SIGTERM stops new papers; in-flight papers drain (300s timeout).
  Re-run the same command to resume; completed papers are skipped.
- A second SIGINT cancels in-flight tasks immediately.

Exit code: 0 if all papers succeeded, 1 if any failed.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

import yaml

from arxiv_mvp.ingest.pipeline import run_pipeline
from arxiv_mvp.store.postgres import PostgresStore
from arxiv_mvp.utils.logging import get_logger

log = get_logger("runner")


async def _load_metadata_index(meta_path: str) -> dict:
    """Load arxiv_id -> {title, authors, categories, abstract} from JSON.
    Runs in thread to not block event loop on 1.9 GB parse."""
    import asyncio
    import json
    import re

    if not meta_path:
        log.warning("metadata_path_empty")
        return {}

    # arXiv OAI authors field is comma-separated with optional " and " before
    # the last author (bug-001 A3): "Xu Feng, Wenjian Yu, and Jie Tang" or
    # "Huayang Li and Yixuan Su and Deng Cai". Split on both separators.
    # Also normalizes internal whitespace (stray newlines in some records).
    _AUTHOR_SPLIT_RE = re.compile(r"\s*,\s*|\s+and\s+")
    _WS_NORMALIZE_RE = re.compile(r"\s+")

    def _parse_authors(raw):
        if isinstance(raw, list):
            out = []
            for a in raw:
                if not isinstance(a, str):
                    continue
                norm = _WS_NORMALIZE_RE.sub(" ", a).strip()
                if norm:
                    out.append(norm)
            return out
        if not isinstance(raw, str) or not raw.strip():
            return []
        return [
            _WS_NORMALIZE_RE.sub(" ", p).strip()
            for p in _AUTHOR_SPLIT_RE.split(raw)
            if p.strip()
        ]

    def _load():
        with open(meta_path) as f:
            data = json.load(f)
        out = {}
        for r in data:
            aid = r.get("id", "").strip()
            if not aid:
                continue
            out[aid] = {
                "title": (r.get("title") or "").strip(),
                "authors": _parse_authors(r.get("authors", "")),
                "categories": (r.get("categories") or "").split(),
                "abstract": (r.get("abstract") or "").strip(),
            }
        return out

    return await asyncio.to_thread(_load)


def resolve_pdf_path(arxiv_id: str, pdfs_root: Path) -> str | None:
    """Find PDF for arxiv_id. Bucket = first 4 chars (YYMM)."""
    if "_" in arxiv_id:
        return None
    bucket = arxiv_id[:4] if "." in arxiv_id else None
    if bucket is None or not bucket.isdigit():
        return None
    bucket_dir = pdfs_root / bucket
    if not bucket_dir.is_dir():
        return None
    exact = bucket_dir / f"{arxiv_id}.pdf"
    if exact.exists():
        return str(exact)
    for p in bucket_dir.glob(f"{arxiv_id}v*.pdf"):
        return str(p)
    return None


async def amain(cfg: dict, limit: int | None, retry_failed: bool,
                reset_in_progress: bool, max_papers: int | None):
    # Apply env vars for parser subprocess (must be set before Pool init)
    parser_env = cfg.get("pipeline", {}).get("parser", {}).get("env", {})
    for k, v in parser_env.items():
        os.environ[k] = str(v)
        log.info("env_set", key=k, value=str(v))

    # Map the config's embedder endpoint/model into the env the Embedder reads.
    # Explicit config wins over the generic localhost default.
    embedder_env = cfg.get("pipeline", {}).get("embedder", {})
    if embedder_env.get("endpoint"):
        os.environ["BGE_M3_ENDPOINT"] = str(embedder_env["endpoint"])
        log.info("embedder_endpoint_set", endpoint=str(embedder_env["endpoint"]))
    if embedder_env.get("model_id"):
        os.environ["BGE_M3_MODEL"] = str(embedder_env["model_id"])
        log.info("embedder_model_set", model=str(embedder_env["model_id"]))

    # Paths are explicit in the config: no machine-specific defaults.
    for key in ("subsample_path", "pdf_root", "metadata_path"):
        if not cfg.get(key):
            raise SystemExit(f"config missing required key: {key}")

    # Read subsample
    sample_path = Path(cfg["subsample_path"])
    with sample_path.open() as f:
        all_ids = [line.strip() for line in f if line.strip()]
    log.info("loaded_subsample", total=len(all_ids), path=str(sample_path))

    # Load metadata index once (JSON array; keep it filtered to the target set)
    meta_index = await _load_metadata_index(cfg["metadata_path"])
    log.info("metadata_index_loaded", n=len(meta_index))

    pdfs_root = Path(cfg["pdf_root"])
    paper_list = []
    missing = 0
    for aid in all_ids:
        p = resolve_pdf_path(aid, pdfs_root)
        if p is None:
            missing += 1
        else:
            paper_list.append((aid, p))
    log.info("resolved_pdfs", found=len(paper_list), missing=missing)
    if missing:
        log.warning("missing_pdfs_skipped", n=missing)

    # Apply limit
    if limit is not None:
        paper_list = paper_list[:limit]
        log.info("limit_applied", n=len(paper_list))

    # Apply --max-papers (separate from --limit for checkpoint testing:
    # --limit caps how many to resolve; --max-papers caps how many to
    # actually enqueue. Useful for "parse 100, stop, restart, see it resume".
    if max_papers is not None:
        paper_list = paper_list[:max_papers]
        log.info("max_papers_applied", n=len(paper_list))

    # --retry-failed: prepend failed arxiv_ids (resolved to PDFs)
    if retry_failed:
        pg_dsn = cfg["storage"]["postgres_dsn"]
        async with PostgresStore(dsn=pg_dsn) as pg:
            failed_ids = await pg.get_failed_arxiv_ids()
        log.info("retry_failed_found", n=len(failed_ids))
        retry_papers = []
        for aid in failed_ids:
            p = resolve_pdf_path(aid, pdfs_root)
            if p is not None:
                # Clear failed status so pipeline will retry
                retry_papers.append((aid, p))
        # Retry-list goes to front
        paper_list = retry_papers + [p for p in paper_list
                                     if p[0] not in {a for a, _ in retry_papers}]
        log.info("retry_failed_added", n=len(retry_papers),
                 total_to_process=len(paper_list))

    if not paper_list:
        log.info("nothing_to_process")
        return 0

    # Run pipeline
    p = cfg["pipeline"]
    s = cfg["storage"]
    summary = await run_pipeline(
        paper_list=paper_list,
        parser_workers=p["parser"]["workers"],
        embedder_concurrency=p["embedder"]["concurrency"],
        embedder_batch_size=p["embedder"]["batch_size"],
        in_flight=p.get("in_flight", 20),
        target_tokens=p["chunker"]["target_tokens"],
        progress_every=p.get("progress_every", 50),
        pg_dsn=s["postgres_dsn"],
        qdrant_url=s["qdrant_url"],
        qdrant_collection=s["qdrant_collection"],
        recreate_qdrant=s.get("recreate_qdrant", False),
        write_native_sparse=s.get("write_native_sparse", True),
        reset_in_progress=reset_in_progress,
        meta_index=meta_index,
    )

    print("\n" + "=" * 60, flush=True)
    print("PIPELINE RUN COMPLETE", flush=True)
    print("=" * 60, flush=True)
    print(f"papers_done:    {summary['papers_done']}", flush=True)
    print(f"papers_failed:  {summary['papers_failed']}", flush=True)
    print(f"total_chunks:   {summary['total_chunks']:,}", flush=True)
    print(f"wall:           {summary['wall_s']:.1f}s", flush=True)
    print(f"rate:           {summary['rate_per_s']:.3f} papers/s", flush=True)
    if summary.get("stopped_by_signal"):
        print("stopped_by_signal: True (resume with same command)", flush=True)
    print("=" * 60, flush=True)

    # Exit 0 on natural completion with no failures OR on signal-driven
    # shutdown (so callers can detect "stopped, not failed" via JSON log).
    if summary.get("stopped_by_signal"):
        return 0
    return 0 if summary["papers_failed"] == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--limit", type=int, default=None,
                    help="resolve only first N papers (for smoke tests)")
    ap.add_argument("--max-papers", type=int, default=None,
                    help="enqueue at most N papers after resume filter "
                         "(separate from --limit; useful for checkpoint testing)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also re-enqueue papers with status='failed'")
    ap.add_argument("--reset-in-progress", action="store_true",
                    help="clear status='in_progress' rows before starting "
                         "(pipeline is idempotent, but this keeps PG tidy)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log.info("config_loaded", path=args.config)

    return asyncio.run(amain(cfg, args.limit, args.retry_failed,
                              args.reset_in_progress, args.max_papers))


if __name__ == "__main__":
    sys.exit(main())
