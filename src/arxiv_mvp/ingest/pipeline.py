"""End-to-end pipeline orchestrator: parse → chunk → embed → write.

Design:
- Papers are processed in bounded batches with ``in_flight`` concurrency;
  this avoids materializing hundreds of thousands of asyncio tasks at once.
- Each paper is a self-contained async coroutine (process_paper).
- Stages within process_paper are sequential (parse → chunk → embed → write).
  Recovery state lives in one ``ingestion_progress`` row per paper.
- Effective parallelism:
    parse:   min(in_flight, parser.workers)  (parser-pool bound)
    chunk:   one per active paper            (CPU-cheap)
    embed:   min(active papers, Embedder.concurrency)
    writes:  PG + Qdrant batch operations.
- Bottleneck is normally PDF parsing.

Checkpoint semantics:
- On start, query PG for stage='indexed' AND status='done' → skip set.
- SIGINT/SIGTERM stops new papers; in-flight papers drain (timeout 300s).
- Crashed papers leave status='in_progress' rows in PG; on restart they are
  re-processed from scratch (process_paper is idempotent: paper upsert,
  chunk ON CONFLICT DO NOTHING, Qdrant upsert by point_id). Optional
  PostgresStore.reset_in_progress() clears those rows for tidiness.
- A second SIGINT cancels in-flight tasks immediately.

--retry-failed:
- After the main loop, optionally re-enqueue status='failed' papers.
"""
import asyncio
import os
import signal
import time
from dataclasses import dataclass

from ..utils.logging import get_logger
from ..utils.metrics import ThroughputCounter
from .chunker import chunk_markdown
from .embedder import BGE_M3_MODEL_ID, Embedder
from .parser import ParserPool

log = get_logger("pipeline")

DEFAULT_IN_FLIGHT = 20  # concurrent papers in flight
# max wall-clock per paper (parse+chunk+embed+write)
# ⚠ This wraps the WHOLE process_paper, including time spent waiting for a free
# parser worker. With in_flight >> parser_workers a short timeout fires on
# papers that were never parsed (2026-09-14: in_flight=80 + 6 workers → 78
# spurious timeouts in 20 min). Keep in_flight ≈ 2-3× workers, or widen this.
PAPER_TIMEOUT_S = float(os.environ.get("PAPER_TIMEOUT_S", "300"))
SHUTDOWN_TIMEOUT_S = 300  # max wait for in-flight paper to finish on SIGINT


@dataclass
class PaperResult:
    arxiv_id: str
    ok: bool
    error: str | None = None
    n_chunks: int = 0
    parse_latency_s: float = 0.0
    embed_latency_s: float = 0.0
    write_latency_s: float = 0.0


async def process_paper(
    arxiv_id: str,
    pdf_path: str,
    parser: ParserPool,
    embedder: Embedder,
    pg,  # PostgresStore
    qdrant,  # QdrantStore
    target_tokens: int = 300,
    meta_index: dict | None = None,  # arxiv_id -> {title, authors, categories, abstract}
    write_native_sparse: bool = True,  # write named {"dense":..,"sparse":..} vectors
) -> PaperResult:
    """End-to-end processing of one paper."""

    # Stage 0: upsert paper record (needed for FK constraint on chunks)
    meta = (meta_index or {}).get(arxiv_id, {})
    from ..store.postgres import PaperRecord
    await pg.upsert_paper(PaperRecord(
        arxiv_id=arxiv_id,
        title=meta.get("title", ""),
        authors=meta.get("authors", []),
        categories=meta.get("categories", []),
        abstract=meta.get("abstract", ""),
        submit_date=None,
        journal_ref=None,
        pdf_path=pdf_path,
    ))

    # Stage 1: parse
    try:
        parse_result = await parser.parse_one_async(pdf_path)
    except Exception as e:
        await pg.mark_stage(arxiv_id, "parsed", "failed", error=str(e)[:300])
        return PaperResult(arxiv_id, ok=False, error=f"parse_exc: {e}")

    if not parse_result.ok:
        await pg.mark_stage(arxiv_id, "parsed", "failed", error=parse_result.error)
        return PaperResult(arxiv_id, ok=False, error=parse_result.error,
                          parse_latency_s=parse_result.latency_s)

    if parse_result.error == "suspiciously_short_md":
        log.warning("suspicious_parse", arxiv_id=arxiv_id,
                    md_chars=parse_result.markdown_chars)

    await pg.mark_stage(arxiv_id, "parsed", "in_progress")

    # Stage 2: chunk
    chunks = chunk_markdown(parse_result.markdown, arxiv_id,
                            target_tokens=target_tokens)
    if not chunks:
        err = f"no_chunks (md_chars={parse_result.markdown_chars})"
        await pg.mark_stage(arxiv_id, "chunked", "failed", error=err)
        return PaperResult(arxiv_id, ok=False, error=err,
                          parse_latency_s=parse_result.latency_s)

    await pg.mark_stage(arxiv_id, "chunked", "in_progress")

    # Stage 3: embed
    texts = [c.text for c in chunks]
    embed_t0 = time.monotonic()
    vectors = await embedder.embed_many(texts)
    embed_latency = time.monotonic() - embed_t0

    n_failed = sum(1 for v in vectors if v is None)
    if n_failed == len(vectors):
        err = "all_embeddings_failed"
        await pg.mark_stage(arxiv_id, "embedded", "failed", error=err)
        return PaperResult(arxiv_id, ok=False, error=err,
                          parse_latency_s=parse_result.latency_s,
                          embed_latency_s=embed_latency)

    await pg.mark_stage(arxiv_id, "embedded", "in_progress")

    # Stage 4: write to all three stores (in parallel)
    write_t0 = time.monotonic()
    chunk_rows = []
    for chunk in chunks:
        chunk_rows.append({
            "content_hash_sha256": chunk.content_hash_sha256,
            "text": chunk.text,
            "heading_path": chunk.heading_path,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "embedding_model": BGE_M3_MODEL_ID,
        })

    # PG insert with dedup
    await pg.insert_chunks(arxiv_id, chunk_rows)

    # Query PG for chunk_id of freshly-inserted chunks (by arxiv_id)
    async with pg.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT chunk_id::text, chunk_index, content_hash_sha256, text,
                   heading_path, token_count
            FROM chunks WHERE arxiv_id = $1
            ORDER BY chunk_index
            """,
            arxiv_id,
        )

    # Build a hash → chunk_id map for matching
    hash_to_row = {r["content_hash_sha256"]: r for r in rows}

    # Match chunks to vectors via SHA (only first occurrence per hash in this paper)
    seen_hashes = set()
    qdrant_items = []
    for chunk, vector in zip(chunks, vectors):
        if vector is None:
            continue
        if chunk.content_hash_sha256 in seen_hashes:
            continue
        r = hash_to_row.get(chunk.content_hash_sha256)
        if r is None:
            continue  # aliased to existing chunk, skip Qdrant write
        seen_hashes.add(chunk.content_hash_sha256)

        # Named vectors: dense (BGE-M3) + sparse (hashed-unigram). Sparse is
        # computed with the same encoder the retriever uses at query time.
        if write_native_sparse:
            from ..retrieval.qdrant_native_sparse import encode_sparse
            sp_indices, sp_values = encode_sparse(chunk.text)
            vec_struct: dict = {"dense": vector}
            if sp_indices:
                vec_struct["sparse"] = {"indices": sp_indices, "values": sp_values}
        else:
            vec_struct = vector

        qdrant_items.append({
            "point_id": r["chunk_id"],
            "vector": vec_struct,
            "payload": {
                # Text is hydrated from PG by the API before rerank/response,
                # so the payload intentionally does not duplicate chunk text.
                "arxiv_id": arxiv_id,
                "chunk_id": r["chunk_id"],
                "chunk_index": chunk.chunk_index,
                "heading_path": chunk.heading_path,
                "token_count": chunk.token_count,
                "content_hash_sha256": chunk.content_hash_sha256,
                "title": meta.get("title", ""),
                "year": _year_from_arxiv_id(arxiv_id),
                "categories": meta.get("categories", []),
            },
        })

    writes = []
    # Batch Qdrant upserts: a pathological paper with thousands of chunks must
    # not turn one failed request into a half-written paper.
    QDRANT_UPSERT_BATCH = 256
    for start in range(0, len(qdrant_items), QDRANT_UPSERT_BATCH):
        writes.append(
            qdrant.upsert_chunks(qdrant_items[start:start + QDRANT_UPSERT_BATCH]))
    if writes:
        await asyncio.gather(*writes)

    write_latency = time.monotonic() - write_t0

    # Stage 5: mark done
    await pg.mark_stage(arxiv_id, "indexed", "done")

    return PaperResult(
        arxiv_id=arxiv_id,
        ok=True,
        n_chunks=len(chunks),
        parse_latency_s=parse_result.latency_s,
        embed_latency_s=embed_latency,
        write_latency_s=write_latency,
    )


def _year_from_arxiv_id(arxiv_id: str) -> int | None:
    """0704.xxx → 2007, 2310.xxx → 2023. None for legacy 'cs.AI_xxx'."""
    try:
        yymm = arxiv_id.split(".")[0]
        yy = int(yymm[:2])
        return (1900 + yy) if yy > 80 else (2000 + yy)
    except (ValueError, IndexError):
        return None


async def run_pipeline(
    paper_list: list[tuple[str, str]],  # (arxiv_id, pdf_path)
    *,
    parser_workers: int = 3,
    embedder_concurrency: int = 2,
    embedder_batch_size: int = 256,
    in_flight: int = DEFAULT_IN_FLIGHT,
    target_tokens: int = 300,
    pg_dsn: str = "postgresql://arxiv:arxiv@localhost:5432/arxiv_mvp",
    qdrant_url: str = "http://localhost:6333",
    qdrant_collection: str = "arxiv_chunks",
    recreate_qdrant: bool = False,
    write_native_sparse: bool = True,   # write named dense+sparse vectors
    skip_done: bool = True,
    reset_in_progress: bool = False,
    install_signal_handlers: bool = False,
    progress_every: int = 50,
    meta_index: dict | None = None,
) -> dict:
    """Run end-to-end pipeline. Returns summary stats.

    Checkpoint behavior (ticket-002.3 Phase B):
    - SIGINT/SIGTERM triggers graceful shutdown: stop accepting new papers,
      drain in-flight with SHUTDOWN_TIMEOUT_S (300s) hard timeout.
    - Second SIGINT cancels everything immediately.
    - On restart, skip_done=True excludes papers with stage='indexed',
      status='done'. Crashed papers (status='in_progress') are re-processed
      from scratch — process_paper() is idempotent.
    - reset_in_progress=True explicitly clears in_progress rows on start
      (cosmetic — they'd be reprocessed anyway, but PG stays clean).
    - install_signal_handlers=True installs SIGINT/SIGTERM handlers for
      graceful shutdown. Default False because handler installation
      interacts poorly with fork-based Docling workers (pipeline hangs
      after first batch — under investigation 2026-06-18). When False,
      Ctrl+C still works via KeyboardInterrupt → hard cancel.
    """
    from ..store.postgres import PostgresStore
    from ..store.qdrant import QdrantStore

    overall_t0 = time.monotonic()
    counter = ThroughputCounter("pipeline")

    # Filter out already-done papers (resume) + optional reset
    async with PostgresStore(dsn=pg_dsn) as pg_check:
        await pg_check.init_schema()
        if reset_in_progress:
            n_reset = await pg_check.reset_in_progress()
            log.info("reset_in_progress_on_start", cleared=n_reset)
        if skip_done:
            done_ids = await pg_check.get_done_arxiv_ids()
            before = len(paper_list)
            paper_list = [(a, p) for a, p in paper_list if a not in done_ids]
            log.info("resume_filter", before=before, after=len(paper_list),
                     skipped=before - len(paper_list))

    if not paper_list:
        log.info("nothing_to_do")
        return {
            "papers_done": 0, "papers_failed": 0,
            "total_chunks": 0, "wall_s": 0, "rate_per_s": 0.0,
            "stopped_by_signal": False,
        }

    # Enter long-lived store connections for the whole pipeline run.
    from contextlib import AsyncExitStack
    async with AsyncExitStack() as _stack:
        pg = await _stack.enter_async_context(PostgresStore(dsn=pg_dsn))
        qd = await _stack.enter_async_context(
            QdrantStore(url=qdrant_url, collection_name=qdrant_collection))
        emb = await _stack.enter_async_context(
            Embedder(concurrency=embedder_concurrency,
                     batch_size=embedder_batch_size))
        parser = await _stack.enter_async_context(ParserPool(workers=parser_workers))

        # Init stores
        await pg.init_schema()
        await qd.init_collection(recreate=recreate_qdrant)

        log.info("pipeline_started",
                 papers=len(paper_list),
                 in_flight=in_flight,
                 parser_workers=parser_workers,
                 embedder_concurrency=embedder_concurrency,
                 embedder_batch_size=embedder_batch_size)

        # ─── Checkpoint: signal-driven graceful shutdown ───────────────
        # stop_event set by SIGINT/SIGTERM handler. Producer checks before
        # each enqueue; consumers finish current paper then exit.
        stop_event = asyncio.Event()
        force_stop_event = asyncio.Event()  # second signal → hard cancel

        def _signal_handler(name):
            log.warning("signal_received", signal=name)
            if stop_event.is_set():
                # Second signal → force stop
                log.warning("signal_second_forcing_hard_stop", signal=name)
                force_stop_event.set()
            else:
                stop_event.set()

        # Register handlers (restore on exit so nested runs don't stack).
        # Disabled by default — installing signal handlers from inside an
        # async function appears to interact poorly with fork-based Docling
        # worker startup, causing the pipeline to hang after the first batch
        # of papers (ticket-002.3 Phase B bug, 2026-06-18). Re-enable via
        # install_signal_handlers=True for production runs once root cause
        # is isolated. SIGINT still works via KeyboardInterrupt → asyncio
        # cancellation, just less gracefully.
        old_handlers: dict = {}
        if install_signal_handlers:
            for sig in (signal.SIGINT, signal.SIGTERM):
                sig_name = sig.name  # capture in closure
                try:
                    old_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, lambda s, f, _n=sig_name: _signal_handler(_n))
                except (ValueError, OSError) as e:
                    # ValueError = not main thread; OSError = not supported
                    log.warning("signal_handler_not_installed",
                                signal=sig_name, error=str(e))

        try:
            # ─── Streaming task dispatch ─────────────────────────────────
            # Process papers in chunks of `task_batch` to bound memory.
            # Within a chunk: create all tasks upfront, gate via Semaphore.
            # Between chunks: check stop_event, log progress.
            #
            # Earlier attempt used asyncio.Queue + producer/consumer pattern
            # but it hung after `in_flight` papers under fork+asyncio (root
            # cause not isolated, see Phase B bug note 2026-06-18). This
            # semaphore + chunked pattern is the proven ticket-002.1 design
            # and works correctly.
            sem = asyncio.Semaphore(in_flight)
            task_batch = max(in_flight * 4, 100)  # bounded memory per chunk

            async def bounded_process(arxiv_id, pdf_path):
                async with sem:
                    if stop_event.is_set() or force_stop_event.is_set():
                        return None
                    try:
                        r = await asyncio.wait_for(
                            process_paper(
                                arxiv_id, pdf_path, parser, emb, pg, qd,
                                target_tokens=target_tokens,
                                meta_index=meta_index,
                                write_native_sparse=write_native_sparse,
                            ),
                            timeout=PAPER_TIMEOUT_S,
                        )
                        counter.add(1)
                        return r
                    except TimeoutError:
                        log.error("paper_timed_out", arxiv_id=arxiv_id,
                                  timeout_s=PAPER_TIMEOUT_S)
                        await pg.mark_stage(arxiv_id, "indexed", "failed",
                                            error=f"timeout_{PAPER_TIMEOUT_S}s")
                        return PaperResult(arxiv_id, ok=False,
                                          error=f"timeout_{PAPER_TIMEOUT_S}s")
                    except Exception as e:
                        log.error("paper_crashed", arxiv_id=arxiv_id, error=str(e)[:300])
                        await pg.mark_stage(arxiv_id, "indexed", "failed",
                                            error=f"unhandled: {type(e).__name__}: {e}"[:300])
                        return PaperResult(arxiv_id, ok=False, error=str(e)[:300])

            done = 0
            failed = 0
            total_chunks = 0
            last_log = time.monotonic()
            _lat = {"parse": 0.0, "embed": 0.0, "write": 0.0, "n": 0}

            for chunk_start in range(0, len(paper_list), task_batch):
                if stop_event.is_set() or force_stop_event.is_set():
                    log.info("pipeline_stopping_between_chunks",
                             completed=done + failed,
                             remaining=len(paper_list) - chunk_start)
                    break

                chunk = paper_list[chunk_start:chunk_start + task_batch]
                tasks = [
                    asyncio.create_task(bounded_process(a, p)) for a, p in chunk
                ]

                for fut in asyncio.as_completed(tasks):
                    r = await fut
                    if r is None:
                        continue  # skipped due to stop_event
                    if r.ok:
                        done += 1
                        total_chunks += r.n_chunks
                        # Stage-latency means make throughput regressions
                        # diagnosable from the log alone (2026-09-14: needed to
                        # tell parse vs embed vs write apart during the 2609 run).
                        _lat["parse"] += r.parse_latency_s
                        _lat["embed"] += r.embed_latency_s
                        _lat["write"] += r.write_latency_s
                        _lat["n"] += 1
                    else:
                        failed += 1
                        log.warning("paper_failed",
                                    arxiv_id=r.arxiv_id, error=r.error)

                    # Periodic progress
                    n = done + failed
                    if n % progress_every == 0 or time.monotonic() - last_log > 30:
                        elapsed = time.monotonic() - overall_t0
                        rate = n / max(elapsed, 1e-9)
                        log.info("progress",
                                 done=done, failed=failed, total=len(paper_list),
                                 elapsed_s=round(elapsed, 1),
                                 rate_per_s=round(rate, 3),
                                 eta_s=round((len(paper_list) - n) / max(rate, 1e-9), 0),
                                 mean_parse_s=round(_lat["parse"] / _lat["n"], 2) if _lat["n"] else None,
                                 mean_embed_s=round(_lat["embed"] / _lat["n"], 2) if _lat["n"] else None,
                                 mean_write_s=round(_lat["write"] / _lat["n"], 2) if _lat["n"] else None)
                        last_log = time.monotonic()

        finally:
            # Restore signal handlers
            for sig, oh in old_handlers.items():
                try:
                    signal.signal(sig, oh)
                except (ValueError, OSError):
                    pass

        if stop_event.is_set() or force_stop_event.is_set():
            log.warning("pipeline_stopped_by_signal",
                        done=done, failed=failed, remaining=len(paper_list) - done - failed)
        else:
            log.info("pipeline_complete_naturally",
                     done=done, failed=failed)

    wall = time.monotonic() - overall_t0
    summary = {
        "papers_done": done,
        "papers_failed": failed,
        "total_chunks": total_chunks,
        "wall_s": wall,
        "rate_per_s": (done + failed) / max(wall, 1e-9),
        "stopped_by_signal": stop_event.is_set(),
    }
    log.info("pipeline_complete", **summary)
    return summary
