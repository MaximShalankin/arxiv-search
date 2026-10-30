"""Docling-based PDF parser wrapped in multiprocessing.Pool.

Design:
- One DocumentConverter per worker process (created in initializer, reused
  for all PDFs assigned to that worker). Saves 30s of converter-init overhead
  per PDF.
- imap_unordered for out-of-order yielding (best throughput under varying
  PDF parse time).
- chunksize=1: each PDF is large work, no benefit from batching tasks.
- Worker failures are isolated: bad PDF returns ParseResult.ok=False,
  does not crash the pool.

Device/OCR configuration (env vars, inherited by fork):
  DOCLING_DEVICE             = "auto" (default, GPU if available) | "cpu" | "cuda"
  DOCLING_DO_OCR             = "true" (default) | "false"
  DOCLING_DO_TABLE_STRUCTURE = "true" (default) | "false"
       ⚠ M2.1 spike (2026-06-17): `false` gives +28% throughput (1.08 → 0.294
       papers/s/worker → 0.294×3 ≈ 0.88 aggregate at 3 workers) with identical
       structural output (same 23.4 headings, 153.1 paragraphs per PDF).
       Conditional ADOPT for M7 — pre-flight gate on 10-20 math-heavy papers
       required before flipping production to `false` (see ticket-002.3 A1).
  DOCLING_NUM_THREADS = "4" (default, CPU thread hint)

⚠ **GPU contention finding (verified 2026-06-17 on 5 PDFs):**
  - 1 worker GPU default: 1.95 s/PDF median (faster than spike baseline!)
  - 4 workers GPU default: 20.4 s/PDF median — GPU contention makes 4
    workers 5× SLOWER aggregate than 1 worker.
  - Production pipeline must use `DOCLING_DEVICE=cpu` so Docling workers
    don't compete with vLLM BGE-M3 (running on GPU during embed phase).
  - `DOCLING_DO_OCR=false` is safe for arxiv (born-digital PDFs); OCR
    fallback for scanned pages is post-MVP unless smoke test shows >5%
    empty parses.

Stdlib + docling only.
"""
import multiprocessing as mp
import os
import signal
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Worker-level hard timeout: SIGALRM delivered inside the forked process kills
# stuck Docling calls (GPU contention, infinite layout loop). Without this, a
# stuck worker holds a Pool slot forever → after N_workers timeouts the entire
# pipeline deadlocks (ticket-007 hang #3, 2026-06-30). The async-level timeout
# in parse_one_async returns an error to the pipeline but cannot un-stick the
# worker process — only an in-process signal can interrupt a spinning C/CUDA
# call. 600 s = 10 min, well above the 99.9th percentile parse time (~60s).
_PARSE_WORKER_TIMEOUT = 600

# Module-level converter for worker processes
_CONVERTER = None


def _build_converter():
    """Construct DocumentConverter per env-configured options."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )

    device = os.environ.get("DOCLING_DEVICE", "auto").lower()
    do_ocr = os.environ.get("DOCLING_DO_OCR", "true").lower() != "false"
    do_table_structure = (
        os.environ.get("DOCLING_DO_TABLE_STRUCTURE", "true").lower() != "false"
    )
    num_threads = int(os.environ.get("DOCLING_NUM_THREADS", "4"))

    pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
    )
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=device,  # Docling accepts "auto" | "cpu" | "cuda" strings
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _init_worker():
    """Pool initializer: build DocumentConverter once per worker."""
    global _CONVERTER
    _CONVERTER = _build_converter()
    print(
        f'[worker] pid={mp.current_process().pid} '
        f'device={os.environ.get("DOCLING_DEVICE", "auto")} '
        f'do_ocr={(os.environ.get("DOCLING_DO_OCR", "true") != "false")} '
        f'do_table_structure={(os.environ.get("DOCLING_DO_TABLE_STRUCTURE", "true") != "false")} '
        f'ready',
        file=sys.stderr, flush=True,
    )


def _parse_one(pdf_path: str) -> dict:
    """Worker function: parse one PDF → markdown.

    Uses SIGALRM as a hard deadline: if Docling hangs (GPU contention /
    infinite layout loop), the signal interrupts the process and converts
    the hang into a clean error. Without this, a stuck worker holds a Pool
    slot forever → after N_workers timeouts the pipeline deadlocks.
    """
    global _CONVERTER
    t0 = time.monotonic()
    try:
        size = os.path.getsize(pdf_path)
    except OSError:
        size = 0

    # Install SIGALRM handler — raises TimeoutError after _PARSE_WORKER_TIMEOUT s
    def _on_timeout(signum, frame):
        raise TimeoutError(
            f"Docling parse timed out after {_PARSE_WORKER_TIMEOUT}s"
        )

    old_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(_PARSE_WORKER_TIMEOUT)
    try:
        result = _CONVERTER.convert(pdf_path)
        md = result.document.export_to_markdown()
        latency = time.monotonic() - t0
        # Suspiciously short markdown suggests scanned PDF without OCR
        # (do_ocr=False path). Flag for caller — treated as soft failure.
        suspicious = len(md.strip()) < 100

        # ticket-002 Phase D RAM leak fix (2026-06-19):
        # Docling accumulates ~3 MB of unreclaimed memory per PDF (likely
        # PIL/torch refs in document state). Without explicit teardown +
        # periodic worker recycling, RSS grows linearly to 38 GB/worker
        # after 13K PDFs (115 GB total → OOM at ~10h). Force release of
        # result.document and its lazily-loaded page artifacts.
        try:
            result.document._pages = None
        except (AttributeError, TypeError):
            pass
        del result
        import gc
        gc.collect()

        return {
            "pdf_path": pdf_path,
            "arxiv_id": Path(pdf_path).stem,
            "markdown": md,
            "ok": True,
            "error": ("suspiciously_short_md" if suspicious else None),
            "latency_s": latency,
            "pdf_bytes": size,
            "markdown_chars": len(md),
        }
    except Exception as e:
        latency = time.monotonic() - t0
        # Even on failure, try to release any partial state Docling built.
        import gc
        gc.collect()
        return {
            "pdf_path": pdf_path,
            "arxiv_id": Path(pdf_path).stem,
            "markdown": None,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "latency_s": latency,
            "pdf_bytes": size,
            "markdown_chars": 0,
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@dataclass
class ParseResult:
    arxiv_id: str
    pdf_path: str
    markdown: str | None
    ok: bool
    error: str | None
    latency_s: float
    pdf_bytes: int
    markdown_chars: int


class ParserPool:
    """Context-managed multiprocessing.Pool of Docling workers."""

    def __init__(self, workers: int = 8):
        self.workers = workers
        self._pool: mp.Pool | None = None

    def __enter__(self) -> "ParserPool":
        # 'fork' is default on Linux — workers inherit venv site-packages
        ctx = mp.get_context("fork")
        self._pool = ctx.Pool(
            processes=self.workers,
            initializer=_init_worker,
            # ticket-002 Phase D RAM leak fix (2026-06-19):
            # Each worker recycles after MAX_TASKS_PER_CHILD PDFs, capping
            # the Docling leak (~3 MB/PDF) at ~3 GB per worker before the
            # process is replaced. Cost: ~5s Docling model reload per recycle
            # × 596K/(3×1000) ≈ 1.4h overhead across full corpus (~1%).
            # Without this, RSS grew linearly to 38 GB/worker at 13K PDFs
            # (Phase D incident 2026-06-19: 115 GB total → OOM at ~10h).
            maxtasksperchild=200,   # ticket-007: был 1000 → 65 GB RSS у воркера (2026-06-30)
                                     # Реальная утечка ~16 MB/paper, не 3 MB. 200 × 16 MB = 3.2 GB cap.
        )
        return self

    def __exit__(self, *a):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    # Async context manager support (used by pipeline.py async with)
    async def __aenter__(self) -> "ParserPool":
        return self.__enter__()

    async def __aexit__(self, *a):
        self.__exit__(*a)

    def parse_many(
        self,
        pdf_paths: Iterable[str],
        chunksize: int = 1,
    ) -> Iterator[ParseResult]:
        if self._pool is None:
            raise RuntimeError("ParserPool not entered (use 'with')")
        for d in self._pool.imap_unordered(_parse_one, list(pdf_paths), chunksize=chunksize):
            yield ParseResult(
                arxiv_id=d["arxiv_id"],
                pdf_path=d["pdf_path"],
                markdown=d["markdown"],
                ok=d["ok"],
                error=d["error"],
                latency_s=d["latency_s"],
                pdf_bytes=d["pdf_bytes"],
                markdown_chars=d["markdown_chars"],
            )

    async def parse_one_async(self, pdf_path: str, timeout: float = 600.0) -> ParseResult:
        """Async wrapper: parse one PDF via the multiprocessing pool.

        Uses apply_async + 20ms polling. For 3s parses this is ~150 polls/GPU
        — negligible overhead. Cleaner than thread-bridging via run_in_executor
        which would serialize through default ThreadPoolExecutor.

        timeout (default 600s): max seconds to wait for the parse. If exceeded,
        returns a ParseResult with ok=False and error="parse_timeout". This
        prevents a hung Docling worker from deadlocking the pipeline (ticket-007
        deadlock 2026-06-29: Docling can spin forever on GPU contention /
        fork+asyncio race).
        """
        import asyncio
        if self._pool is None:
            raise RuntimeError("ParserPool not entered (use 'with')")
        async_result = self._pool.apply_async(_parse_one, (pdf_path,))
        # Async-friendly wait with timeout guard
        n_polls = 0
        max_polls = timeout / 0.02
        while not async_result.ready():
            n_polls += 1
            if n_polls > max_polls:
                arxiv_id = pdf_path.rsplit("/", 1)[-1].replace(".pdf", "")
                return ParseResult(
                    arxiv_id=arxiv_id,
                    pdf_path=pdf_path,
                    markdown="",
                    ok=False,
                    error="parse_timeout",
                    latency_s=timeout,
                    pdf_bytes=0,
                    markdown_chars=0,
                )
            await asyncio.sleep(0.02)
        d = async_result.get()
        return ParseResult(
            arxiv_id=d["arxiv_id"],
            pdf_path=d["pdf_path"],
            markdown=d["markdown"],
            ok=d["ok"],
            error=d["error"],
            latency_s=d["latency_s"],
            pdf_bytes=d["pdf_bytes"],
            markdown_chars=d["markdown_chars"],
        )
