"""Async embedder for an OpenAI-compatible BGE-M3 endpoint.

Contract:
- Endpoint: ``BGE_M3_ENDPOINT`` (default http://localhost:8005/v1/embeddings)
- Model id: ``BGE_M3_MODEL`` (default BAAI/bge-m3); must match the served name.
- Concurrency=2 (Semaphore), batch size=256 by default.
- normalize_embeddings=True on client (BGE-M3 not unit-norm; vLLM may or may
  not normalize depending on version — explicit normalize on client is the
  contract). Qdrant Distance.COSINE provides a second normalization safety net.
- Zero-norm guard: chunks with ||v|| < 1e-12 are skipped + logged (junk parse).
- Retry strategy:
    3 retries on same batch_size with backoff 1s/2s/5s (transient HTTP errors)
    → ONE split to batch/2, min 64, 2 retries  (persistent failures)
    → fail (raise EmbedError)

Usage:
    async with Embedder() as emb:
        vectors = await emb.embed(["text1", "text2", ...])
"""
import asyncio
import os
import time
from dataclasses import dataclass

import httpx
import numpy as np

from ..utils.logging import get_logger

log = get_logger("embedder")

BGE_M3_MODEL_ID = "BAAI/bge-m3"
FALLBACK_ENDPOINT = "http://localhost:8005/v1/embeddings"
# Kept for callers that import the constant; prefer resolve_endpoint().
DEFAULT_ENDPOINT = os.environ.get("BGE_M3_ENDPOINT", FALLBACK_ENDPOINT)


def resolve_endpoint(explicit: str | None = None) -> str:
    """Endpoint precedence: explicit arg > BGE_M3_ENDPOINT (read now) > default."""
    return explicit or os.environ.get("BGE_M3_ENDPOINT") or FALLBACK_ENDPOINT


def resolve_model(explicit: str | None = None) -> str:
    """Model precedence: explicit arg > BGE_M3_MODEL (read now) > default."""
    return explicit or os.environ.get("BGE_M3_MODEL") or BGE_M3_MODEL_ID


DEFAULT_CONCURRENCY = 2
DEFAULT_BATCH_SIZE = 256
ZERO_NORM_EPS = 1e-12

# Defensive cap: vLLM rejects inputs > 131072 chars with 400 Bad Request.
# Chunker is supposed to enforce 24K cap, but this guard catches regressions
# and protects the entire batch from one pathological chunk. Cap at 100K
# leaves comfortable margin.
_EMBED_HARD_CHAR_CAP = 100_000


class EmbedError(RuntimeError):
    pass


@dataclass
class EmbedResult:
    """Result of embedding a batch of texts.

    `vectors[i]` is None for texts that produced zero-norm embeddings (junk
    parse) or failed all retries.
    """
    vectors: list[list[float] | None]
    n_ok: int
    n_skipped: int
    n_failed: int
    latency_s: float


class Embedder:
    """Async client for vLLM BGE-M3 endpoint."""

    def __init__(
        self,
        endpoint: str | None = None,
        model_id: str | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        # 600s allows slow CPU/ARM embedding servers to finish a large batch.
        timeout_s: float = 600.0,
    ):
        self.endpoint = resolve_endpoint(endpoint)
        self.model_id = resolve_model(model_id)
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self._sem = asyncio.Semaphore(concurrency)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Embedder":
        self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self

    async def __aexit__(self, *a):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Embedder not entered (use 'async with')")
        return self._client

    async def _post_once(self, texts: list[str]) -> list[list[float]]:
        """Single POST. Returns list of raw (unnormalized) embedding vectors.
        Raises httpx.HTTPStatusError / httpx.RequestError on failure."""
        body = {
            "model": self.model_id,
            "input": texts,
            "encoding_format": "float",
        }
        resp = await self.client.post(self.endpoint, json=body)
        resp.raise_for_status()
        data = resp.json()
        # Sort by index in case vLLM reorders
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in items]

    async def _post_with_retries(self, texts: list[str]) -> list[list[float]]:
        """Retry policy: 3 attempts on same batch_size with backoff 1/2/5s,
        then ONE split to batch/2 with 2 more retries. Then raise EmbedError."""
        # Phase 1: 3 retries on full batch
        last_err = None
        for attempt, backoff in enumerate([1.0, 2.0, 5.0], start=1):
            try:
                t0 = time.monotonic()
                result = await self._post_once(texts)
                if attempt > 1:
                    log.info("retry_recovered", n=len(texts),
                             attempt=attempt, latency_s=round(time.monotonic() - t0, 3))
                return result
            except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
                last_err = e
                log.warning("embed_attempt_failed", n=len(texts),
                            attempt=attempt, error=str(e)[:200])
                await asyncio.sleep(backoff)

        # Phase 2: split batch in half (min 64), 2 retries
        if len(texts) >= 128:
            half = max(64, len(texts) // 2)
            log.warning("embed_splitting", n=len(texts), half=half)
            halves = [texts[:half], texts[half:half * 2]]
            if len(texts) > half * 2:
                halves.append(texts[half * 2:])
            gathered = await asyncio.gather(
                *[self._post_with_retries(h) for h in halves],
                return_exceptions=True,
            )
            combined: list[list[float]] = []
            for g in gathered:
                if isinstance(g, Exception):
                    raise EmbedError(f"split-half failed: {g}") from g
                combined.extend(g)
            return combined

        raise EmbedError(f"all retries exhausted: {last_err}")

    async def embed(self, texts: list[str]) -> EmbedResult:
        """Embed a list of texts. Length must be ≤ batch_size (caller's responsibility).
        Returns EmbedResult with per-text vectors (None for zero-norm or failed)."""
        if not texts:
            return EmbedResult(vectors=[], n_ok=0, n_skipped=0, n_failed=0, latency_s=0.0)
        if len(texts) > self.batch_size:
            raise ValueError(f"batch len {len(texts)} > batch_size {self.batch_size}")

        # Defensive size guard: truncate any pathological chunk to protect the
        # batch. Chunker should already cap at 24K chars; this is a backstop.
        truncated_texts: list[str] = []
        for i, t in enumerate(texts):
            if len(t) > _EMBED_HARD_CHAR_CAP:
                log.warning("embed_text_truncated",
                            idx=i, orig_chars=len(t),
                            capped_chars=_EMBED_HARD_CHAR_CAP)
                truncated_texts.append(t[:_EMBED_HARD_CHAR_CAP] + "\n[...truncated...]")
            else:
                truncated_texts.append(t)
        texts = truncated_texts

        t0 = time.monotonic()
        async with self._sem:
            try:
                raw_vectors = await self._post_with_retries(texts)
            except EmbedError as e:
                log.error("embed_failed_hard", n=len(texts), error=str(e)[:200])
                return EmbedResult(
                    vectors=[None] * len(texts),
                    n_ok=0, n_skipped=0, n_failed=len(texts),
                    latency_s=time.monotonic() - t0,
                )

        # Normalize + zero-norm guard
        vectors: list[list[float] | None] = []
        n_ok = 0
        n_skipped = 0
        for text, raw in zip(texts, raw_vectors):
            v = np.asarray(raw, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            if norm < ZERO_NORM_EPS:
                log.warning("zero_norm_chunk", preview=text[:80])
                vectors.append(None)
                n_skipped += 1
                continue
            # L2 normalize → cosine = dot product
            v_norm = (v / norm).tolist()
            vectors.append(v_norm)
            n_ok += 1

        return EmbedResult(
            vectors=vectors,
            n_ok=n_ok,
            n_skipped=n_skipped,
            n_failed=0,
            latency_s=time.monotonic() - t0,
        )

    async def embed_many(
        self,
        texts: list[str],
    ) -> list[list[float] | None]:
        """Embed an arbitrary-size list, batching internally.
        Dispatches up to `concurrency` batches at once (Semaphore-gated)."""
        if not texts:
            return []
        batches = [texts[i:i + self.batch_size]
                   for i in range(0, len(texts), self.batch_size)]
        # Run batches with overall concurrency limit (Semaphore is inside embed())
        results = await asyncio.gather(*[self.embed(b) for b in batches])
        out: list[list[float] | None] = []
        for r in results:
            out.extend(r.vectors)
        return out
