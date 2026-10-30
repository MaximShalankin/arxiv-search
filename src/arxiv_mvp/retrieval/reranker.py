"""Cross-encoder reranker: BGE-Reranker-Large via HF transformers.

- Model: ``BGE_RERANKER_MODEL`` (default BAAI/bge-reranker-large).
- Loaded lazily on first rerank() call; ``ensure_loaded()`` preloads.
- Cross-encoder forward: (query, candidate_text) -> relevance logit.
- max_length=512; candidates are truncated before scoring.
- ``RERANKER_HTTP_URL`` switches to an HTTP reranker (e.g. llama-server
  ``--reranking``) instead of the in-process torch model.

Usage:
    async with Reranker() as rr:
        reranked = await rr.rerank(query, hits, top_k=10)
"""
import asyncio
import os

import httpx

from ..utils.logging import get_logger

log = get_logger("reranker")

DEFAULT_MODEL_ID = "BAAI/bge-reranker-large"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
# dtype: fp32 (default) | bf16 | fp16. Env BGE_RERANKER_DTYPE overrides.
_VALID_DTYPES = ("fp32", "fp16", "bf16")
DEFAULT_DTYPE = os.environ.get("BGE_RERANKER_DTYPE", "fp32").lower()
if DEFAULT_DTYPE not in _VALID_DTYPES:
    raise ValueError(
        f"BGE_RERANKER_DTYPE={DEFAULT_DTYPE!r} invalid; must be one of {_VALID_DTYPES}"
    )


class Reranker:
    """Cross-encoder reranker wrapping BGE-Reranker-Large.

    Loads model lazily on first rerank() call. Caller can use `ensure_loaded()`
    to pre-load (e.g. at API startup) for predictable latency on first query.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "auto",
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: str | None = None,
        dtype: str = DEFAULT_DTYPE,
    ):
        if dtype not in _VALID_DTYPES:
            raise ValueError(
                f"dtype={dtype!r} invalid; must be one of {_VALID_DTYPES}"
            )
        self.model_id = model_id or os.environ.get(
            "BGE_RERANKER_MODEL", DEFAULT_MODEL_ID)
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir or os.environ.get("BGE_RERANKER_CACHE")
        self.dtype = dtype
        self._device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
        # ticket-016 (board serving): HTTP mode — RERANKER_HTTP_URL points at
        # llama-server --reranking (bge-reranker-v2-m3 Q8_0). Replaces in-process
        # torch cross-encoder. NOTE score scale differs from bge-reranker-large:
        # min_score tail-trim must be recalibrated (−4.0 → −1.47) by the caller.
        self._http_url = os.environ.get("RERANKER_HTTP_URL")
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "Reranker":
        await self.ensure_loaded()
        return self

    async def __aexit__(self, *a):
        await self.unload()

    async def ensure_loaded(self) -> None:
        """Load model + tokenizer on GPU if not already loaded."""
        if self._loaded:
            return
        if self._http_url:
            # HTTP mode: no local model — just a client (lazy health-check).
            self._http_client = httpx.AsyncClient(timeout=120.0)
            self._loaded = True
            log.info("reranker_http_mode", url=self._http_url)
            return
        # Heavy imports inside ensure_loaded so module import is cheap
        # (useful for tests that mock Reranker entirely).
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve dtype to a torch constant. CPU only supports fp32 reliably
        # for this model — silently coerce fp16/bf16 → fp32 on CPU.
        if self.dtype == "bf16" and device != "cpu":
            torch_dtype = torch.bfloat16
        elif self.dtype == "fp16" and device != "cpu":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        def _load():
            tok = AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=self.cache_dir,
            )
            mdl = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                torch_dtype=torch_dtype,
            )
            mdl.to(device)
            mdl.eval()
            return tok, mdl, device

        # Tokenizer/model downloads + weights load are blocking — offload
        self._tokenizer, self._model, device = await asyncio.to_thread(_load)
        self._device = device
        self._loaded = True
        n_params = sum(p.numel() for p in self._model.parameters())
        log.info(
            "reranker_loaded",
            model=self.model_id,
            device=device,
            dtype=self.dtype,
            torch_dtype=str(torch_dtype),
            params_m=round(n_params / 1e6, 1),
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

    async def unload(self) -> None:
        """Release model from VRAM."""
        if not self._loaded:
            return
        import torch
        del self._model
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            await asyncio.to_thread(torch.cuda.empty_cache)
        log.info("reranker_unloaded")

    def _score_batch_sync(self, query: str, texts: list[str]) -> list[float]:
        """Score one batch of (query, text) pairs. Blocking — call via to_thread."""
        import torch
        # Pair-tokenize: each entry is (query, text)
        enc = self._tokenizer(
            [query] * len(texts),
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            # Model loaded with torch_dtype=bf16/fp16 handles internal casting
            # itself (embedding lookup accepts Long input_ids, output is bf16).
            # Do NOT cast input_ids — that breaks the embedding lookup
            # ("Expected Long, Int; but got CUDABFloat16Type" error).
            logits = self._model(**enc).logits
            # BGE-Reranker-Large is a single-output regression head.
            # .float() upcast ensures stable math regardless of model dtype.
            scores = logits.squeeze(-1).float().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores

    async def rerank(
        self,
        query: str,
        hits: list,  # list[RetrievalHit]
        top_k: int | None = None,
        candidate_window: int = 200,
    ) -> list:
        """Re-score hits with cross-encoder, return sorted RetrievalHits.

        Args:
            query: user query string
            hits: candidates from hybrid_search (or any retriever)
            top_k: if set, return only top_k after reranking
            candidate_window: truncate each candidate's text to this many chars
                (saves tokens for the cross-encoder — most relevance signal is
                in the first paragraph)

        Returns:
            New list of RetrievalHit (same objects, with score=rerank_score
            and source='reranked', sorted desc by rerank_score).
        """
        if not hits:
            return []
        await self.ensure_loaded()

        if self._http_url:
            return await self._rerank_http(query, hits, top_k, candidate_window)

        # Truncate candidate text to first candidate_window chars (cheap CPU op)
        cand_texts = []
        for h in hits:
            t = h.text or ""
            if len(t) > candidate_window:
                t = t[:candidate_window]
            cand_texts.append(t)

        # Batch + to_thread (each batch is ~10-50 ms on RTX 5090)
        all_scores: list[float] = []
        for i in range(0, len(cand_texts), self.batch_size):
            batch = cand_texts[i:i + self.batch_size]
            scores = await asyncio.to_thread(
                self._score_batch_sync, query, batch,
            )
            all_scores.extend(scores)

        # Attach scores to copies of hits, sort, optionally trim
        from .dense import RetrievalHit  # local import to avoid cycle
        reranked = []
        for h, rs in zip(hits, all_scores):
            reranked.append(RetrievalHit(
                chunk_id=h.chunk_id,
                score=float(rs),
                source="reranked",
                text=h.text,
                arxiv_id=h.arxiv_id,
                heading_path=h.heading_path,
                title=h.title,
                chunk_index=h.chunk_index,
                year=h.year,
                categories=h.categories,
                raw_score=h.raw_score,  # preserve pre-rerank signal
                tier=getattr(h, "tier", "fulltext"),
            ))
        reranked.sort(key=lambda x: -x.score)
        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked

    async def _rerank_http(
        self,
        query: str,
        hits: list,
        top_k: int | None,
        candidate_window: int,
    ) -> list:
        """HTTP mode (ticket-016): llama-server --reranking endpoint.

        Same contract as the in-process path: candidates truncated to
        candidate_window chars, batches of batch_size, relevance_score →
        hit.score. llama-server batch cap is n_batch/n_ubatch on its side;
        chunking by 16 here matches the measured latency profile (2.8s/batch
        on short docs, ~20s on full 2000-char chunks).
        """
        from .dense import RetrievalHit  # local import to avoid cycle

        cand_texts = []
        for h in hits:
            t = h.text or ""
            if len(t) > candidate_window:
                t = t[:candidate_window]
            cand_texts.append(t)

        scores: list[float] = [0.0] * len(cand_texts)
        for i in range(0, len(cand_texts), self.batch_size):
            batch = cand_texts[i:i + self.batch_size]
            body = {"query": query, "documents": batch}
            try:
                r = await self._http_client.post(
                    self._http_url, json=body)
                r.raise_for_status()
                results = r.json().get("results", [])
            except Exception as exc:
                # Single retry; on persistent failure keep zeros (tail-trimmed).
                log.warning(
                    "rerank_http_failed",
                    url=self._http_url,
                    error=f"{type(exc).__name__}",
                )
                r = await self._http_client.post(self._http_url, json=body)
                r.raise_for_status()
                results = r.json().get("results", [])
            for res in results:
                idx = res.get("index")
                if idx is not None and 0 <= idx < len(scores):
                    scores[i + idx] = float(res.get("relevance_score", 0.0))

        reranked = []
        for h, s in zip(hits, scores):
            reranked.append(RetrievalHit(
                chunk_id=h.chunk_id,
                score=s,
                source="reranked",
                text=h.text,
                arxiv_id=h.arxiv_id,
                heading_path=h.heading_path,
                title=h.title,
                chunk_index=h.chunk_index,
                year=h.year,
                categories=h.categories,
                raw_score=h.raw_score,
                tier=getattr(h, "tier", "fulltext"),
            ))
        reranked.sort(key=lambda x: -x.score)
        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked
