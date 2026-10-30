"""Lifespan-scoped resources for the FastAPI app.

Expensive objects live in ``app.state`` and are created at startup:
Embedder (HTTP), Qdrant client, PostgreSQL pool, Reranker, Generator,
DenseRetriever, SparseRetriever.

Every endpoint/model is resolved from the environment at startup and logged.
There are no machine-specific defaults; unset values fall back to localhost
URLs that must still be provided by a running service.
"""
import os
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..ingest.embedder import Embedder
from ..rag import Generator
from ..retrieval import DenseRetriever, Reranker
from ..retrieval.qdrant_native_sparse import QdrantNativeSparseRetriever
from ..store.postgres import PostgresStore
from ..store.qdrant import QdrantStore
from ..utils.logging import get_logger

log = get_logger("api")

DEFAULT_PG_DSN = "postgresql://arxiv:arxiv@localhost:5432/arxiv_mvp"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "arxiv_chunks"


def _redact_dsn(dsn: str) -> str:
    return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", dsn)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: open connections and warm the models. Shutdown: close cleanly."""
    pg_dsn = _env("PG_DSN", DEFAULT_PG_DSN)
    qdrant_url = _env("QDRANT_URL", DEFAULT_QDRANT_URL)
    collection = _env("QDRANT_COLLECTION", DEFAULT_COLLECTION)
    embed_endpoint = _env("BGE_M3_ENDPOINT", "http://localhost:8005/v1/embeddings")
    embed_model = _env("BGE_M3_MODEL", "BAAI/bge-m3")
    llm_url = _env("LLM_URL", "http://localhost:8000/v1/chat/completions")
    llm_model = _env("LLM_MODEL", "Qwen/Qwen3.5-4B")
    rerank_http = _env("RERANKER_HTTP_URL")

    log.info(
        "startup_config",
        pg_dsn=_redact_dsn(pg_dsn),
        qdrant_url=qdrant_url,
        collection=collection,
        embed_endpoint=embed_endpoint,
        embed_model=embed_model,
        llm_url=llm_url,
        llm_model=llm_model,
        reranker_http=rerank_http or "in-process",
    )

    embedder = Embedder(
        endpoint=embed_endpoint,
        model_id=embed_model,
        concurrency=int(_env("EMBED_CONCURRENCY", "2")),
        batch_size=int(_env("EMBED_BATCH_SIZE", "256")),
        timeout_s=float(_env("EMBED_TIMEOUT_S", "120.0")),
    )
    await embedder.__aenter__()

    qdrant = QdrantStore(
        url=qdrant_url, collection_name=collection, vector_name="dense")
    await qdrant.__aenter__()
    # Storage may be busy/restarting; keep the API alive and let endpoints
    # return a clear 503 until it settles.
    try:
        await qdrant.init_collection(recreate=False)
    except Exception as e:
        log.warning("qdrant_init_deferred",
                    error=f"{type(e).__name__}: {str(e)[:200]}")

    pg = PostgresStore(dsn=pg_dsn)
    await pg.__aenter__()
    try:
        await pg.init_schema()
    except Exception as e:
        log.warning("pg_init_deferred",
                    error=f"{type(e).__name__}: {str(e)[:200]}")

    reranker = Reranker()
    reranker_ready = False
    try:
        await reranker.ensure_loaded()
        reranker_ready = True
    except Exception as e:
        log.warning("reranker_unavailable",
                    error=f"{type(e).__name__}: {str(e)[:300]}",
                    hint="install the [rerank] extra or set RERANKER_HTTP_URL")

    generator = Generator(url=llm_url, model=llm_model)
    await generator.__aenter__()

    dense = DenseRetriever(
        embedder=embedder, qdrant=qdrant, top_k=50, vector_name="dense")
    sparse = QdrantNativeSparseRetriever(qdrant=qdrant, top_k=50)

    app.state.embedder = embedder
    app.state.qdrant = qdrant
    app.state.postgres = pg
    app.state.reranker = reranker
    app.state.reranker_ready = reranker_ready
    app.state.generator = generator
    app.state.dense = dense
    app.state.sparse = sparse

    # Best-effort warmups. A failure here is logged, not fatal: the first real
    # request will surface the actual error.
    try:
        t0 = time.monotonic()
        await embedder.embed(["warmup"])
        log.info("embed_warmup_complete",
                 elapsed_ms=round((time.monotonic() - t0) * 1000, 1))
    except Exception as e:
        log.warning("embed_warmup_failed",
                    error=f"{type(e).__name__}: {str(e)[:200]}")
    try:
        await generator.warmup()
    except Exception as e:
        log.warning("llm_warmup_exception",
                    error=f"{type(e).__name__}: {str(e)[:200]}")

    # Corpus statistics: best-effort; /stats serves zeros until a refresh.
    app.state.corpus_stats = {
        "n_papers": 0, "n_chunks": 0,
        "year_range": [None, None], "top_categories": [],
    }
    app.state.corpus_stats_ts = 0.0
    try:
        from .app import _recompute_corpus_stats
        app.state.corpus_stats = await _recompute_corpus_stats(app.state)
        app.state.corpus_stats_ts = time.monotonic()
        log.info("corpus_stats_loaded",
                 n_papers=app.state.corpus_stats["n_papers"],
                 n_chunks=app.state.corpus_stats["n_chunks"])
    except Exception as e:
        log.warning("corpus_stats_failed",
                    error=f"{type(e).__name__}: {str(e)[:200]}")

    log.info("startup_complete")
    try:
        yield
    finally:
        log.info("shutdown_begin")
        await generator.aclose()
        try:
            await reranker.unload()
        except Exception:
            pass
        await pg.__aexit__(None, None, None)
        await qdrant.__aexit__(None, None, None)
        await embedder.__aexit__(None, None, None)
        log.info("shutdown_complete")
