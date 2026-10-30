# Dependencies

Versions below are the declared minimums in `pyproject.toml`; `uv.lock` pins
the exact resolved set. If you do not use uv, install the same packages with
pip and keep a lockfile of your own.

## Python packages

| Package | Extra | Why |
|---|---|---|
| `fastapi`, `uvicorn[standard]` | core | HTTP API |
| `pydantic` | core | request/response schemas |
| `httpx` | core | embedding/LLM/HTTP-reranker clients |
| `asyncpg` | core | PostgreSQL |
| `qdrant-client>=1.12,<1.13` | core | Qdrant dense + sparse |
| `numpy` | core | embedding normalization |
| `PyYAML` | core | ingest config file |
| `python-dotenv` | core | optional `.env` loading |
| `docling` | `[ingest]` | PDF -> markdown for full-text |
| `tiktoken` | `[ingest]` | token counts (word estimate fallback exists) |
| `ijson` | `[ingest]` | stream the 5 GB Kaggle JSON array |
| `torch`, `transformers` | `[rerank]` | in-process BGE-reranker-large |
| `pytest`, `ruff` | `[dev]` | tests and lint |

Install:

```bash
uv sync                    # core API; rerank needs an HTTP reranker
uv sync --extra rerank     # in-process reranker
uv sync --extra ingest     # PDF/Docling ingest
uv sync --all-extras       # everything
```

## External services

| Service | Version | Required for | Notes |
|---|---|---|---|
| PostgreSQL | 16 | all | metadata, chunks, progress; compose included |
| Qdrant | 1.12.x | all | named vectors `dense` + `sparse`; compose included |
| Embeddings | any | dense/hybrid/rerank/ask | OpenAI-compatible `/v1/embeddings`, BGE-M3, 1024-d, cosine |
| LLM | any instruct model | `/ask` | OpenAI-compatible `/v1/chat/completions` |
| Reranker | BGE-reranker-large | `mode=rerank`, `/ask` | in-process via `[rerank]`, or HTTP |

### Embedding endpoint

The service expects a full embeddings URL, not a base URL:

```
BGE_M3_ENDPOINT=http://localhost:8005/v1/embeddings
BGE_M3_MODEL=BAAI/bge-m3
```

`BGE_M3_MODEL` must match the model id the server reports. A common setup is a
vLLM server with `--task embed` or a llama.cpp server with `--embeddings`.
The API does not host or download the embedding model.

### LLM endpoint

```
LLM_URL=http://localhost:8000/v1/chat/completions
LLM_MODEL=Qwen/Qwen3.5-4B
LLM_API_KEY=            # optional bearer token
```

Any OpenAI-compatible instruct model works. The reference deployment uses
Qwen3.5-4B behind vLLM. If your server requires a specific chat-template flag
(e.g. Qwen thinking disabled), set it on the server side; the client sends
plain messages.

### Reranker

In-process:

```bash
uv sync --extra rerank
BGE_RERANKER_MODEL=BAAI/bge-reranker-large
BGE_RERANKER_DTYPE=fp32        # bf16/fp16 on GPU
```

HTTP mode (e.g. `llama-server --reranking`):

```
RERANKER_HTTP_URL=http://localhost:8006/reranking
```

The client sends `{"query": ..., "documents": [...]}` and reads
`results[].relevance_score`. Score scales differ from the in-process model;
recalibrate `min_score` if you use it.

## Reference resource profile

Not a guarantee — measure on your hardware:

- BGE-M3 embeddings: ~2 GB VRAM (vLLM), or CPU/GGUF alternatives.
- BGE-reranker-large: ~1.1 GB VRAM at bf16, ~2.2 GB fp32.
- Qwen3.5-4B: ~13–17 GB VRAM with KV cache.
- Full-text Docling workers: several GB RAM each; do not co-locate with
  Qdrant on a small host.
- Disk: PDFs dominate (hundreds of GB for a large corpus); PG/Qdrant grow with
  chunks. A small abstract-only slice is megabytes.

## Models and licenses

| Model | License | Role |
|---|---|---|
| `BAAI/bge-m3` | MIT | dense embeddings, 1024-d |
| `BAAI/bge-reranker-large` | MIT | cross-encoder rerank |
| Qwen3.5-4B or any instruct model | check the model card | `/ask` generation |

The sparse vectorizer is code, not a model; changing it changes the index.
