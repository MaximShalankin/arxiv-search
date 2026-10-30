# API

Base URL in examples: `http://127.0.0.1:8080`. All bodies are JSON.
Interactive schema: `/docs`, `/redoc`, `/openapi.json`.

## GET `/health`

```json
{
  "status": "ok",
  "postgres": "ok",
  "qdrant": "ok",
  "embedder": "ok",
  "reranker_loaded": true,
  "version": "0.1.0"
}
```

`postgres`/`qdrant` are `"ok"` or `"error: <ExceptionName>"`.
`reranker_loaded` is false when the `[rerank]` extra is not installed and no
HTTP reranker is configured.

## GET `/stats`

Cached corpus counts; a background refresh runs every 5 minutes.

```json
{
  "n_papers": 2000,
  "n_chunks": 2000,
  "year_range": [2024, 2024],
  "top_categories": [{"category": "cs.CL", "n_papers": 2000}]
}
```

## POST `/search`

```json
{
  "query": "retrieval augmented generation evaluation",
  "mode": "hybrid",
  "top_k": 10,
  "top_k_hybrid": 50,
  "filters": null,
  "recency": null,
  "dedupe_by_paper": false
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | required | 1–2000 chars |
| `mode` | enum | `rerank` | `bm25`/`sparse`, `dense`, `hybrid`, `rerank` |
| `top_k` | int | 10 | 1–100 |
| `top_k_hybrid` | int | 50 | candidate pool before RRF/rerank |
| `filters` | object/null | null | exact, list, or range |
| `recency` | enum/null | null | `last_12m` / `last_36m` / `all` |
| `min_score` | float/null | null | rerank mode only: drop raw reranker scores below this |
| `dedupe_by_paper` | bool | false | keep the best chunk per arXiv ID |

Filters:

```json
{"year": 2024}
{"categories": ["cs.CL", "cs.LG"]}
{"year": {"$gte": 2022, "$lte": 2024}}
{"arxiv_id": ["2401.00001", "2401.00002"]}
```

Modes:

- `bm25` (alias `sparse`) — Qdrant native sparse, hashed-unigram `log(1+TF)`.
  The name is kept for compatibility; it is not a Lucene BM25 engine.
- `dense` — BGE-M3 + Qdrant cosine.
- `hybrid` — dense + sparse in parallel, RRF `k=60`, weights 0.5/0.5.
- `rerank` — hybrid + BGE-reranker-large cross-encoder.

Response:

```json
{
  "query": "...",
  "mode": "hybrid",
  "top_k": 10,
  "elapsed_ms": 42.1,
  "hits": [
    {
      "chunk_id": "uuid",
      "score": 0.03,
      "source": "hybrid",
      "text": "passage text",
      "arxiv_id": "2401.00001",
      "heading_path": "Introduction",
      "title": "Paper title",
      "chunk_index": 3,
      "year": 2024,
      "categories": ["cs.CL"],
      "raw_score": 0.71,
      "score_norm": null,
      "tier": "fulltext"
    }
  ],
  "breakdown_ms": {"parallel_dense_sparse_ms": 31.2, "rrf_ms": 0.1, "total_ms": 42.1}
}
```

`score_norm` is `sigmoid(score)` for `source="reranked"` hits, otherwise null.

## POST `/ask`

```json
{"query": "What does the paper say about reranking?", "top_k": 10}
```

Optional fields: `recency`, `min_score`, `dedupe_by_paper`.
The response:

```json
{
  "query": "...",
  "answer": "Reranking improves ... [1]. It also ... [2, 3].",
  "context": [ /* SearchHit list */ ],
  "citations": [
    {
      "index": 1,
      "chunk_id": "uuid",
      "arxiv_id": "2401.00001",
      "quote": "first 60 chars of the supporting passage",
      "section": "Results",
      "abs_url": "https://arxiv.org/abs/2401.00001",
      "pdf_url": "https://arxiv.org/pdf/2401.00001"
    }
  ],
  "elapsed_ms": 1200.0,
  "grounding": {
    "confidence": "medium",
    "context_strength": 0.62,
    "n_cited": 2,
    "n_context": 10,
    "refusal": false,
    "warnings": []
  }
}
```

If retrieval is empty, the answer is a refusal and `context` is empty. If the
LLM call fails, the answer contains `[generation failed: ...]` and the
retrieved context is still returned.

## GET `/paper/{arxiv_id}`

```json
{
  "arxiv_id": "2401.00001",
  "title": "...",
  "authors": ["..."],
  "categories": ["cs.CL"],
  "abstract": "...",
  "n_chunks": 1,
  "chunk_ids": ["uuid"],
  "tier": "abstract"
}
```

`tier` is `abstract` only when the paper has exactly one chunk with
`heading_path="Abstract"`.

## POST `/paper/{arxiv_id}/ask`

Same request/response shape as `/ask`, but retrieval is filtered to the single
paper. Returns 404 if the paper has no chunks.

## POST `/warmup`

Loads the reranker and returns `{"reranker_loaded": true, "elapsed_ms": ...}`.
Returns 503 if the reranker is unavailable.

## Errors

| Status | When |
|---|---|
| 400 | invalid filters/recency, unknown search mode |
| 404 | paper not found / no chunks |
| 422 | request validation (empty query, out-of-range fields) |
| 429 | a heavy request is already running |
| 503 | PostgreSQL/Qdrant/reranker unavailable, or upstream HTTP error |
| 504 | upstream LLM/embedding timeout |

Heavy endpoints (`/ask`, `/paper/{id}/ask`, `/search mode=rerank`) are
serialized. `/search` and `/ask` run a storage preflight before heavy work.
