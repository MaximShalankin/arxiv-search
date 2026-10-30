# Operations

## Start order

1. Storage: `docker compose up -d` (PostgreSQL + Qdrant).
2. Model endpoints: embedding server (`BGE_M3_ENDPOINT`) and, for `/ask`, the
   LLM server (`LLM_URL`).
3. Ingest: abstract-only or full-text (see `DATA.md`).
4. API: `uv run uvicorn arxiv_mvp.api.app:app --host 127.0.0.1 --port 8080`.

The API does not start the model servers. It only opens HTTP clients.

## Configuration

Copy `.env.example` to `.env` and edit it, or export the variables in your
shell. The app loads `.env` at import time without overriding existing
variables. Every resolved endpoint is printed in the `startup_config` log line;
the DSN password is redacted.

For shell commands that need the same variables (for example `pg_dump`), load
the file first:

```bash
set -a; . ./.env; set +a
```

Compose reads `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and
`POSTGRES_PORT`; keep them consistent with `PG_DSN`.

Key variables:

| Variable | Default | Meaning |
|---|---|---|
| `PG_DSN` | `postgresql://arxiv:arxiv@localhost:5432/arxiv_mvp` | PostgreSQL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP |
| `QDRANT_COLLECTION` | `arxiv_chunks` | collection name |
| `BGE_M3_ENDPOINT` | `http://localhost:8005/v1/embeddings` | embeddings URL |
| `BGE_M3_MODEL` | `BAAI/bge-m3` | served model id |
| `LLM_URL` | `http://localhost:8000/v1/chat/completions` | chat completions |
| `LLM_MODEL` | `Qwen/Qwen3.5-4B` | model id |
| `LLM_API_KEY` | empty | optional bearer token |
| `RERANKER_HTTP_URL` | empty | HTTP reranker instead of in-process |
| `BGE_RERANKER_MODEL` | `BAAI/bge-reranker-large` | in-process model |
| `BGE_RERANKER_DTYPE` | `fp32` | `fp32`/`bf16`/`fp16` |

The ingest scripts take their paths/endpoints from CLI flags or the YAML
config; `.env` is not a substitute for `configs/example.yaml`.

## Health and smoke

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/stats
curl -s -X POST http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"retrieval augmented generation","mode":"hybrid","top_k":5}'
```

`/health` reports storage connectivity and whether the reranker is loaded. A
storage outage yields `error: <ExceptionName>` in the field; heavy endpoints
return 503 with a reason.

## Logs

The service writes one JSON object per line to stderr. Ingest runners do the
same via `arxiv_mvp.utils.logging`. Redirect to a file or a log collector; do
not commit logs.

## Backup and restore

### PostgreSQL

```bash
mkdir -p data/dumps
pg_dump "$PG_DSN" -Fc -f data/dumps/arxiv.dump
# restore into an empty database:
pg_restore --clean --if-exists -d "$PG_DSN" data/dumps/arxiv.dump
```

### Qdrant

```bash
# create a snapshot
curl -s -X POST "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots"
# list and download it
curl -s "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots"
curl -s -o data/dumps/qdrant.snapshot \
  "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots/<snapshot-name>"
```

Restore by uploading the snapshot to the same collection name on a fresh
Qdrant (see the Qdrant snapshot API) and re-running `check_corpus.py`.

Backups belong on durable storage outside the repository working tree. The
repository intentionally does not include a backup scheduler.

## Safety

- **No authentication.** Bind the API to `127.0.0.1` or a private interface.
- Do not commit `.env`, DSNs with real passwords, API keys, dumps, or data.
- One heavy request at a time is enforced with HTTP 429. A second concurrent
  `/ask`, `/paper/{id}/ask`, or `/search mode=rerank` is rejected, not queued.
- When Qdrant is down, all search modes fail (including `bm25`); this is
  expected because native sparse vectors live in Qdrant.

## Rebuild

A deliberate collection rebuild is destructive:

```bash
# wipe and recreate the Qdrant collection, then re-ingest
# set recreate_qdrant: true in configs/example.yaml (full-text)
# or pass a fresh --collection name to ingest_abstracts.py
```

For a model change (embedding dimension or normalization), always rebuild the
collection; old and new vectors must not mix. The collection dimension is set
at creation (`QdrantStore(vector_size=...)`, default 1024 for BGE-M3).
