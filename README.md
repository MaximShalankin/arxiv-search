# arxiv-search

Self-hosted research assistant over an arXiv corpus: hybrid retrieval
(BGE-M3 dense + native sparse, RRF), optional cross-encoder rerank, and RAG
answers with inline `[N]` citations.

This repository is a **distilled core** of a larger private deployment. It is
not a universal product and not a hosted service. The code is the working
base: PostgreSQL + Qdrant + FastAPI + external embedding/LLM endpoints.
See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before deploying anything.

## Included

- FastAPI service: `/search`, `/ask`, `/paper/{id}`, `/paper/{id}/ask`,
  `/stats`, `/health`, `/warmup`.
- PostgreSQL schema/store: papers, chunks, chunk aliases, ingestion progress.
- Qdrant hybrid collection: named vectors `dense` (BGE-M3, 1024-d) and
  `sparse` (hashed-unigram).
- Retrieval modes: `bm25` (native sparse), `dense`, `hybrid` (RRF),
  `rerank` (cross-encoder).
- RAG with passage-level citations and a grounding heuristic.
- Ingest paths:
  - abstract-only: title + abstract, one chunk per paper, no PDF;
  - full-text: PDF -> Docling -> chunks -> embeddings.
- Data helpers: arXiv ID collection, metadata fetch, Kaggle snapshot filter,
  PDF download, corpus invariant check.
- Tests for the core invariants.

## Not included

- SEO/impact analytics, claim matrix, conflicting evidence, table extraction,
  citation graph, promotion queue, multi-turn sessions, export endpoints.
- Any frontend or landing page.
- Vault/inbox/tickets/history of the private deployment.
- Bundled datasets, PDFs, indexes, dumps, or model weights.
- Authentication, multi-tenancy, rate limiting beyond one-heavy-request-at-a-time.

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.11–3.13 |
| PostgreSQL | 16 (compose file provided) |
| Qdrant | 1.12.x (compose file provided) |
| Embeddings | OpenAI-compatible `/v1/embeddings`, BGE-M3, 1024-d |
| LLM | OpenAI-compatible `/v1/chat/completions` for `/ask` |
| Reranker | in-process `[rerank]` extra or HTTP `RERANKER_HTTP_URL` |
| PDF ingest | `[ingest]` extra (Docling) |

Exact Python packages and model versions: [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## Quickstart (abstract-only, no PDFs)

```bash
cp .env.example .env
# edit .env if your storage/embedding/LLM endpoints differ

docker compose up -d                  # PostgreSQL + Qdrant
uv sync --extra rerank                # API + in-process reranker
# uv sync                            # or: no reranker; use RERANKER_HTTP_URL

# 1) IDs: a small cs.CL 2024 slice (arXiv export API, ~3s between requests)
uv run python scripts/collect_ids.py \
  --query 'cat:cs.CL AND submittedDate:[202401010000 TO 202412312359]' \
  --max 2000 --output data/ids.txt

# 2) metadata for those IDs
uv run python scripts/fetch_metadata.py \
  --id-file data/ids.txt --output data/metadata.json

# 3) abstract-only index (writes to PG + Qdrant; --apply is required)
uv run python scripts/ingest_abstracts.py \
  --policy configs/abstract_tier.json \
  --metadata data/metadata.json --ids-file data/ids.txt --apply

# 4) serve
uv run uvicorn arxiv_mvp.api.app:app --host 127.0.0.1 --port 8080
```

Check the corpus and query it:

```bash
uv run python scripts/check_corpus.py
curl -s -X POST http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"retrieval augmented generation","mode":"hybrid","top_k":5}'
```

Full-text (PDF -> Docling) is a second, heavier path:

```bash
uv sync --extra ingest
uv run python scripts/download_pdfs_https.py --id-file data/ids.txt --pdf-dir data/pdfs
uv run python scripts/run_fulltext_ingest.py --config configs/example.yaml
```

Details, rate limits, and gates: [docs/DATA.md](docs/DATA.md).

## Data is not in git

`data/` is ignored by git. The repository contains only IDs, scripts, and
instructions. Download arXiv metadata/PDFs yourself; do not commit them.
See [docs/DATA.md](docs/DATA.md).

## API

See [docs/API.md](docs/API.md) and the generated OpenAPI docs at `/docs`.

## Operations

See [docs/OPERATIONS.md](docs/OPERATIONS.md). The API has **no
authentication**; the compose ports bind to `127.0.0.1` and the API should
too. Do not expose it to the public internet.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data flow and storage model.
- [docs/DATA.md](docs/DATA.md) — how to obtain the corpus.
- [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md) — Python/services/models.
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — what this build does not do.
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — run, check, back up.
- [docs/API.md](docs/API.md) — endpoint contract.
- [docs/reproduction/README.md](docs/reproduction/README.md) — reference
  specification of the complete deployment (includes features not implemented
  in this distilled repository).

## License

MIT. See [LICENSE](LICENSE). arXiv metadata and PDFs are subject to arXiv's
terms; model weights are subject to their own licenses (BGE-M3 and
BGE-reranker-large are MIT).
