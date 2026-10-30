# Limitations

This is a distilled core, not the full private deployment. The omissions are
deliberate; do not assume the missing pieces are "coming later" without a code
change.

The full reference specification (including the omitted features and the
measured scale) is in [`reproduction/README.md`](reproduction/README.md). It
describes the complete system, not this repository's API surface.

## Functional scope

- **Search and RAG only.** No SEO/impact analysis, claim matrix, conflicting
  evidence, table extraction/table-read, citation graph, export (BibTeX/RIS),
  or multi-turn sessions.
- **No frontend.** The service exposes JSON + FastAPI's `/docs`; there is no
  web UI.
- **No promotion queue.** Abstract-only papers stay abstract-only; there is no
  API to upgrade them to full-text. Re-run the full-text ingest for a chosen ID
  set instead.
- **No authentication or multi-tenancy.** It is a single-user/local service.
- **No bundled data or models.** You must obtain metadata/PDFs and run the
  embedding and LLM endpoints yourself.

## Operational limits

- `/ask`, `/paper/{id}/ask`, and `/search` with `mode=rerank` are serialized
  (one heavy request at a time). A concurrent heavy request gets HTTP 429.
- Heavy endpoints run a storage preflight and return HTTP 503 if PostgreSQL or
  Qdrant is unavailable. Search modes that need Qdrant fail the same way —
  including `bm25`, because native sparse vectors live in Qdrant.
- Startup logs and continues if the reranker cannot load; rerank/`/ask`
  requests then return HTTP 503 with an install/config hint. Dense/hybrid
  search without rerank still works if the embedder is up.
- `/ask` catches LLM generation failures and returns HTTP 200 with an error
  text in `answer` and the retrieved `context`. Check `grounding` and the
  answer prefix if the LLM endpoint is flaky.
- The API has no rate limiting beyond the heavy lane. Do not expose it to the
  public internet; bind to `127.0.0.1` or a private interface.

## Tiers and data quality

- Full-text quality depends on Docling and the PDF. Some papers fail parsing;
  they remain `failed` in `ingestion_progress` and can be retried.
- Abstract-only chunks are one per paper; `heading_path` is `Abstract`.
- The sparse encoder is hashed-unigram with `log(1+TF)`, not Lucene BM25. The
  `bm25` mode name is kept for API compatibility; its ranking differs from a
  BM25 engine.
- Retrieval/answer quality depends on the embedding model and the LLM. The
  reference choices (BGE-M3, BGE-reranker-large, Qwen3.5-4B) are documented in
  `DEPENDENCIES.md`; replacing them requires rebuilding the index and
  recalibrating `min_score`.

## Tested environment

- PostgreSQL 16, Qdrant 1.12.4, Python 3.11–3.13.
- Other versions may work but are not part of the acceptance path.
- The code is not a cross-platform installer; Windows is untested (WSL is the
  reasonable path).
