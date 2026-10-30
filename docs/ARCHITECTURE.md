# Architecture

```
                        ┌──────────────────────────────┐
    arXiv metadata ───▶ │ ingest (abstract or full-text)│
    PDFs (optional)     └──────────────┬───────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                  PostgreSQL                    Qdrant
             papers / chunks /             dense + sparse
             ingestion_progress            named vectors
                         │                           │
                         └─────────────┬─────────────┘
                                       ▼
                              FastAPI (/search, /ask)
                                       │
                         embedding + LLM endpoints
```

## Data model

### PostgreSQL

- `papers` — arXiv ID (PK), title, authors[], categories[], abstract,
  submit_date, journal_ref, pdf_path.
- `chunks` — UUID (PK), `content_hash_sha256` (unique), text, `arxiv_id` FK,
  heading_path, chunk_index, token_count, embedding_model, created_at.
- `chunk_aliases` — records when identical text appears in multiple papers
  (cross-document dedup); the chunk itself is stored once.
- `ingestion_progress` — per-paper stage/status for resume and diagnostics.

`chunks.text` is the source of truth for passage text.

### Qdrant

Collection with named vectors:

- `dense` — BGE-M3, 1024 dimensions, cosine, INT8 quantization + rescore;
- `sparse` — hashed-unigram `log(1+TF)` vector, 2^18 buckets.

Payload: `arxiv_id`, `chunk_id`, `chunk_index`, `heading_path`,
`token_count`, `content_hash_sha256`, `title`, `year`, `categories`, `tier`
(abstract-only ingest). Payload intentionally omits `text`; the API hydrates
text from PostgreSQL by `chunk_id` before rerank and before returning hits.

Point IDs are the PostgreSQL `chunk_id` UUIDs, so re-ingest is idempotent.

## Ingest paths

### Abstract-only

Input: metadata JSON + ID list. For each paper:

1. upsert `papers` row (`pdf_path=NULL`);
2. build `title + "\n\n" + abstract`;
3. embed + encode sparse;
4. insert one `chunks` row (`heading_path="Abstract"`, `chunk_index=0`);
5. upsert one Qdrant point (`tier="abstract"`);
6. mark `ingestion_progress` `abstract_indexed/done`.

No PDF, no Docling.

### Full-text

Input: metadata JSON + ID list + PDFs on disk.

1. Docling parses the PDF to markdown (pool of workers);
2. markdown is chunked (~300 tokens, hard cap 9000 chars, no overlap);
3. chunks are embedded in HTTP batches;
4. rows are written to PG (`chunks`), points to Qdrant (`dense` + `sparse`);
5. `ingestion_progress` goes to `indexed/done`.

Both paths are resumable; a crashed paper is re-processed on the next run.
Abstract ingest skips papers that already have non-Abstract chunks.

## Retrieval

```
query
  ├─ sparse: hashed-unigram -> Qdrant "sparse" top-N
  ├─ dense:  BGE-M3(query)  -> Qdrant "dense"  top-N
  ├─ RRF (k=60, weights 0.5/0.5) -> fused top-N
  ├─ drop References/bibliography/term-list boilerplate
  ├─ hydrate text from PostgreSQL
  ├─ [rerank] cross-encoder(query, candidates) -> top-k
  └─ hydrate text/title for the response
```

Search modes expose the steps separately (`bm25`, `dense`, `hybrid`, `rerank`).
Rerank is optional: without the `[rerank]` extra it returns 503; dense/hybrid
still work.

## RAG

`/ask` is stateless:

1. retrieve + rerank (same path as `/search mode=rerank`);
2. drop hits below `min_score`, optionally dedupe by paper;
3. build a prompt with numbered passages `[1] ... [N]`;
4. call the LLM (`LLM_URL`), system prompt requires inline `[N]` citations;
5. parse `[N]` markers into `citations` (passage-level: chunk, section, arXiv
   links);
6. compute a grounding heuristic from cited-passage rerank scores and
   citation count.

Empty retrieval produces a refusal, not a 500. LLM failures produce an answer
prefix `[generation failed: ...]` plus the retrieved context.

## Deliberate simplifications

- No multi-turn sessions, table injection, claim extraction, citation graph,
  promotion, SEO, or frontend. See `LIMITATIONS.md`.
- No authentication; one heavy request at a time.
- The sparse encoder is fixed; changing it invalidates the index.
- The embedding dimension is fixed at collection creation (1024 for BGE-M3).
  Changing the model requires a rebuild.

## Provenance

This code is distilled from a larger private deployment. The retrieval core,
storage schema, chunker, Docling parser, sparse encoder, RRF, reranker, and
RAG prompt are the working base; the API surface and ingest runners were
trimmed to the pieces required for a self-hosted search/RAG service.
