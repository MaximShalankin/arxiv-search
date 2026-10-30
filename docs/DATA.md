# Data

The repository ships **no corpus**. `data/` is git-ignored; only scripts and
ID files belong in git. Build the data locally with one of the two paths below.

## Path A — small slice via the arXiv export API (recommended first)

No Kaggle account, no 5 GB download. The export API is rate-limited; the
scripts sleep between requests.

```bash
# 2000 cs.CL papers submitted in 2024
uv run python scripts/collect_ids.py \
  --query 'cat:cs.CL AND submittedDate:[202401010000 TO 202412312359]' \
  --max 2000 --output data/ids.txt

# metadata for those IDs (batches of 100, ~3s pause)
uv run python scripts/fetch_metadata.py \
  --id-file data/ids.txt --output data/metadata.json
```

`collect_ids.py` options:

- `--query` — any arXiv `search_query` (`cat:cs.LG`, `all:"retrieval augmented"`,
  boolean `AND`/`OR`/`ANDNOT`, date ranges);
- `--max` — stop after N IDs (0 = all);
- legacy IDs like `cs/0102003` are dropped by default (the PDF downloader
  cannot bucket them); pass `--include-legacy` to keep them;
- `--sort-by submittedDate --sort-order descending` is the default (newest
  first); use `ascending` for a historical sweep. Keep a stable sort if you
  paginate a large query.

API limits and etiquette:

- ~1 request / 3 seconds; `fetch_metadata.py` and `collect_ids.py` already
  sleep.
- The export API returns at most 2000 records per request; `collect_ids.py`
  paginates with `start`.
- Send a descriptive `User-Agent`; both scripts do.
- On 429/503 the scripts retry with backoff.

## Path B — full Kaggle snapshot (large corpus)

Dataset: <https://www.kaggle.com/datasets/Cornell-University/arxiv>
File: `arxiv-metadata-oai-snapshot.json` (~5 GB JSON array).

```bash
# after downloading and extracting the snapshot
uv run python scripts/build_corpus.py \
  --snapshot data/arxiv-metadata-oai-snapshot.json \
  --output-metadata data/metadata.json \
  --output-ids data/ids.txt \
  --categories cs,stat \
  --from-year 2007 \
  --limit 20000          # 0 = all matches; deterministic hash sample otherwise
```

`build_corpus.py` streams the JSON array with `ijson` when available (the
`[ingest]` extra), otherwise falls back to loading the whole file. It writes
the metadata array expected by the ingest scripts and a plain ID list.

## PDFs (full-text path)

```bash
uv run python scripts/download_pdfs_https.py \
  --id-file data/ids.txt \
  --pdf-dir data/pdfs \
  --workers 6 \
  --failed-out data/download_failed.txt
```

Sources per paper, in order:

1. public GCS mirror:
   `https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/{YYMM}/{id}v{n}.pdf`
2. `https://arxiv.org/pdf/{id}v{n}` (polite fallback for the newest papers).

Layout: `data/pdfs/{YYMM}/{id}.pdf`. The downloader is idempotent: existing
valid PDFs are skipped, so re-runs fetch only gaps. It skips legacy IDs
(`cs/0102003`) because they have no `{YYMM}` bucket.

## Ingest

### Abstract-only (fast, no PDFs)

```bash
uv run python scripts/ingest_abstracts.py \
  --policy configs/abstract_tier.json \
  --metadata data/metadata.json \
  --ids-file data/ids.txt \
  --apply
```

Writes one `papers` row, one `chunks` row (`heading_path="Abstract"`), and one
Qdrant point per paper (dense + sparse). It is resumable and skips papers that
already have full-text chunks. Without `--apply` it only reports.

### Full-text (PDF -> Docling -> chunks)

```bash
uv sync --extra ingest
uv run python scripts/run_fulltext_ingest.py --config configs/example.yaml
```

Edit `configs/example.yaml` first: `subsample_path`, `pdf_root`,
`metadata_path`, storage DSN, Qdrant URL, and the embedding endpoint. The
runner is resumable (`ingestion_progress`); `--retry-failed` and
`--reset-in-progress` are available.

Full-text and abstract-only share the same collection. Do not index the same
ID twice; the abstract ingest skips papers that already have non-Abstract
chunks.

## Gates

After ingest:

```bash
uv run python scripts/check_corpus.py
```

The gate is:

- `chunks == qdrant_points` (exact count);
- `papers_without_chunks == 0`;
- a second ingest run adds no duplicates;
- `ingestion_progress` has no surprising `failed` rows for the target batch.

A fresh corpus may differ from a long-lived one: check the **delta** of the
batch, not only the global counts, if you add to an existing index.

## What must not be committed

- PDFs, metadata JSON/JSONL, Qdrant storage, PG dumps.
- Model weights or caches.
- API keys, DSNs with real passwords.

Only ID lists and scripts belong in git. `data/` is ignored, with a
`.gitkeep` placeholder.
