# Scripts

Data and ingestion helpers. All scripts are stdlib-only unless noted.

Run them through the project environment (`uv run python scripts/...`), or
set `PYTHONPATH=src` if you use a plain Python interpreter; the ingest scripts
import `arxiv_mvp`.

| Script | Purpose | Needs |
|---|---|---|
| `collect_ids.py` | arXiv export API query -> ID list | stdlib, network |
| `build_corpus.py` | Kaggle snapshot -> metadata JSON + IDs | `ijson` for streaming (optional; falls back to `json.load`) |
| `fetch_metadata.py` | IDs -> metadata JSON (batches of 100) | stdlib, network |
| `download_pdfs_https.py` | IDs -> `data/pdfs/{YYMM}/{id}.pdf` | stdlib, network |
| `ingest_abstracts.py` | title+abstract -> PG + Qdrant | core Python deps, running embedder |
| `run_fulltext_ingest.py` | PDF -> Docling -> chunks -> PG + Qdrant | `[ingest]` extra, running embedder |
| `check_corpus.py` | PG/Qdrant count invariants | core Python deps |

Typical order:

```text
collect_ids.py / build_corpus.py
        -> fetch_metadata.py
        -> download_pdfs_https.py   (only for full-text)
        -> ingest_abstracts.py or run_fulltext_ingest.py
        -> check_corpus.py
```

Notes:

- `ingest_abstracts.py` requires `--apply` to write; without it, it reports.
- Both ingest paths are idempotent; re-running skips completed papers.
- The collection is created automatically with named `dense` + `sparse`
  vectors if it does not exist.
- Do not commit outputs under `data/`; it is git-ignored.
- The arXiv API is rate-limited (~1 request / 3 s). Keep the built-in sleeps.
