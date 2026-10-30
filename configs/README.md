# Configs

- `example.yaml` — full-text ingest configuration: ID file, PDF root, metadata
  path, parser/chunker/embedder settings, PostgreSQL/Qdrant targets. All paths
  are explicit; there are no machine-specific defaults.
- `abstract_tier.json` — policy for the abstract-only ingest: one chunk per
  paper from `title + "\n\n" + abstract`, `heading_path="Abstract"`.

Copy/edit these files for your environment. Do not put credentials in git;
use environment variables or a local `.env` (git-ignored).
