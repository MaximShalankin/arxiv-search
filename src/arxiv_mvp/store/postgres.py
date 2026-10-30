"""PostgreSQL async store: papers + chunks + ingestion_progress + chunk_aliases.

Schema:
  - papers(arxiv_id PK, title, authors[], categories[], abstract, ...)
  - chunks(chunk_id UUID PK, content_hash_sha256 UNIQUE, text, arxiv_id FK,
           heading_path, chunk_index, token_count, embedding_model, created_at)
  - chunk_aliases(chunk_id FK, arxiv_id, chunk_index) — references for
                 cross-document dedup: same text appears in multiple papers.
  - ingestion_progress(arxiv_id PK, stage, status, updated_at, error)

Stage values: 'queued' | 'parsed' | 'chunked' | 'embedded' | 'indexed'
Status values: 'in_progress' | 'done' | 'failed'

Usage:
    async with PostgresStore(dsn=...) as pg:
        await pg.init_schema()
        await pg.upsert_paper(...)
        await pg.mark_stage(arxiv_id, 'parsed', 'done')
"""
from collections.abc import Iterable
from dataclasses import dataclass

import asyncpg

from ..utils.logging import get_logger

log = get_logger("postgres")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,
    title             TEXT,
    authors           TEXT[],
    categories        TEXT[],
    abstract          TEXT,
    submit_date       DATE,
    journal_ref       TEXT,
    pdf_path          TEXT,
    parsed_at         TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash_sha256 TEXT NOT NULL,
    text               TEXT NOT NULL,
    arxiv_id           TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    heading_path       TEXT,
    chunk_index        INTEGER NOT NULL,
    token_count        INTEGER NOT NULL,
    embedding_model    TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (content_hash_sha256)
);

CREATE INDEX IF NOT EXISTS chunks_arxiv_id_idx ON chunks(arxiv_id);
CREATE INDEX IF NOT EXISTS chunks_arxiv_id_chunk_idx ON chunks(arxiv_id, chunk_index);

CREATE TABLE IF NOT EXISTS chunk_aliases (
    chunk_id    UUID NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    arxiv_id    TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, arxiv_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS ingestion_progress (
    arxiv_id   TEXT PRIMARY KEY,
    stage      TEXT NOT NULL,           -- 'queued'|'parsed'|'chunked'|'embedded'|'indexed'
    status     TEXT NOT NULL,           -- 'in_progress'|'done'|'failed'
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error      TEXT
);

CREATE INDEX IF NOT EXISTS ingestion_progress_status_idx
    ON ingestion_progress(status, stage);
"""


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    abstract: str
    submit_date: object  # date or None
    journal_ref: str | None
    pdf_path: str


class PostgresStore:
    def __init__(self, dsn: str = "postgresql://arxiv:arxiv@localhost:5432/arxiv_mvp"):
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def __aenter__(self) -> "PostgresStore":
        self._pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        return self

    async def __aexit__(self, *a):
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresStore not entered (use 'async with')")
        return self._pool

    async def init_schema(self):
        """Create tables and indexes if not exist. Idempotent."""
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        log.info("schema_initialized")

    async def upsert_paper(self, p: PaperRecord):
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO papers (arxiv_id, title, authors, categories, abstract,
                                    submit_date, journal_ref, pdf_path, parsed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    categories = EXCLUDED.categories,
                    abstract = EXCLUDED.abstract,
                    submit_date = EXCLUDED.submit_date,
                    journal_ref = EXCLUDED.journal_ref,
                    pdf_path = EXCLUDED.pdf_path,
                    parsed_at = NOW()
                """,
                p.arxiv_id, p.title, p.authors, p.categories, p.abstract,
                p.submit_date, p.journal_ref, p.pdf_path,
            )

    async def insert_chunks(self, arxiv_id: str, chunks: Iterable[dict]):
        """Insert chunks for a paper. Each chunk dict must have:
        content_hash_sha256, text, heading_path, chunk_index, token_count,
        embedding_model.

        On UNIQUE conflict (cross-document dedup), record alias to existing chunk.
        Returns dict {inserted: int, aliased: int}.
        """
        inserted = 0
        aliased = 0
        async with self.pool.acquire() as conn, conn.transaction():
            for c in chunks:
                row = await conn.fetchrow(
                    """
                        INSERT INTO chunks (content_hash_sha256, text, arxiv_id,
                                            heading_path, chunk_index, token_count,
                                            embedding_model)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (content_hash_sha256) DO NOTHING
                        RETURNING chunk_id
                        """,
                    c["content_hash_sha256"], c["text"], arxiv_id,
                    c.get("heading_path"), c["chunk_index"],
                    c["token_count"], c.get("embedding_model"),
                )
                if row is not None:
                    inserted += 1
                else:
                    # Existing chunk by hash → record alias
                    existing = await conn.fetchval(
                        "SELECT chunk_id FROM chunks WHERE content_hash_sha256 = $1",
                        c["content_hash_sha256"],
                    )
                    if existing is not None:
                        await conn.execute(
                            """
                                INSERT INTO chunk_aliases (chunk_id, arxiv_id, chunk_index)
                                VALUES ($1, $2, $3)
                                ON CONFLICT DO NOTHING
                                """,
                            existing, arxiv_id, c["chunk_index"],
                        )
                        aliased += 1
        log.info("chunks_inserted", arxiv_id=arxiv_id,
                 inserted=inserted, aliased=aliased)
        return {"inserted": inserted, "aliased": aliased}

    async def mark_stage(self, arxiv_id: str, stage: str, status: str,
                         error: str | None = None):
        """Per-document commit: record stage transition."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ingestion_progress (arxiv_id, stage, status, updated_at, error)
                VALUES ($1, $2, $3, NOW(), $4)
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    stage = EXCLUDED.stage,
                    status = EXCLUDED.status,
                    updated_at = NOW(),
                    error = EXCLUDED.error
                """,
                arxiv_id, stage, status, error,
            )

    async def get_progress(self, arxiv_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT stage, status, updated_at, error FROM ingestion_progress WHERE arxiv_id = $1",
                arxiv_id,
            )
            return dict(row) if row else None

    async def get_done_arxiv_ids(self) -> set[str]:
        """Return arxiv_ids that are fully indexed (for resume)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT arxiv_id FROM ingestion_progress WHERE stage='indexed' AND status='done'"
            )
            return {r["arxiv_id"] for r in rows}

    async def get_failed_arxiv_ids(self) -> list[str]:
        """For --retry-failed."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT arxiv_id FROM ingestion_progress WHERE status='failed'"
            )
            return [r["arxiv_id"] for r in rows]

    async def get_in_progress_arxiv_ids(self) -> list[str]:
        """Return arxiv_ids with status='in_progress' (crashed mid-stage).

        Used for checkpoint diagnostics + reset_in_progress() preview.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT arxiv_id FROM ingestion_progress WHERE status='in_progress'"
            )
            return [r["arxiv_id"] for r in rows]

    async def reset_in_progress(self) -> int:
        """Delete ingestion_progress rows with status='in_progress'.

        These represent papers that crashed mid-stage (SIGINT, OOM, etc).
        On next pipeline run, they will be re-processed from scratch —
        process_paper() is idempotent (paper upsert, chunk ON CONFLICT DO
        NOTHING, Qdrant upsert by point_id, OS bulk_index by doc_id), so
        partial work from the crashed run is harmless.

        Returns count of reset rows. Safe to call before run_pipeline()
        or as a standalone cleanup tool.

        ticket-002.3 Phase B checkpoint (2026-06-18).
        """
        async with self.pool.acquire() as conn:
            n = await conn.execute(
                "DELETE FROM ingestion_progress WHERE status='in_progress'"
            )
            # asyncpg returns 'DELETE N' — parse N
            count = int(n.split()[-1]) if n else 0
            if count:
                log.info("in_progress_reset", count=count)
            return count
