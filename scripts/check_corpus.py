#!/usr/bin/env python3
"""Check corpus invariants: PG counts vs Qdrant points.

Usage:
  python3 scripts/check_corpus.py \
      --pg-dsn postgresql://arxiv:arxiv@localhost:5432/arxiv_mvp \
      --qdrant-url http://localhost:6333 --collection arxiv_chunks

Exit code 0 when ``chunks == qdrant points`` and there are no chunks without
a paper; 1 otherwise. Prints a JSON summary.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from arxiv_mvp.store.postgres import PostgresStore
from arxiv_mvp.store.qdrant import QdrantStore


async def run(args) -> int:
    async with PostgresStore(dsn=args.pg_dsn) as pg, \
            QdrantStore(url=args.qdrant_url,
                        collection_name=args.collection) as qdrant:
        async with pg.pool.acquire() as conn:
            papers = await conn.fetchval("SELECT count(*) FROM papers")
            chunks = await conn.fetchval("SELECT count(*) FROM chunks")
            missing = await conn.fetchval(
                "SELECT count(*) FROM papers p "
                "WHERE NOT EXISTS (SELECT 1 FROM chunks c "
                "                  WHERE c.arxiv_id = p.arxiv_id)")
            progress = await conn.fetch(
                "SELECT status, count(*) AS n FROM ingestion_progress "
                "GROUP BY status ORDER BY status")
        points = (await qdrant.client.count(
            collection_name=qdrant.collection_name, exact=True)).count

    summary = {
        "papers": papers,
        "chunks": chunks,
        "qdrant_points": points,
        "papers_without_chunks": missing,
        "ingestion_progress": {r["status"]: r["n"] for r in progress},
        "ok": chunks == points and missing == 0,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pg-dsn", default=os.environ.get(
        "PG_DSN", "postgresql://arxiv:arxiv@localhost:5432/arxiv_mvp"))
    ap.add_argument("--qdrant-url", default=os.environ.get(
        "QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default=os.environ.get(
        "QDRANT_COLLECTION", "arxiv_chunks"))
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
