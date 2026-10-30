#!/usr/bin/env python3
"""Abstract-tier ingest (title + abstract, no PDF).

For every paper in --ids-file, write:
  * one ``papers`` row (pdf_path = NULL);
  * one ``chunks`` row: title + abstract, heading_path="Abstract", chunk_index=0;
  * one Qdrant point with dense + native sparse vectors, payload tier="abstract".

No PDF is downloaded and no Docling parse is involved. The operation is
idempotent: papers already fully indexed are skipped; an interrupted run is
resumed by re-upserting the same chunk hash / point id.

Safety: writes are opt-in (``--apply``). Without it the script only reports
what it would do. Full-text papers are never touched.

Example:
  PYTHONPATH=src python3 scripts/ingest_abstracts.py \
      --policy configs/abstract_tier.json \
      --metadata data/metadata.json \
      --ids-file data/ids.txt \
      --pg-dsn postgresql://arxiv:arxiv@localhost:5432/arxiv_mvp \
      --qdrant-url http://localhost:6333 \
      --collection arxiv_chunks \
      --embed-url http://localhost:8005/v1/embeddings \
      --apply
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from arxiv_mvp.ingest.chunker import count_tokens
from arxiv_mvp.ingest.embedder import BGE_M3_MODEL_ID, Embedder
from arxiv_mvp.retrieval.qdrant_native_sparse import encode_sparse
from arxiv_mvp.store.postgres import PaperRecord, PostgresStore
from arxiv_mvp.store.qdrant import QdrantStore

ALGORITHM = "ingest_abstracts"
ALGORITHM_VERSION = 1
_AUTHOR_SPLIT_RE = re.compile(r"\s*,\s*|\s+and\s+")
_WS_RE = re.compile(r"\s+")


def log(msg: str) -> None:
    print(f"[abstracts] {msg}", file=sys.stderr, flush=True)


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iter_metadata(path: Path):
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            yield from json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def parse_authors(raw) -> list[str]:
    if isinstance(raw, list):
        return [_WS_RE.sub(" ", str(a)).strip() for a in raw if str(a).strip()]
    if not raw:
        return []
    return [_WS_RE.sub(" ", p).strip()
            for p in _AUTHOR_SPLIT_RE.split(str(raw)) if p.strip()]


def year_from_arxiv_id(arxiv_id: str) -> int | None:
    try:
        if "." in arxiv_id and arxiv_id[:2].isdigit():
            return 2000 + int(arxiv_id[:2])
    except ValueError:
        pass
    return None


def build_text(policy_abstract: dict, rec: dict) -> str:
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    template = policy_abstract.get("text_template") or "{title}\n\n{abstract}"
    # Plain replace, not str.format: abstracts routinely contain LaTeX braces
    # (e.g. "{x}"), which .format() would interpret as format fields.
    return (template.replace("{title}", title)
            .replace("{abstract}", abstract).strip())


def redact_dsn(dsn: str) -> str:
    # postgresql://user:password@host:port/db -> postgresql://user:***@host:port/db
    return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", dsn)


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_model(embed_url: str) -> str:
    """Ask the embed server for its served model id (the policy's
    `BAAI/bge-m3` is documentation, not necessarily the vLLM served name)."""
    models_url = embed_url.rsplit("/embeddings", 1)[0] + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["data"][0]["id"]
    except Exception:
        return BGE_M3_MODEL_ID


async def run(args) -> int:
    policy_path = Path(args.policy)
    policy = json.loads(policy_path.read_text())
    acfg = policy.get("tiers", {}).get("abstract", {})
    if not acfg.get("enabled", True):
        raise SystemExit("policy.tiers.abstract.enabled is false")
    heading = acfg.get("heading_path", "Abstract")
    stage = acfg.get("ingestion_stage", "abstract_indexed")
    tier_payload = acfg.get("tier_payload", "abstract")
    chunk_index = int(acfg.get("chunk_index", 0))
    policy_embed_model = acfg.get("embedding_model") or BGE_M3_MODEL_ID
    embedding_model = args.embed_model or resolve_model(args.embed_url)

    ids = load_ids(Path(args.ids_file))
    if not ids:
        raise SystemExit("ids file is empty")
    if len(set(ids)) != len(ids):
        log(f"warning: ids file has {len(ids) - len(set(ids))} duplicate IDs; deduplicating")
        ids = list(dict.fromkeys(ids))

    if args.fulltext_ids_file:
        ft = set(load_ids(Path(args.fulltext_ids_file)))
        overlap = ft & set(ids)
        if overlap:
            raise SystemExit(
                f"abstract ids overlap fulltext ids ({len(overlap)}), e.g. {sorted(overlap)[:3]}")

    recs: dict[str, dict] = {}
    for r in iter_metadata(Path(args.metadata)):
        aid = str(r.get("id") or "").strip()
        if aid in ids and aid not in recs:
            recs[aid] = r
    missing = [i for i in ids if i not in recs]
    if missing:
        log(f"warning: {len(missing)} ids have no metadata, e.g. {missing[:3]}")

    todo_ids = [i for i in ids if i in recs]
    if args.limit:
        todo_ids = todo_ids[:args.limit]

    log(f"ids={len(ids)} with_metadata={len(todo_ids)} missing={len(missing)}")
    log(f"targets: pg={redact_dsn(args.pg_dsn)} qdrant={args.qdrant_url} "
        f"collection={args.collection} embed={args.embed_url} model={embedding_model}")
    if not args.apply:
        sample = todo_ids[0] if todo_ids else None
        log(f"DRY RUN: would write {len(todo_ids)} abstract chunks"
            + (f"; sample={sample} title={recs[sample].get('title','')[:80]!r}" if sample else ""))
        return 0

    t0 = time.monotonic()
    counts = dict(total=len(ids), metadata=len(todo_ids), inserted=0, skipped_done=0,
                  skipped_fulltext=0, failed=0, missing=len(missing))
    async with PostgresStore(dsn=args.pg_dsn) as pg, \
            QdrantStore(url=args.qdrant_url, collection_name=args.collection) as qdrant, \
            Embedder(endpoint=args.embed_url, model_id=embedding_model,
                     concurrency=args.embed_concurrency,
                     batch_size=args.embed_batch) as embedder:
        await pg.init_schema()
        await qdrant.init_collection(recreate=False)

        # Existing chunks + progress for the target set: skip full-text papers
        # and already-done abstract papers; resume everything else.
        rows = await pg.pool.fetch(
            "SELECT arxiv_id, heading_path FROM chunks WHERE arxiv_id = ANY($1::text[])",
            todo_ids)
        fulltext = {r["arxiv_id"] for r in rows if (r["heading_path"] or "") != heading}
        abstract_chunks = {r["arxiv_id"] for r in rows if (r["heading_path"] or "") == heading}
        prog = await pg.pool.fetch(
            "SELECT arxiv_id, stage, status FROM ingestion_progress "
            "WHERE arxiv_id = ANY($1::text[])", todo_ids)
        done = {r["arxiv_id"] for r in prog
                if r["stage"] == stage and r["status"] == "done"}
        counts["skipped_fulltext"] = len(fulltext)
        counts["skipped_done"] = len(done - fulltext)
        if fulltext:
            log(f"skipping {len(fulltext)} already-fulltext papers")
        todo = [i for i in todo_ids if i not in fulltext and i not in done]
        log(f"todo={len(todo)} (abstract chunks present: {len(abstract_chunks)})")

        write_batch = max(1, args.write_batch)
        for start in range(0, len(todo), write_batch):
            batch = todo[start:start + write_batch]
            texts = [build_text(acfg, recs[i]) for i in batch]
            vectors = await embedder.embed_many(texts)
            items: list[dict] = []
            success: list[str] = []
            for aid, text, vec in zip(batch, texts, vectors):
                if vec is None:
                    counts["failed"] += 1
                    await pg.mark_stage(aid, stage, "failed", error="embedding_failed")
                    continue
                try:
                    await pg.mark_stage(aid, stage, "in_progress")
                    rec = recs[aid]
                    await pg.upsert_paper(PaperRecord(
                        arxiv_id=aid,
                        title=(rec.get("title") or "").strip(),
                        authors=parse_authors(rec.get("authors") or rec.get("authors_parsed") or []),
                        categories=(rec.get("categories") or "").split()
                        if isinstance(rec.get("categories"), str)
                        else list(rec.get("categories") or []),
                        abstract=(rec.get("abstract") or "").strip(),
                        submit_date=None,
                        journal_ref=rec.get("journal_ref"),
                        pdf_path=None,
                    ))
                    # Include arxiv_id in the hash: abstract chunks must never
                    # alias across papers (chunks.content_hash is global UNIQUE).
                    content_hash = sha256_text(f"{aid}\n{text}")
                    await pg.insert_chunks(aid, [{
                        "content_hash_sha256": content_hash,
                        "text": text,
                        "heading_path": heading,
                        "chunk_index": chunk_index,
                        "token_count": count_tokens(text),
                        "embedding_model": embedding_model,
                    }])
                    chunk_id = await pg.pool.fetchval(
                        "SELECT chunk_id::text FROM chunks "
                        "WHERE arxiv_id = $1 AND content_hash_sha256 = $2",
                        aid, content_hash)
                    if chunk_id is None:
                        raise RuntimeError("chunk insert/select returned no chunk_id")

                    sp_idx, sp_val = encode_sparse(text)
                    vec_struct: dict = {"dense": vec}
                    if sp_idx:
                        vec_struct["sparse"] = {"indices": sp_idx, "values": sp_val}
                    items.append({
                        "point_id": chunk_id,
                        "vector": vec_struct,
                        "payload": {
                            "arxiv_id": aid,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_index,
                            "heading_path": heading,
                            "token_count": count_tokens(text),
                            "content_hash_sha256": content_hash,
                            "title": (rec.get("title") or "").strip(),
                            "year": year_from_arxiv_id(aid),
                            "categories": (rec.get("categories") or "").split()
                            if isinstance(rec.get("categories"), str)
                            else list(rec.get("categories") or []),
                            "tier": tier_payload,
                            "tier_policy": f"{policy.get('policy_id')}:v{policy.get('policy_version')}",
                        },
                    })
                    success.append(aid)
                except Exception as e:  # noqa: BLE001 - one paper must not stop the batch
                    counts["failed"] += 1
                    log(f"FAILED {aid}: {e}")
                    try:
                        await pg.mark_stage(aid, stage, "failed", error=str(e)[:300])
                    except Exception:
                        pass

            if items:
                await qdrant.upsert_chunks(items)
            for aid in success:
                await pg.mark_stage(aid, stage, "done")
            counts["inserted"] += len(success)
            log(f"progress {min(start + write_batch, len(todo))}/{len(todo)} "
                f"inserted={counts['inserted']} failed={counts['failed']}")

    manifest = {
        "algorithm": ALGORITHM,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": utcnow(),
        "run_id": args.run_id,
        "policy": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
            "id": policy.get("policy_id"),
            "version": policy.get("policy_version"),
        },
        "inputs": {
            "metadata": str(args.metadata),
            "metadata_sha256": sha256_file(Path(args.metadata)),
            "ids_file": str(args.ids_file),
            "ids_sha256": sha256_file(Path(args.ids_file)),
            "limit": args.limit,
        },
        "targets": {
            "pg": redact_dsn(args.pg_dsn),
            "qdrant": args.qdrant_url,
            "collection": args.collection,
            "embed": args.embed_url,
            "embed_model": embedding_model,
            "policy_embed_model": policy_embed_model,
        },
        "counts": counts,
        "runtime_s": round(time.monotonic() - t0, 1),
    }
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        log(f"manifest: {args.manifest}")
    log(f"done: {counts}")
    return 0 if counts["failed"] == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--fulltext-ids-file", default=None,
                    help="optional: assert the abstract set does not overlap fulltext")
    ap.add_argument("--pg-dsn", default=os.environ.get(
        "PG_DSN", "postgresql://arxiv:arxiv@localhost:5432/arxiv_mvp"))
    ap.add_argument("--qdrant-url", default=os.environ.get(
        "QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default=os.environ.get(
        "QDRANT_COLLECTION", "arxiv_chunks"))
    ap.add_argument("--embed-url", default=os.environ.get(
        "BGE_M3_ENDPOINT", "http://localhost:8005/v1/embeddings"))
    ap.add_argument("--embed-model", default=os.environ.get("BGE_M3_MODEL"))
    ap.add_argument("--embed-batch", type=int, default=int(
        os.environ.get("EMBED_BATCH_SIZE", "256")))
    ap.add_argument("--embed-concurrency", type=int, default=int(
        os.environ.get("EMBED_CONCURRENCY", "2")))
    ap.add_argument("--write-batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it the script is a dry run")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
