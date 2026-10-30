#!/usr/bin/env python3
"""Build a filtered arXiv metadata file + ID list from the Kaggle snapshot.

Input: https://www.kaggle.com/datasets/Cornell-University/arxiv
       ``arxiv-metadata-oai-snapshot.json`` (~5 GB JSON array).

Output:
  * ``--output-metadata``: JSON array with id/title/authors/categories/
    abstract/primary_category/update_date (the ingest pipeline reads this);
  * ``--output-ids``: one arXiv ID per line.

Filtering is by category prefix (default ``cs,stat``) and optional year range.
If ``--limit`` is set, a deterministic hash sample is written.

Examples:
  python3 scripts/build_corpus.py \
      --snapshot data/arxiv-metadata-oai-snapshot.json \
      --output-metadata data/metadata.json \
      --output-ids data/ids.txt \
      --categories cs,stat --from-year 2020 --limit 20000

  # JSONL input is also accepted (one record per line)
  python3 scripts/build_corpus.py --snapshot data/snapshot.jsonl ...
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def iter_records(path: Path):
    """Yield records from a JSON array (streamed with ijson if available) or JSONL."""
    with path.open("rb") as f:
        head = f.read(64).lstrip()
    if head.startswith(b"["):
        try:
            import ijson  # type: ignore
        except ImportError:
            log("ijson not installed; loading the whole JSON array into memory")
            with path.open() as f:
                for rec in json.load(f):
                    yield rec
            return
        with path.open("rb") as f:
            yield from ijson.items(f, "item")
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def record_year(rec: dict) -> int | None:
    date = str(rec.get("update_date") or rec.get("created") or "")
    if len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    aid = str(rec.get("id") or "")
    if len(aid) >= 2 and aid[:2].isdigit():
        yy = int(aid[:2])
        return yy + (1900 if yy >= 91 else 2000)
    return None


def category_match(cats: list[str], prefixes: list[str]) -> bool:
    for c in cats:
        for p in prefixes:
            if c == p or c.startswith(p + "."):
                return True
    return False


def normalize(rec: dict) -> dict | None:
    aid = str(rec.get("id") or "").strip()
    if not aid:
        return None
    cats_raw = rec.get("categories") or ""
    cats = cats_raw.split() if isinstance(cats_raw, str) else list(cats_raw)
    authors = rec.get("authors") or rec.get("authors_parsed") or []
    if isinstance(authors, list):
        authors = [a if isinstance(a, str) else " ".join(map(str, a))
                   for a in authors]
    return {
        "id": aid,
        "title": (rec.get("title") or "").strip(),
        "authors": authors,
        "categories": " ".join(cats),
        "abstract": (rec.get("abstract") or "").strip(),
        "primary_category": rec.get("primary_category") or (cats[0] if cats else ""),
        "update_date": rec.get("update_date") or "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True,
                    help="Kaggle metadata file (JSON array or JSONL)")
    ap.add_argument("--output-metadata", required=True)
    ap.add_argument("--output-ids", required=True)
    ap.add_argument("--categories", default="cs,stat",
                    help="comma-separated category prefixes")
    ap.add_argument("--from-year", type=int, default=None)
    ap.add_argument("--to-year", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="deterministic sample size (0 = all matches)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prefixes = [p.strip() for p in args.categories.split(",") if p.strip()]
    records: list[dict] = []
    scanned = 0
    for raw in iter_records(Path(args.snapshot)):
        scanned += 1
        rec = normalize(raw)
        if rec is None:
            continue
        cats = rec["categories"].split()
        if not category_match(cats, prefixes):
            continue
        year = record_year(raw)
        if args.from_year and (year is None or year < args.from_year):
            continue
        if args.to_year and (year is None or year > args.to_year):
            continue
        records.append(rec)
        if scanned % 100_000 == 0:
            log(f"scanned={scanned:,} matched={len(records):,}")

    if args.limit and len(records) > args.limit:
        records.sort(key=lambda r: hashlib.sha256(
            f"{args.seed}:{r['id']}".encode()).hexdigest())
        records = records[:args.limit]

    Path(args.output_metadata).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_ids).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_metadata, "w") as f:
        json.dump(records, f, ensure_ascii=False)
    with open(args.output_ids, "w") as f:
        f.writelines(r["id"] + "\n" for r in records)

    log(f"scanned={scanned:,} selected={len(records):,} "
        f"metadata={args.output_metadata} ids={args.output_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
