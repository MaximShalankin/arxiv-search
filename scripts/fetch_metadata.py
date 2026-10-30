#!/usr/bin/env python3
"""Fetch metadata for a list of arXiv IDs via API batch queries.
Batches 100 IDs per call (arXiv API limit). Stdlib-only.

Usage:
  python3 scripts/fetch_metadata.py \
      --id-file data/ids.txt \
      --output data/metadata.json
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
BATCH_SIZE = 100
SLEEP_BETWEEN = 3.0

ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def fetch_batch(ids: list[str]) -> list[dict]:
    """Fetch metadata for a batch of arXiv IDs. Returns list of dicts.

    Uses ``id_list`` (not ``search_query=id:...``): the latter does not match
    legacy IDs such as ``cs/0102003``. IDs must be version-less.
    """
    params = urllib.parse.urlencode({
        "id_list": ",".join(ids),
        "start": 0,
        "max_results": BATCH_SIZE,
    })
    url = f"{ARXIV_API}?{params}"

    req = urllib.request.Request(
        url, headers={"User-Agent": "arxiv-search/0.1 (self-hosted corpus builder)"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  FATAL batch: {e}", file=sys.stderr)
                return []

    root = ET.fromstring(data)
    results = []

    for entry in root.findall("atom:entry", ARXIV_NS):
        rec = {}

        # ID
        id_el = entry.find("atom:id", ARXIV_NS)
        if id_el is not None and id_el.text:
            raw = id_el.text.strip().replace("http://arxiv.org/abs/", "")
            if "v" in raw:
                raw = raw[:raw.rindex("v")]
            rec["id"] = raw

        # Title
        title_el = entry.find("atom:title", ARXIV_NS)
        rec["title"] = title_el.text.strip() if title_el is not None and title_el.text else ""

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", ARXIV_NS):
            name_el = author_el.find("atom:name", ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())
        rec["authors"] = ", ".join(authors)

        # Categories
        cats = []
        for cat_el in entry.findall("atom:category", ARXIV_NS):
            term = cat_el.get("term", "")
            if term:
                cats.append(term)
        rec["categories"] = " ".join(cats)

        # Abstract
        abs_el = entry.find("atom:summary", ARXIV_NS)
        rec["abstract"] = abs_el.text.strip() if abs_el is not None and abs_el.text else ""

        # Primary category
        primary_el = entry.find("arxiv:primary_category", ARXIV_NS)
        if primary_el is not None:
            rec["primary_category"] = primary_el.get("term", "")
        elif cats:
            rec["primary_category"] = cats[0]

        results.append(rec)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-file", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Load IDs
    with open(args.id_file) as f:
        all_ids = [l.strip() for l in f if l.strip()]

    print(f"Fetching metadata for {len(all_ids):,} papers in batches of {BATCH_SIZE}")
    t0 = time.monotonic()

    all_records = []
    batches = [all_ids[i:i + BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]
    total_batches = len(batches)

    for bi, batch in enumerate(batches):
        recs = fetch_batch(batch)
        all_records.extend(recs)

        if (bi + 1) % 10 == 0 or bi == total_batches - 1:
            elapsed = time.monotonic() - t0
            rate = (bi + 1) / elapsed if elapsed > 0 else 0
            eta = (total_batches - bi - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{bi+1}/{total_batches}] {len(all_records):,} records | "
                  f"{rate:.2f} batch/s | ETA: {eta:.1f}min", flush=True)

        if bi < total_batches - 1:
            time.sleep(SLEEP_BETWEEN)

    elapsed = time.monotonic() - t0

    # Write output as a JSON array (the ingest pipeline reads this format).
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_records, f)

    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Records: {len(all_records):,}")
    print(f"Written to: {args.output}")

    # Stats
    missing = len(all_ids) - len(all_records)
    if missing:
        print(f"Missing: {missing} (no metadata from API)")
    cs_count = sum(1 for r in all_records if any(c.startswith("cs.") for c in r.get("categories", "").split()))
    print(f"CS papers: {cs_count}")


if __name__ == "__main__":
    main()
