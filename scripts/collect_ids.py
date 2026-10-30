#!/usr/bin/env python3
"""Collect arXiv IDs for a query via the public export API.

Example:
  python3 scripts/collect_ids.py \
      --query 'cat:cs.CL AND submittedDate:[202401010000 TO 202412312359]' \
      --max 5000 --output data/ids.txt

The API returns Atom XML and caps each response at 2000 records; this script
paginates with ``start`` and sleeps between requests. It is deliberately
conservative: ~1 request / 3 s, with retries on transient errors.
"""
import argparse
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
PAGE_CAP = 2000


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fetch_page(query: str, start: int, page_size: int,
               sort_by: str, sort_order: str, user_agent: str) -> tuple[list[str], int]:
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": start,
        "max_results": page_size,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    })
    url = f"{ARXIV_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            break
        except Exception as e:  # noqa: BLE001 - network retries
            last_err = e
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"batch failed: {last_err}")

    root = ET.fromstring(data)
    total_el = root.find("opensearch:totalResults", NS)
    total = int(total_el.text) if total_el is not None and total_el.text else 0
    ids = []
    for entry in root.findall("atom:entry", NS):
        id_el = entry.find("atom:id", NS)
        if id_el is None or not id_el.text:
            continue
        raw = id_el.text.strip()
        raw = raw.replace("https://arxiv.org/abs/", "").replace(
            "http://arxiv.org/abs/", "")
        if "v" in raw:
            raw = raw[:raw.rindex("v")]
        ids.append(raw)
    return ids, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", required=True,
                    help="arXiv search_query, e.g. \"cat:cs.CL\"")
    ap.add_argument("--output", required=True, help="output ID file")
    ap.add_argument("--max", type=int, default=5000,
                    help="maximum IDs to collect (0 = no limit)")
    ap.add_argument("--page-size", type=int, default=PAGE_CAP,
                    help=f"records per request (max {PAGE_CAP})")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="seconds between requests")
    ap.add_argument("--sort-by", default="submittedDate",
                    choices=["relevance", "lastUpdatedDate", "submittedDate"])
    ap.add_argument("--sort-order", default="descending",
                    choices=["ascending", "descending"])
    ap.add_argument("--include-legacy", action="store_true",
                    help="keep pre-2007 IDs like cs/0102003 (no PDF bucket); "
                         "they are dropped by default")
    ap.add_argument("--user-agent", default="arxiv-search/0.1 (corpus builder)")
    args = ap.parse_args()

    page_size = min(max(1, args.page_size), PAGE_CAP)
    all_ids: list[str] = []
    seen: set[str] = set()
    start = 0
    total = None
    t0 = time.monotonic()

    while total is None or start < total:
        wanted = page_size
        if args.max:
            wanted = min(wanted, args.max - len(all_ids))
            if wanted <= 0:
                break
        ids, total = fetch_page(
            args.query, start, wanted, args.sort_by, args.sort_order,
            args.user_agent)
        new = 0
        for aid in ids:
            if not args.include_legacy and "." not in aid:
                continue
            if aid in seen:
                continue
            seen.add(aid)
            all_ids.append(aid)
            new += 1
        log(f"start={start} fetched={len(ids)} new={new} "
            f"total={len(all_ids)}/{total} "
            f"elapsed={time.monotonic() - t0:.0f}s")
        if not ids:
            break
        start += len(ids)
        if total is None or start < total:
            time.sleep(args.sleep)

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for aid in all_ids:
            f.write(aid + "\n")
    log(f"wrote {len(all_ids)} IDs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
