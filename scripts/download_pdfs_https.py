#!/usr/bin/env python3
"""Download arXiv PDFs for a list of IDs via plain HTTPS (no gsutil needed).

Source order per paper (auto): the public GCS bucket (fast, no rate limit),
then arxiv.org (politely, for newest papers GCS may not have yet):

  GCS:   https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf/{YYMM}/{id}v{n}.pdf
  arXiv: https://arxiv.org/pdf/{id}v{n}

Local layout: ``{pdf_dir}/{YYMM}/{id}.pdf``, where YYMM is the first four
characters of a new-style ID (2609.12345 -> 2609). Legacy IDs are skipped.

Idempotent: a file already on disk that looks like a PDF (>500 bytes, starts
with %PDF) is skipped, so re-runs only fetch the gaps.

Usage:
  python3 scripts/download_pdfs_https.py \
      --id-file  data/ids.txt \
      --pdf-dir  data/pdfs \
      --workers  6 \
      --log      data/logs/download.log \
      --failed-out data/download_failed.txt
"""
import argparse
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

GCS_BASE = "https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf"
ARXIV_BASE = "https://arxiv.org/pdf"
UA = "arxiv-search/0.1 (self-hosted corpus builder)"
MAX_VERSION = 3
LOG_EVERY = 200

_print_lock = threading.Lock()


def log(msg, log_file=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (ts, msg)
    with _print_lock:
        print(line, flush=True)
        if log_file:
            with open(log_file, "a") as f:
                f.write(line + "\n")


def force_ipv4():
    """Resolve A records only.

    Some hosts (the GPU box, 2026-09-14) get AAAA answers from DNS but have no
    working IPv6 route: urllib then burns the connect timeout on IPv6 and the
    download looks "broken" (measured there: 0.3 MB/s and 40% misses; with
    IPv4 forced: 4.9 MB/s, 48/48 OK).
    """
    original = socket.getaddrinfo

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_only


def load_id_list(path):
    ids = []
    with open(path) as f:
        for line in f:
            aid = line.strip()
            if aid:
                ids.append(aid)
    return ids


def is_valid_pdf(path):
    """Cheap PDF sniff: non-trivial size + %PDF magic."""
    try:
        if os.path.getsize(path) < 500:
            return False
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except OSError:
        return False


def fetch(url, timeout=90):
    """GET one URL. Returns bytes, or None for 404/403/410 (real miss)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code in (404, 403, 410):
            return None
        if e.code in (429, 503, 500, 502, 504):
            return "RETRY"
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return "RETRY"


def try_sources(arxiv_id, sources):
    """Return (pdf_bytes, source_label) or (None, last_reason)."""
    yymm = arxiv_id[:4]
    reason = "not-found"
    for source in sources:
        for v in range(1, MAX_VERSION + 1):
            if source == "gcs":
                url = "%s/%s/%sv%d.pdf" % (GCS_BASE, yymm, arxiv_id, v)
            else:
                url = "%s/%sv%d" % (ARXIV_BASE, arxiv_id, v)
            for attempt in range(3):
                res = fetch(url)
                if res == "RETRY":
                    time.sleep(2 + 3 * attempt)
                    reason = "retry-exhausted(%s v%d)" % (source, v)
                    continue
                if res is None:
                    reason = "404(%s v%d)" % (source, v)
                    break  # next version
                if res[:4] != b"%PDF":
                    reason = "not-pdf(%s v%d)" % (source, v)
                    break
                return res, "%s v%d" % (source, v)
            if reason.startswith("retry-exhausted"):
                break  # source-level trouble: fall through to the next source
    return None, reason


def download_one(arxiv_id, pdf_dir, sources, counters, log_file):
    yymm = arxiv_id[:4]
    out_dir = os.path.join(pdf_dir, yymm)
    out_path = os.path.join(out_dir, arxiv_id + ".pdf")

    if is_valid_pdf(out_path):
        counters["skipped"] += 1
        return None

    data, label = try_sources(arxiv_id, sources)
    if data is None:
        counters["failed"] += 1
        log("[miss] %s — %s" % (arxiv_id, label), log_file)
        return arxiv_id

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + ".part"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, out_path)  # atomic: never leave a half PDF in place

    n = counters["downloaded"] + 1
    counters["downloaded"] = n
    counters["bytes"] += len(data)
    if n % LOG_EVERY == 0:
        log("[progress] %d downloaded (%.1f MB), %d skipped, %d failed"
            % (n, counters["bytes"] / 1e6, counters["skipped"], counters["failed"]),
            log_file)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id-file", required=True)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log", default=None)
    ap.add_argument("--failed-out", default=None)
    ap.add_argument("--source", choices=["auto", "gcs", "arxiv"], default="auto",
                    help="auto = GCS first, arxiv.org fallback (default)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N IDs (debug)")
    ap.add_argument("--ipv4", action="store_true",
                    help="resolve A records only (hosts with a broken IPv6 route)")
    args = ap.parse_args()

    if args.ipv4:
        force_ipv4()

    sources = {"auto": ["gcs", "arxiv"], "gcs": ["gcs"], "arxiv": ["arxiv"]}[args.source]
    ids = load_id_list(args.id_file)
    if args.limit:
        ids = ids[:args.limit]

    log("=== download start: %d IDs, workers=%d, source=%s ==="
        % (len(ids), args.workers, args.source), args.log)
    log("pdf-dir=%s" % args.pdf_dir, args.log)

    counters = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0}
    failed = []
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_one, aid, args.pdf_dir, sources, counters, args.log)
                   for aid in ids]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                miss = fut.result()
            except Exception as e:  # keep the run alive on unexpected errors
                log("[error] %r" % (e,), args.log)
                continue
            if miss:
                failed.append(miss)
            if i % 1000 == 0:
                el = time.monotonic() - t0
                log("[progress] %d/%d processed, %.0f id/s (%.0f min elapsed)"
                    % (i, len(ids), i / max(el, 1e-9), el / 60), args.log)

    el = time.monotonic() - t0
    log("=== done in %.1f min: downloaded=%d skipped=%d failed=%d (%.1f MB) ==="
        % (el / 60, counters["downloaded"], counters["skipped"], counters["failed"],
           counters["bytes"] / 1e6), args.log)

    if args.failed_out:
        with open(args.failed_out, "w") as f:
            f.write("\n".join(failed) + ("\n" if failed else ""))
        log("failed IDs -> %s" % args.failed_out, args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
