"""
Download real PDFs from arXiv and upload to S3 to seed the demo.

We use the arXiv export API (https://info.arxiv.org/help/api/index.html), which
is free and explicitly intended for programmatic access. Be a good citizen:
default concurrency is low (4) and we sleep briefly between downloads.

Usage:
    python -m demo.seed_documents --bucket my-input --count 100

Defaults pull recent papers from cs.AI. Override with --query.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import requests

from agent.config import AWS_REGION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("demo.seed")

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "self-healing-agent-demo/0.1 (https://github.com/DHILIP-S-E/self-healing-agent)"


# =============================================================================
# arXiv discovery
# =============================================================================

def fetch_arxiv_pdf_urls(query: str, count: int) -> list[tuple[str, str]]:
    """Return [(arxiv_id, pdf_url)] for the top `count` results matching `query`.

    Source: https://info.arxiv.org/help/api/user-manual.html
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": count,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    log.info("Querying arXiv: %s (count=%d)", query, count)
    resp = requests.get(ARXIV_API, params=params, timeout=90, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    results: list[tuple[str, str]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        id_elem = entry.find("atom:id", ATOM_NS)
        if id_elem is None or not id_elem.text:
            continue
        # id like http://arxiv.org/abs/2401.12345v2 — keep version for correctness.
        full_id = id_elem.text.rsplit("/", 1)[-1]
        canonical_id = full_id.split("v")[0] if "v" in full_id else full_id
        pdf_url = f"https://arxiv.org/pdf/{full_id}"
        results.append((canonical_id, pdf_url))
    log.info("Got %d paper URLs from arXiv", len(results))
    return results


# =============================================================================
# Download
# =============================================================================

def download_one(arxiv_id: str, url: str, out_dir: Path) -> Path | None:
    """Download one PDF. Skip if already cached. Return local path or None."""
    out_path = out_dir / f"{arxiv_id}.pdf"
    if out_path.exists() and out_path.stat().st_size > 0:
        log.debug("cached: %s", arxiv_id)
        return out_path

    try:
        r = requests.get(
            url, timeout=60, allow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
    except Exception as e:
        log.warning("download failed for %s: %s", arxiv_id, e)
        return None

    if not r.content.startswith(b"%PDF"):
        log.warning("%s did not return a PDF (got %r)", url, r.content[:8])
        return None

    out_path.write_bytes(r.content)
    log.info("downloaded %s (%d bytes)", arxiv_id, len(r.content))
    return out_path


# =============================================================================
# Upload
# =============================================================================

def upload_pdfs_to_s3(s3_client, bucket: str, prefix: str, paths: list[Path]) -> int:
    """Upload local PDFs to s3://bucket/prefix/<filename>. Returns upload count."""
    prefix = prefix.strip("/")
    uploaded = 0
    for p in paths:
        if p is None or not p.exists():
            continue
        key = f"{prefix}/{p.name}" if prefix else p.name
        s3_client.upload_file(
            str(p), bucket, key,
            ExtraArgs={"ContentType": "application/pdf", "ServerSideEncryption": "AES256"},
        )
        uploaded += 1
    return uploaded


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Seed demo PDFs from arXiv to S3.")
    parser.add_argument("--bucket", required=True, help="Destination S3 bucket")
    parser.add_argument("--prefix", default="papers", help="S3 key prefix (default: papers)")
    parser.add_argument("--query", default="cat:cs.AI", help="arXiv search query (default: cat:cs.AI)")
    parser.add_argument("--count", type=int, default=100, help="Number of PDFs to seed (default: 100)")
    parser.add_argument("--workdir", default="./demo/sample_pdfs", help="Local download dir")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--region", default=AWS_REGION, help="AWS region for the S3 client")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=args.region)

    # 1. Discover URLs
    pairs = fetch_arxiv_pdf_urls(args.query, args.count)
    if not pairs:
        log.error("No PDF URLs returned. Check your query.")
        return 1

    # 2. Download in parallel (gentle to arXiv: low concurrency + small sleep)
    log.info("Downloading %d PDFs to %s with concurrency=%d ...", len(pairs), workdir, args.concurrency)
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(download_one, aid, url, workdir): aid for aid, url in pairs}
        for fut in as_completed(futures):
            p = fut.result()
            if p is not None:
                paths.append(p)
            time.sleep(0.1)

    if not paths:
        log.error("No PDFs downloaded successfully.")
        return 1

    # 3. Upload to S3
    log.info("Uploading %d PDFs to s3://%s/%s ...", len(paths), args.bucket, args.prefix)
    count = upload_pdfs_to_s3(s3, args.bucket, args.prefix, paths)
    log.info("Done. Uploaded %d PDFs to s3://%s/%s/", count, args.bucket, args.prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
