#!/usr/bin/env python3
"""Run ONE OCR batch on Unilever PDF, save to cache, show results.

Usage:
    python scripts/run_ocr_batch.py          # processes next batch from cache
    python scripts/run_ocr_batch.py --reset   # clear cache and start fresh

Each run processes 50 pages then exits. Re-run to continue.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ocr_batch")

# Handle --reset flag
if "--reset" in sys.argv:
    import shutil
    cache_dir = Path("cache/ocr_pages")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("Cleared OCR cache")

# Load PDF
pdf_path = "cache/unilever_test.pdf"
try:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
except FileNotFoundError:
    logger.error("PDF not found at %s -- download it first", pdf_path)
    sys.exit(1)

from operator1.clients.fuzzy_pdf_parser import (
    _ocr_extract_segments,
    _load_ocr_cache,
    _ocr_cache_path,
)

# Check existing cache
cached = _load_ocr_cache(pdf_bytes)
n_cached = len(cached.get("page_texts", {}))
last_page = cached.get("last_page_processed", -1)
total_pages = cached.get("n_pages", 0)
if n_cached:
    logger.info("Resume: %d pages cached, last=%d, total=%d", n_cached, last_page, total_pages)

# UK/IFRS + Unilever segment keywords
keywords = [
    "segment", "revenue", "operating profit", "turnover",
    "beauty", "personal care", "home care", "nutrition",
    "ice cream", "foods", "refreshment",
    "beauty & wellbeing",
]

# Run ONE batch (50 pages) then return
segments = _ocr_extract_segments(
    pdf_bytes,
    keywords,
    min_page_score=2,
    max_pages=10,
    market_id="uk_companies_house",
    batch_size=25,
    max_batches_per_run=1,  # ONE batch of 25 pages per invocation
)

# Report cache state
cached = _load_ocr_cache(pdf_bytes)
n_cached = len(cached.get("page_texts", {}))
last_page = cached.get("last_page_processed", -1)
total_pages = cached.get("n_pages", 0)
remaining = total_pages - last_page - 1 if total_pages > 0 else 0

logger.info("=" * 60)
logger.info("BATCH COMPLETE")
logger.info("  Pages OCR'd: %d / %d", n_cached, total_pages)
logger.info("  Last page:   %d", last_page)
logger.info("  Remaining:   %d pages (%d more runs needed)", remaining, (remaining + 24) // 25)

# Show scored pages from cache
if cached.get("page_texts"):
    scored = []
    for pg_str, text in cached["page_texts"].items():
        lower = text.lower()
        score = sum(1 for kw in keywords if kw in lower)
        if score >= 2:
            scored.append((int(pg_str), score, len(text)))
    scored.sort(key=lambda x: -x[1])
    if scored:
        logger.info("  Top scored pages (score >= 2):")
        for pg, sc, tl in scored[:5]:
            text = cached["page_texts"].get(str(pg), "")
            preview = text[:100].replace("\n", " ") if text else "(no text)"
            logger.info("    Page %d: score=%d (%d chars) -- %s...", pg, sc, tl, preview)

# Show segments
if segments:
    logger.info("=" * 60)
    logger.info("SEGMENTS FOUND: %d", len(segments))
    for name, rev in segments.items():
        if isinstance(rev, (int, float)):
            logger.info("  %-40s %15s", name, f"{rev:,.0f}")
        else:
            logger.info("  %-40s %s", name, rev)
else:
    logger.info("  No segments extracted yet. Run again to process next batch.")
logger.info("=" * 60)
