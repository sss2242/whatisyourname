#!/usr/bin/env python3
"""Test script: run batched OCR on Unilever PDF and check results.

Usage:
    python scripts/test_ocr_batch.py

Each run processes the next batch of 50 pages from cache.
Re-run to continue from where the last run left off.
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("test_ocr")

# Load saved PDF
pdf_path = "cache/unilever_test.pdf"
try:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    logger.info("Loaded PDF: %d bytes", len(pdf_bytes))
except FileNotFoundError:
    logger.error("PDF not found at %s. Run the download step first.", pdf_path)
    sys.exit(1)

from operator1.clients.fuzzy_pdf_parser import (
    _ocr_extract_segments,
    _load_ocr_cache,
    _ocr_cache_path,
)

# Check existing cache
cached = _load_ocr_cache(pdf_bytes)
if cached:
    logger.info(
        "Existing OCR cache: %d pages OCR'd, last_page=%d",
        len(cached.get("page_texts", {})),
        cached.get("last_page_processed", -1),
    )
else:
    logger.info("No existing OCR cache -- starting fresh")

# UK/IFRS + Unilever-specific segment keywords
keywords = [
    "segment", "revenue", "operating profit", "turnover",
    "beauty", "personal care", "home care", "nutrition",
    "ice cream", "foods", "refreshment",
    "beauty & wellbeing", "personal care", "home care",
    "nutrition", "ice cream",
]

logger.info("Running batched OCR (50 pages/batch)...")
segments = _ocr_extract_segments(
    pdf_bytes,
    keywords,
    min_page_score=2,
    max_pages=10,
    market_id="uk_companies_house",
    batch_size=50,
)

# Check cache state after run
cache_file = _ocr_cache_path(pdf_bytes)
logger.info("Cache file: %s (exists=%s)", cache_file, cache_file.exists())
if cache_file.exists():
    cache_data = json.loads(cache_file.read_text())
    n_cached = len(cache_data.get("page_texts", {}))
    last_page = cache_data.get("last_page_processed", -1)
    n_pages = cache_data.get("n_pages", 0)
    logger.info(
        "Cache state: %d/%d pages OCR'd (last_page=%d)",
        n_cached, n_pages, last_page,
    )

    # Show which pages had text
    pages_with_text = sorted(int(k) for k in cache_data.get("page_texts", {}).keys())
    logger.info("Pages with OCR text: %d pages", len(pages_with_text))
    if pages_with_text:
        logger.info("  Range: %d-%d", min(pages_with_text), max(pages_with_text))

    # Show scored pages
    scored = []
    for pg_str, text in cache_data.get("page_texts", {}).items():
        lower = text.lower()
        score = sum(1 for kw in keywords if kw in lower)
        if score >= 2:
            scored.append((int(pg_str), score, len(text)))
    scored.sort(key=lambda x: -x[1])
    if scored:
        logger.info("Top scored pages (score >= 2):")
        for pg, sc, tlen in scored[:10]:
            preview = cache_data["page_texts"][str(pg)][:120].replace("\n", " ")
            logger.info("  Page %d: score=%d, %d chars -- %s...", pg, sc, tlen, preview)

# Report segments
if segments:
    logger.info("=" * 60)
    logger.info("SEGMENTS FOUND: %d", len(segments))
    logger.info("=" * 60)
    for name, rev in segments.items():
        if isinstance(rev, (int, float)):
            logger.info("  %-40s %15s", name, f"{rev:,.0f}")
        else:
            logger.info("  %-40s %s", name, rev)
else:
    logger.info("No segments extracted yet. Re-run to process next batch.")
    logger.info("(Each run processes 50 more pages from cache)")
