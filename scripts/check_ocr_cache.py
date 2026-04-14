#!/usr/bin/env python3
"""Check OCR cache state and show scored pages."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

with open("cache/unilever_test.pdf", "rb") as f:
    pdf_bytes = f.read()

from operator1.clients.fuzzy_pdf_parser import _ocr_cache_path, _load_ocr_cache

cached = _load_ocr_cache(pdf_bytes)
n = len(cached.get("page_texts", {}))
last = cached.get("last_page_processed", -1)
total = cached.get("n_pages", 0)
print(f"Cache: {n} pages with OCR text, last_page_processed={last}, total_pages={total}")

keywords = [
    "segment", "revenue", "operating profit", "turnover",
    "beauty", "personal care", "home care", "nutrition", "ice cream",
]

scored = []
for pg_str, text in cached.get("page_texts", {}).items():
    lower = text.lower()
    score = sum(1 for kw in keywords if kw in lower)
    if score >= 1:
        scored.append((int(pg_str), score, len(text)))

scored.sort(key=lambda x: -x[1])
print(f"Scored pages with >= 1 keyword: {len(scored)}")
for pg, sc, tl in scored[:10]:
    preview = cached["page_texts"][str(pg)][:200].replace("\n", " ")
    print(f"  Page {pg}: score={sc}, {tl} chars")
    print(f"    Preview: {preview}...")
    print()
