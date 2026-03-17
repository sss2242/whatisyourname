"""Fuzzy PDF financial table parser -- LLM-free extraction fallback.

Extracts structured financial data from SEBI-format and IFRS financial
result PDFs using pdfplumber + fuzzy string matching.  No LLM required.

Strategy:
1. Score each PDF page for financial content density (numbers, keywords).
2. Extract tables from the top-scoring pages only.
3. Fuzzy-match row labels against a canonical concept dictionary.
4. Parse numeric values handling Indian formats (lakhs, crores, commas,
   parenthetical negatives).
5. Return canonical long-format DataFrame with filing_date + report_date.

This module is used as a fallback when no LLM key is available.  The
LLM extraction path (via LLMFilingExtractor) produces higher-quality
results for complex PDFs but requires an API key.

Usage:
    from operator1.clients.fuzzy_pdf_parser import extract_financials_from_pdf
    rows = extract_financials_from_pdf(pdf_bytes, filing_date, report_date)
"""

from __future__ import annotations

import io
import logging
import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical concept dictionaries for fuzzy matching.
# Keys are lowercase label patterns, values are canonical field names.
# Ordered from most specific to least specific to avoid false matches.
# ---------------------------------------------------------------------------

_INCOME_CONCEPTS: list[tuple[str, str]] = [
    ("revenue from operations", "revenue"),
    ("total income from operations", "revenue"),
    ("net sales", "revenue"),
    ("total income", "revenue"),
    ("value of sales", "revenue"),
    ("cost of materials consumed", "cost_of_revenue"),
    ("cost of goods sold", "cost_of_revenue"),
    ("cost of revenue", "cost_of_revenue"),
    ("total expenses", "cost_of_revenue"),
    ("purchase of stock-in-trade", "cost_of_revenue"),
    ("gross profit", "gross_profit"),
    ("profit from operations before other income", "operating_income"),
    ("profit from operations", "operating_income"),
    ("operating profit", "operating_income"),
    ("profit before exceptional items and tax", "ebit"),
    ("profit before tax", "ebit"),
    ("profit/(loss) before tax", "ebit"),
    ("earnings before interest and tax", "ebit"),
    ("ebitda", "ebitda"),
    ("profit after tax", "net_income"),
    ("profit/(loss) after tax", "net_income"),
    ("profit for the period", "net_income"),
    ("net profit", "net_income"),
    ("net profit/(loss)", "net_income"),
    ("total comprehensive income", "net_income"),
    ("tax expense", "taxes"),
    ("current tax", "taxes"),
    ("income tax expense", "taxes"),
    ("finance costs", "interest_expense"),
    ("interest expense", "interest_expense"),
    ("depreciation and amortisation", "sga_expenses"),
    ("depreciation", "sga_expenses"),
    ("employee benefits expense", "sga_expenses"),
    ("basic earnings per share", "eps"),
    ("basic eps", "eps"),
    ("diluted earnings per share", "eps_diluted"),
    ("diluted eps", "eps_diluted"),
]

_BALANCE_CONCEPTS: list[tuple[str, str]] = [
    ("total assets", "total_assets"),
    ("total equity", "total_equity"),
    ("total equity and liabilities", "total_assets"),
    ("total liabilities", "total_liabilities"),
    ("net worth", "total_equity"),
    ("shareholders' funds", "total_equity"),
    ("shareholder funds", "total_equity"),
    ("current assets", "current_assets"),
    ("total current assets", "current_assets"),
    ("current liabilities", "current_liabilities"),
    ("total current liabilities", "current_liabilities"),
    ("cash and cash equivalents", "cash_and_equivalents"),
    ("cash and bank balances", "cash_and_equivalents"),
    ("trade receivables", "receivables"),
    ("inventories", "inventory"),
    ("trade payables", "payables"),
    ("short-term borrowings", "short_term_debt"),
    ("current borrowings", "short_term_debt"),
    ("long-term borrowings", "long_term_debt"),
    ("non-current borrowings", "long_term_debt"),
    ("total borrowings", "total_debt"),
    ("total debt", "total_debt"),
    ("retained earnings", "retained_earnings"),
    ("reserves and surplus", "retained_earnings"),
    ("goodwill", "goodwill"),
    ("intangible assets", "intangible_assets"),
    ("paid-up equity share capital", "shares_outstanding"),
    ("paid up share capital", "shares_outstanding"),
    ("equity share capital", "shares_outstanding"),
]

_CASHFLOW_CONCEPTS: list[tuple[str, str]] = [
    ("cash flows from operating activities", "operating_cash_flow"),
    ("net cash from operating activities", "operating_cash_flow"),
    ("cash generated from operations", "operating_cash_flow"),
    ("cash flows from investing activities", "investing_cf"),
    ("net cash used in investing activities", "investing_cf"),
    ("cash flows from financing activities", "financing_cf"),
    ("net cash from financing activities", "financing_cf"),
    ("purchase of property plant and equipment", "capex"),
    ("capital expenditure", "capex"),
    ("dividends paid", "dividends_paid"),
    ("dividend paid", "dividends_paid"),
    ("free cash flow", "free_cash_flow"),
]

_ALL_CONCEPTS = _INCOME_CONCEPTS + _BALANCE_CONCEPTS + _CASHFLOW_CONCEPTS

# Keywords that indicate a page has financial content
_FINANCIAL_KEYWORDS = [
    "revenue", "profit", "loss", "income", "expense", "assets",
    "liabilities", "equity", "cash", "earnings", "ebitda", "ebit",
    "tax", "depreciation", "borrowings", "receivables", "payables",
    "dividend", "capital", "turnover", "net worth", "balance sheet",
    "statement of profit", "financial results", "standalone", "consolidated",
]

# Keywords for noise pages to skip
_NOISE_KEYWORDS = [
    "auditor", "chartered accountant", "deloitte", "kpmg", "pwc",
    "ernst & young", "corporate governance", "risk factors",
    "directors' report", "management discussion", "secretarial",
    "related party", "notes to", "accounting policies",
]


def extract_financials_from_pdf(
    pdf_bytes: bytes,
    filing_date: str = "",
    report_date: str = "",
    statement_type: str = "",
    similarity_threshold: float = 0.65,
) -> list[dict[str, Any]]:
    """Extract financial line items from a PDF using fuzzy matching.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF file content.
    filing_date:
        ISO date string of when the filing was published.
    report_date:
        ISO date string of the fiscal period end date.
    statement_type:
        If set ('income', 'balance', 'cashflow'), only match concepts
        for that statement type.  If empty, match all.
    similarity_threshold:
        Minimum SequenceMatcher ratio to accept a fuzzy match (0-1).
        Lower = more permissive, higher = stricter.

    Returns
    -------
    List of dicts with keys: concept, value, filing_date, report_date.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed -- fuzzy PDF parser unavailable")
        return []

    # Select concept list based on statement type
    if statement_type == "income":
        concepts = _INCOME_CONCEPTS
    elif statement_type == "balance":
        concepts = _BALANCE_CONCEPTS
    elif statement_type == "cashflow":
        concepts = _CASHFLOW_CONCEPTS
    else:
        concepts = _ALL_CONCEPTS

    rows: list[dict[str, Any]] = []
    seen_concepts: set[str] = set()

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
        # Step 1: Score and select relevant pages
        scored_pages = _score_pages(doc)
        top_pages = [p for p in scored_pages if p[1] > 0.1][:10]

        if not top_pages:
            logger.debug("No financial pages found in PDF (%d pages total)", len(doc.pages))
            return rows

        logger.debug(
            "Fuzzy parser: %d/%d pages selected (scores: %s)",
            len(top_pages), len(doc.pages),
            [f"p{p[0]+1}={p[1]:.2f}" for p in top_pages[:5]],
        )

        # Step 2: Extract from tables on selected pages
        for page_idx, score in top_pages:
            page = doc.pages[page_idx]
            tables = page.extract_tables()

            for table in tables:
                if not table or len(table) < 2:
                    continue
                table_rows = _extract_from_table(
                    table, concepts, seen_concepts,
                    filing_date, report_date, similarity_threshold,
                )
                rows.extend(table_rows)

            # Step 3: If no tables, try text-based extraction
            if not tables:
                text = page.extract_text()
                if text:
                    text_rows = _extract_from_text(
                        text, concepts, seen_concepts,
                        filing_date, report_date, similarity_threshold,
                    )
                    rows.extend(text_rows)

    if rows:
        logger.info(
            "Fuzzy PDF parser: %d concepts extracted (threshold=%.2f)",
            len(rows), similarity_threshold,
        )
    return rows


def _score_pages(doc) -> list[tuple[int, float]]:
    """Score each page for financial content.  Returns [(page_idx, score)]."""
    scored = []
    for i, page in enumerate(doc.pages):
        text = page.extract_text() or ""
        score = _page_score(text)
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _page_score(text: str) -> float:
    """Score a page for financial table content (0-1)."""
    if not text or len(text) < 50:
        return 0.0

    lower = text.lower()

    # Penalty for noise pages
    noise_count = sum(1 for kw in _NOISE_KEYWORDS if kw in lower)
    if noise_count >= 3:
        return 0.0

    # Financial keyword density
    keyword_hits = sum(1 for kw in _FINANCIAL_KEYWORDS if kw in lower)
    keyword_score = min(keyword_hits / 8.0, 1.0)

    # Number density (financial tables have lots of numbers)
    numbers = re.findall(r"\d[\d,]+\.?\d*", text)
    number_score = min(len(numbers) / 30.0, 1.0)

    # Table structure indicators
    has_table_chars = ("|" in text or "\t" in text)
    table_bonus = 0.2 if has_table_chars else 0.0

    # Combined score
    return (keyword_score * 0.4 + number_score * 0.4 + table_bonus + 0.0) * (1.0 - noise_count * 0.1)


def _extract_from_table(
    table: list[list],
    concepts: list[tuple[str, str]],
    seen: set[str],
    filing_date: str,
    report_date: str,
    threshold: float,
) -> list[dict]:
    """Fuzzy-match table rows against financial concepts."""
    rows: list[dict] = []

    for row in table:
        if not row or not row[0]:
            continue

        # Clean the label cell
        label = _clean_label(str(row[0]))
        if not label or len(label) < 3:
            continue

        # Fuzzy match against concept patterns
        best_match = _fuzzy_match_concept(label, concepts, threshold)
        if not best_match or best_match in seen:
            continue

        # Extract numeric value from the first non-empty value cell
        value = _extract_number_from_row(row[1:])
        if value is None:
            continue

        rows.append({
            "concept": best_match,
            "value": value,
            "filing_date": filing_date,
            "report_date": report_date,
        })
        seen.add(best_match)

    return rows


def _extract_from_text(
    text: str,
    concepts: list[tuple[str, str]],
    seen: set[str],
    filing_date: str,
    report_date: str,
    threshold: float,
) -> list[dict]:
    """Fuzzy-match text lines against financial concepts."""
    rows: list[dict] = []

    for line in text.split("\n"):
        label = _clean_label(line)
        if not label or len(label) < 3:
            continue

        best_match = _fuzzy_match_concept(label, concepts, threshold)
        if not best_match or best_match in seen:
            continue

        # Find numbers on this line
        value = _extract_number_from_text(line)
        if value is None:
            continue

        rows.append({
            "concept": best_match,
            "value": value,
            "filing_date": filing_date,
            "report_date": report_date,
        })
        seen.add(best_match)

    return rows


def _clean_label(text: str) -> str:
    """Clean a cell/line into a matchable label."""
    # Remove serial numbers at the start (e.g. "1.", "a)", "I.")
    text = re.sub(r"^\s*[\divxIVX]+[.)]\s*", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove trailing numbers/amounts
    text = re.sub(r"\s+[\d,.()\-]+\s*$", "", text)
    return text.lower()


def _fuzzy_match_concept(
    label: str,
    concepts: list[tuple[str, str]],
    threshold: float,
) -> str | None:
    """Find the best matching canonical concept for a label.

    Uses exact substring match first (fast), then falls back to
    SequenceMatcher ratio for fuzzy matching.

    Returns the canonical field name, or None if no match.
    """
    # Exact substring match (fast path)
    for pattern, canonical in concepts:
        if pattern in label:
            return canonical

    # Fuzzy match (slow path)
    best_ratio = 0.0
    best_canonical = None

    for pattern, canonical in concepts:
        # Only fuzzy-match if lengths are somewhat similar
        if abs(len(label) - len(pattern)) > max(len(pattern), len(label)) * 0.5:
            continue

        ratio = SequenceMatcher(None, label, pattern).ratio()
        if ratio > best_ratio and ratio >= threshold:
            best_ratio = ratio
            best_canonical = canonical

    return best_canonical


def _extract_number_from_row(cells: list) -> float | None:
    """Extract the first valid number from table row cells.

    Handles Indian number formats: lakhs (1,00,000), crores (1,00,00,000),
    parenthetical negatives (1,234), and mixed content cells.
    """
    for cell in cells[:4]:  # Check first 4 value columns
        if cell is None:
            continue
        val = _parse_indian_number(str(cell))
        if val is not None:
            return val
    return None


def _extract_number_from_text(line: str) -> float | None:
    """Extract the first significant number from a text line."""
    # Find all number patterns
    numbers = re.findall(r"[\(\-]?[\d,]+\.?\d*\)?", line)
    for num_str in numbers:
        val = _parse_indian_number(num_str)
        if val is not None and abs(val) >= 0.01:
            # Skip years (1900-2100)
            if 1900 < abs(val) < 2100 and "." not in num_str:
                continue
            return val
    return None


def _parse_indian_number(text: str) -> float | None:
    """Parse a number string handling Indian conventions.

    Handles:
    - Parenthetical negatives: (1,234) -> -1234
    - Indian commas: 1,00,000 -> 100000
    - Regular commas: 1,234,567 -> 1234567
    - Spaces as thousands separators
    - Negative signs
    """
    text = text.strip()
    if not text:
        return None

    # Check for parenthetical negative
    is_negative = False
    if text.startswith("(") and text.endswith(")"):
        is_negative = True
        text = text[1:-1].strip()
    elif text.startswith("-"):
        is_negative = True
        text = text[1:].strip()

    # Remove all commas and spaces (handles both Indian and Western formats)
    text = text.replace(",", "").replace(" ", "")

    # Must have at least one digit
    if not re.search(r"\d", text):
        return None

    # Extract the number
    match = re.match(r"^(\d+\.?\d*)$", text)
    if not match:
        return None

    try:
        val = float(match.group(1))
        return -val if is_negative else val
    except ValueError:
        return None
