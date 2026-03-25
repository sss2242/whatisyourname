"""Fuzzy PDF financial table parser -- LLM-free extraction fallback.

Extracts structured financial data from SEBI-format and IFRS financial
result PDFs using camelot-py (or pdfplumber fallback) + fuzzy string
matching.  No LLM required.

Strategy:
1. Use camelot-py (stream flavor) for high-accuracy table extraction
   from complex multi-column financial PDFs.  Falls back to pdfplumber
   if camelot is not installed.
2. Fuzzy-match row labels against a canonical concept dictionary using
   difflib.SequenceMatcher with configurable similarity threshold.
3. Parse numeric values handling Indian formats (lakhs, crores, commas,
   parenthetical negatives).
4. Return canonical long-format DataFrame with filing_date + report_date.

camelot-py (MIT license) is specifically designed for financial table
extraction and handles bordered + borderless tables with ~98% accuracy
on SEBI-format results.

Usage:
    from operator1.clients.fuzzy_pdf_parser import extract_financials_from_pdf
    rows = extract_financials_from_pdf(pdf_bytes, filing_date, report_date)
"""

from __future__ import annotations

import io
import logging
import re
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

# ---------------------------------------------------------------------------
# Per-market additional concept patterns.
# Indian SEBI results use specific terminology not found in IFRS filings.
# These supplement the base dictionaries when market_id is provided.
# ---------------------------------------------------------------------------

_MARKET_HINTS: dict[str, list[tuple[str, str]]] = {
    "in_bse": [
        # Indian SEBI-format specific labels
        ("value of sales & services", "revenue"),
        ("less: gst recovered", "cost_of_revenue"),
        ("other income", "revenue"),
        ("cost of materials consumed", "cost_of_revenue"),
        ("purchase of stock-in-trade", "cost_of_revenue"),
        ("changes in inventories", "cost_of_revenue"),
        ("employee benefit expense", "sga_expenses"),
        ("employee benefits expense", "sga_expenses"),
        ("finance costs", "interest_expense"),
        ("depreciation and amortisation expense", "sga_expenses"),
        ("profit/(loss) before exceptional items and tax", "ebit"),
        ("exceptional items", "interest_expense"),
        ("profit/(loss) before tax", "ebit"),
        ("profit/(loss) for the period", "net_income"),
        ("total comprehensive income for the period", "net_income"),
        ("paid-up equity share capital", "shares_outstanding"),
        ("other equity", "retained_earnings"),
        ("other equity excluding revaluation reserve", "retained_earnings"),
        ("net worth", "total_equity"),
        ("earnings per equity share", "eps"),
        ("segment revenue", "revenue"),
        ("segment results", "operating_income"),
        ("segment assets", "total_assets"),
        ("segment liabilities", "total_liabilities"),
        ("capital employed", "total_assets"),
        ("debt service coverage ratio", "interest_expense"),
        ("interest service coverage ratio", "interest_expense"),
        ("debt equity ratio", "total_debt"),
        # Consolidated specific
        ("profit attributable to owners", "net_income"),
        ("profit attributable to non-controlling interests", "net_income"),
    ],
    "hk_hkex": [
        # Hong Kong HKFRS / IFRS labels
        ("turnover", "revenue"),
        ("profit from operations", "operating_income"),
        ("profit attributable to equity holders", "net_income"),
        ("profit attributable to shareholders", "net_income"),
        ("total comprehensive income attributable", "net_income"),
        ("bank balances and cash", "cash_and_equivalents"),
        ("bank borrowings", "total_debt"),
        ("trade and other receivables", "receivables"),
        ("trade and other payables", "payables"),
        ("share capital", "shares_outstanding"),
    ],
    "sg_sgx": [
        # Singapore SFRS / IFRS labels
        ("turnover", "revenue"),
        ("profit from operations", "operating_income"),
        ("profit attributable to equity holders", "net_income"),
        ("cash and short-term deposits", "cash_and_equivalents"),
        ("trade and other receivables", "receivables"),
        ("trade and other payables", "payables"),
    ],
    "sa_tadawul": [
        # Saudi IFRS labels (English versions of Arabic terms)
        ("sales", "revenue"),
        ("cost of sales", "cost_of_revenue"),
        ("gross profit", "gross_profit"),
        ("operating profit", "operating_income"),
        ("zakat", "taxes"),
        ("zakat and income tax", "taxes"),
        ("net profit for the period", "net_income"),
        ("total shareholders equity", "total_equity"),
        ("retained earnings", "retained_earnings"),
    ],
    "ca_sedar": [
        # Canadian IFRS labels
        ("revenues", "revenue"),
        ("cost of sales", "cost_of_revenue"),
        ("selling general and administrative", "sga_expenses"),
        ("income before income taxes", "ebit"),
        ("provision for income taxes", "taxes"),
        ("net earnings", "net_income"),
        ("net income attributable to shareholders", "net_income"),
        ("total shareholders equity", "total_equity"),
        ("retained earnings", "retained_earnings"),
    ],
    "au_asx": [
        # Australian AASB / IFRS labels
        ("sales revenue", "revenue"),
        ("cost of sales", "cost_of_revenue"),
        ("profit before income tax", "ebit"),
        ("income tax expense", "taxes"),
        ("net profit after tax", "net_income"),
        ("profit attributable to members", "net_income"),
        ("total shareholders equity", "total_equity"),
        ("retained profits", "retained_earnings"),
    ],
}

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
    market_id: str = "",
    similarity_threshold: float = 0.65,
) -> list[dict[str, Any]]:
    """Extract financial line items from a PDF using fuzzy matching.

    Uses camelot-py for table extraction (98%+ accuracy on financial
    tables) with pdfplumber as fallback.

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
    market_id:
        Market identifier (e.g. 'in_bse', 'hk_hkex').  When provided,
        market-specific concept patterns are prepended to the dictionary
        for higher-priority matching of regional terminology.
    similarity_threshold:
        Minimum SequenceMatcher ratio to accept a fuzzy match (0-1).
        Lower = more permissive, higher = stricter.

    Returns
    -------
    List of dicts with keys: concept, value, filing_date, report_date.
    """
    if statement_type == "income":
        concepts = list(_INCOME_CONCEPTS)
    elif statement_type == "balance":
        concepts = list(_BALANCE_CONCEPTS)
    elif statement_type == "cashflow":
        concepts = list(_CASHFLOW_CONCEPTS)
    else:
        concepts = list(_ALL_CONCEPTS)

    # Prepend market-specific hints (higher priority = checked first)
    market_hints = _MARKET_HINTS.get(market_id, [])
    if market_hints:
        concepts = market_hints + concepts
        logger.debug(
            "Fuzzy parser: added %d market hints for %s",
            len(market_hints), market_id,
        )

    # Try camelot first (best table extraction quality)
    rows = _extract_with_camelot(pdf_bytes, concepts, filing_date, report_date, similarity_threshold)

    # Fall back to pdfplumber if camelot isn't available or found nothing
    if not rows:
        rows = _extract_with_pdfplumber(pdf_bytes, concepts, filing_date, report_date, similarity_threshold)

    # Post-extraction validation: remove obviously wrong values
    rows = _validate_extracted_rows(rows)

    if rows:
        logger.info(
            "Fuzzy PDF parser: %d concepts extracted (threshold=%.2f)",
            len(rows), similarity_threshold,
        )
    return rows


def _extract_with_camelot(
    pdf_bytes: bytes,
    concepts: list[tuple[str, str]],
    filing_date: str,
    report_date: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Extract using camelot-py (high-accuracy table detection)."""
    try:
        import camelot
    except ImportError:
        logger.debug("camelot-py not installed, skipping camelot extraction")
        return []

    import tempfile
    import os

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    # camelot needs a file path, not bytes
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        # First determine which pages have financial content using pdfplumber
        # for page scoring, then extract tables from those pages with camelot
        financial_pages = _find_financial_pages(pdf_bytes)
        if not financial_pages:
            # Default: scan pages 1-20 (most financial tables are in first 20 pages)
            financial_pages = list(range(1, 21))

        # Convert to camelot page string (1-indexed, comma-separated)
        page_str = ",".join(str(p) for p in financial_pages[:15])

        # Stream flavor works best for borderless SEBI-format tables
        try:
            tables = camelot.read_pdf(tmp_path, pages=page_str, flavor="stream")
        except Exception:
            # Try lattice for bordered tables
            try:
                tables = camelot.read_pdf(tmp_path, pages=page_str, flavor="lattice")
            except Exception as exc:
                logger.debug("camelot extraction failed: %s", exc)
                return []

        logger.debug("camelot found %d tables on pages %s", len(tables), page_str)

        for table in tables:
            if table.shape[0] < 2:
                continue
            df = table.df
            for _, row in df.iterrows():
                cells = row.tolist()
                if not cells or not cells[0]:
                    continue

                label = _clean_label(str(cells[0]))
                if not label or len(label) < 3:
                    continue

                best_match = _fuzzy_match_concept(label, concepts, threshold)
                if not best_match or best_match in seen:
                    continue

                value = _extract_number_from_row(cells[1:])
                if value is None:
                    continue

                rows.append({
                    "concept": best_match,
                    "value": value,
                    "filing_date": filing_date,
                    "report_date": report_date,
                })
                seen.add(best_match)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return rows


def _find_financial_pages(pdf_bytes: bytes) -> list[int]:
    """Score pages and return 1-indexed page numbers with financial content."""
    try:
        import pdfplumber
    except ImportError:
        return []

    pages: list[int] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
        for i, page in enumerate(doc.pages):
            text = page.extract_text() or ""
            score = _page_score(text)
            if score > 0.2:
                pages.append(i + 1)  # camelot uses 1-indexed pages

    return pages


def _extract_with_pdfplumber(
    pdf_bytes: bytes,
    concepts: list[tuple[str, str]],
    filing_date: str,
    report_date: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Fallback: extract using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("Neither camelot-py nor pdfplumber installed")
        return []

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
        scored_pages = _score_pages(doc)
        top_pages = [p for p in scored_pages if p[1] > 0.1][:10]

        for page_idx, score in top_pages:
            page = doc.pages[page_idx]
            tables = page.extract_tables()

            for table in tables:
                if not table or len(table) < 2:
                    continue
                rows.extend(_extract_from_table(
                    table, concepts, seen, filing_date, report_date, threshold,
                ))

            if not tables:
                text = page.extract_text()
                if text:
                    rows.extend(_extract_from_text(
                        text, concepts, seen, filing_date, report_date, threshold,
                    ))

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
    """Fuzzy-match table rows against financial concepts.

    Uses column scoring to identify the most likely "current period"
    value column before extraction.  This prevents cross-column
    contamination where OCR garbage in adjacent columns gets picked
    up instead of the correct number.

    Indian SEBI-format PDFs typically have:
    - Col 0: Labels (Particulars)
    - Cols 1-2: Sometimes empty or OCR artifacts
    - Col 3: Current quarter (most populated with valid numbers)
    - Col 4+: Prior periods
    """
    rows: list[dict] = []
    if not table or len(table) < 2:
        return rows

    # Step 1: Score each column by how many valid numbers it contains.
    # The "current period" column has the most valid numbers.
    n_cols = max(len(r) for r in table if r)
    col_valid_counts = [0] * n_cols

    for row in table:
        if not row:
            continue
        for ci in range(1, min(len(row), n_cols)):
            if row[ci] is None:
                continue
            val = _parse_indian_number(str(row[ci]).strip())
            if val is not None and abs(val) >= 1.0:
                col_valid_counts[ci] += 1

    # Pick the column with the most valid numbers (skip col 0 = labels)
    if sum(col_valid_counts[1:]) == 0:
        return rows

    best_col = max(range(1, len(col_valid_counts)), key=lambda i: col_valid_counts[i])

    # Step 2: Extract values only from the identified best column
    for row in table:
        if not row or not row[0]:
            continue

        label = _clean_label(str(row[0]))
        if not label or len(label) < 3:
            continue

        best_match = _fuzzy_match_concept(label, concepts, threshold)
        if not best_match or best_match in seen:
            continue

        # Primary: get value from the best column
        value = None
        if best_col < len(row) and row[best_col] is not None:
            value = _parse_indian_number(str(row[best_col]).strip())

        # Fallback: try adjacent column if best_col is empty for this row
        if value is None and best_col + 1 < len(row) and row[best_col + 1] is not None:
            value = _parse_indian_number(str(row[best_col + 1]).strip())
        if value is None and best_col - 1 >= 1 and row[best_col - 1] is not None:
            value = _parse_indian_number(str(row[best_col - 1]).strip())

        if value is None:
            continue

        rows.append({
            "concept": best_match,
            "value": value,
            "filing_date": filing_date,
            "report_date": report_date,
        })
        seen.add(best_match)

    # Post-extraction validation: remove obviously wrong values
    rows = _validate_extracted_rows(rows)
    return rows


def _validate_extracted_rows(rows: list[dict]) -> list[dict]:
    """Remove extracted values that fail basic sanity checks.

    Rules:
    - Goodwill and intangible_assets must be non-negative
    - total_assets must not equal total_liabilities exactly (segment confusion)
    - gross_profit must be at least 1% of revenue if both present
    - If total_assets < total_equity, drop total_assets (likely segment value)
    """
    if not rows:
        return rows

    values = {r["concept"]: r["value"] for r in rows}

    to_remove: set[str] = set()

    # Goodwill and intangibles can't be negative
    if values.get("goodwill", 0) < 0:
        to_remove.add("goodwill")
    if values.get("intangible_assets", 0) < 0:
        to_remove.add("intangible_assets")

    # If total_assets == total_liabilities exactly, both are likely
    # from the same "Total Equity and Liabilities" or segment row
    ta = values.get("total_assets")
    tl = values.get("total_liabilities")
    if ta is not None and tl is not None and ta == tl:
        to_remove.add("total_liabilities")
        # Also check if total_assets is actually "segment assets" (too big)
        te = values.get("total_equity")
        if te is not None and ta > te * 5:
            to_remove.add("total_assets")

    # If total_assets < total_equity, it's a segment value not real total_assets
    if ta is not None and values.get("total_equity") is not None:
        if ta < values["total_equity"]:
            to_remove.add("total_assets")

    # gross_profit should be a reasonable fraction of revenue
    rev = values.get("revenue")
    gp = values.get("gross_profit")
    if rev is not None and gp is not None and rev > 0:
        if gp / rev < 0.01 or gp > rev:
            to_remove.add("gross_profit")

    if to_remove:
        rows = [r for r in rows if r["concept"] not in to_remove]
        logger.debug("Validation removed: %s", to_remove)

    # Accounting identity derivation:
    # Total Assets = Total Liabilities + Total Equity
    # If we have two of three, derive the third.
    # If the extracted total_assets fails this check, replace it.
    rows = _apply_accounting_identities(rows)

    return rows


def _apply_accounting_identities(rows: list[dict]) -> list[dict]:
    """Apply double-entry bookkeeping identities to validate and derive values.

    Identity: Total Assets = Total Liabilities + Total Equity

    Cases:
    1. All three present but don't balance -> trust equity + liabilities,
       recompute assets (equity is usually the most reliable from PDFs).
    2. Assets and equity present but no liabilities -> derive liabilities.
    3. Equity and liabilities present but no assets -> derive assets.
    4. Only equity or only assets present -> keep as-is (can't validate).
    """
    if not rows:
        return rows

    values = {r["concept"]: r for r in rows}
    ta_row = values.get("total_assets")
    tl_row = values.get("total_liabilities")
    te_row = values.get("total_equity")

    ta = ta_row["value"] if ta_row else None
    tl = tl_row["value"] if tl_row else None
    te = te_row["value"] if te_row else None

    # Get a reference filing_date/report_date from any existing row
    ref = rows[0] if rows else {}
    fd = ref.get("filing_date", "")
    rd = ref.get("report_date", "")

    if te is not None and tl is not None:
        # We have equity and liabilities -- derive or validate assets
        derived_ta = te + tl
        if ta is not None:
            # Check if extracted total_assets is reasonably close to identity
            if derived_ta > 0 and abs(ta - derived_ta) / derived_ta > 0.1:
                # More than 10% off -- replace with derived value
                logger.debug(
                    "Accounting identity fix: total_assets %.0f -> %.0f "
                    "(= equity %.0f + liabilities %.0f)",
                    ta, derived_ta, te, tl,
                )
                ta_row["value"] = derived_ta
        else:
            # Derive total_assets from identity
            rows.append({
                "concept": "total_assets",
                "value": derived_ta,
                "filing_date": fd,
                "report_date": rd,
            })
            logger.debug(
                "Accounting identity derived: total_assets = %.0f "
                "(equity %.0f + liabilities %.0f)",
                derived_ta, te, tl,
            )

    elif te is not None and ta is not None and tl is None:
        # Derive total_liabilities from identity
        derived_tl = ta - te
        if derived_tl >= 0:
            rows.append({
                "concept": "total_liabilities",
                "value": derived_tl,
                "filing_date": fd,
                "report_date": rd,
            })
            logger.debug(
                "Accounting identity derived: total_liabilities = %.0f "
                "(assets %.0f - equity %.0f)",
                derived_tl, ta, te,
            )

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

    Uses exact substring match first (fast), then rapidfuzz WRatio
    for fuzzy matching (combines ratio, partial_ratio, token_sort_ratio,
    and token_set_ratio with optimal weights).

    Falls back to difflib if rapidfuzz is not installed.

    Returns the canonical field name, or None if no match.
    """
    # Exact substring match (fast path -- no fuzzy overhead)
    for pattern, canonical in concepts:
        if pattern in label:
            return canonical

    # Build choices dict for rapidfuzz: {pattern: canonical}
    choices = {pattern: canonical for pattern, canonical in concepts}

    # Try rapidfuzz (much faster and more accurate than difflib)
    try:
        from rapidfuzz import process, fuzz
        # WRatio is the best general-purpose scorer -- it automatically
        # picks the best combination of ratio, partial_ratio,
        # token_sort_ratio, and token_set_ratio.
        result = process.extractOne(
            label,
            choices.keys(),
            scorer=fuzz.WRatio,
            score_cutoff=threshold * 100,  # rapidfuzz uses 0-100 scale
        )
        if result is not None:
            matched_pattern, score, _ = result
            return choices[matched_pattern]
        return None
    except ImportError:
        pass

    # Fallback: difflib SequenceMatcher
    from difflib import SequenceMatcher
    best_ratio = 0.0
    best_canonical = None

    for pattern, canonical in concepts:
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


# ---------------------------------------------------------------------------
# Shareholder / ownership data extraction from PDFs
# ---------------------------------------------------------------------------

# Keywords that identify pages containing shareholder data (English base)
_SHAREHOLDER_KEYWORDS = [
    "shareholding pattern", "major shareholders", "ownership structure",
    "substantial shareholders", "top shareholders", "significant shareholders",
    "beneficial owners", "institutional holders", "holder name",
    "persons with significant control", "register of members",
    "shareholding of promoters", "category of shareholders",
    "public shareholding", "promoter and promoter group",
    "holding of specified securities", "shares held",
    "percentage of holding", "% of total",
    "directors and key managerial personnel",
]

# Per-region shareholder keywords (added to base keywords when market_id provided)
_SHAREHOLDER_KEYWORDS_BY_MARKET: dict[str, list[str]] = {
    "in_bse": [
        # SEBI format (Regulation 31)
        "shareholding of promoter", "public shareholding",
        "shares held by custodians", "non-institutions",
        "central government", "state government",
        "mutual funds", "financial institutions",
        "foreign institutional investors", "foreign portfolio",
        "bodies corporate", "individuals", "nri",
        "statement showing shareholding pattern",
    ],
    "jp_jquants": [
        # Japanese
        "大株主の状況", "所有者別状況", "株式の状況",
        "major shareholders", "status of shareholders",
    ],
    "kr_dart": [
        # Korean
        "주주현황", "최대주주", "주요주주", "소액주주",
        "지분율", "보유주식수",
        "shareholder status", "largest shareholder",
    ],
    "tw_mops": [
        # Chinese (Traditional)
        "股東名簿", "持股比例", "大股東", "股權結構",
        "主要股東", "股東持股",
    ],
    "br_cvm": [
        # Portuguese
        "composição acionária", "ações detidas", "participação acionária",
        "acionistas", "controlador", "free float",
        "posição acionária", "quadro societário",
    ],
    "cl_cmf": [
        # Spanish
        "composición accionaria", "principales accionistas",
        "estructura de propiedad", "participación accionaria",
    ],
    "fr_esef": [
        # French
        "répartition du capital", "actionnariat",
        "principaux actionnaires", "structure du capital",
        "droits de vote", "capital social",
    ],
    "de_esef": [
        # German
        "aktionärsstruktur", "stimmrechte", "anteilseigner",
        "hauptaktionäre", "aktienbesitz", "grundkapital",
        "stimmrechtsmitteilung",
    ],
    "sa_tadawul": [
        # Arabic + English
        "هيكل الملكية", "المساهمون الرئيسيون",
        "نسبة الملكية", "الأسهم المملوكة",
        "ownership structure", "major shareholders",
    ],
    "sg_sgx": [
        "substantial shareholders", "directors interests",
        "statistics of shareholdings", "distribution of shareholdings",
    ],
    "au_asx": [
        "substantial shareholders", "top 20 shareholders",
        "distribution of equity securities", "voting rights",
    ],
    "ca_sedar": [
        "principal shareholders", "voting securities",
        "ownership of securities", "control persons",
    ],
    "za_jse": [
        "shareholder spread", "major shareholders",
        "beneficial shareholders", "fund managers",
    ],
    "hk_hkex": [
        # Chinese (Simplified) + English
        "股东", "持股", "主要股东", "股权结构",
        "substantial shareholders", "disclosure of interests",
    ],
    "cn_sse": [
        # Chinese (Simplified)
        "股东", "持股", "十大股东", "前十大股东",
        "股权结构", "流通股股东", "实际控制人",
    ],
    "ae_dfm": [
        # Arabic + English
        "المساهمون", "هيكل الملكية",
        "shareholders", "ownership",
    ],
    "mx_bmv": [
        # Spanish
        "estructura accionaria", "principales accionistas",
        "tenencia accionaria", "capital social",
    ],
}

# Keywords that indicate a row is a holder name (not a section header)
_HOLDER_ROW_INDICATORS = [
    "ltd", "limited", "inc", "corp", "llc", "plc", "nv", "ag", "sa", "sas",
    "gmbh", "fund", "trust", "bank", "capital", "asset", "management",
    "investment", "insurance", "pension", "securities", "holdings",
    "group", "partners", "advisors", "state", "government", "mutual",
    "fidelity", "vanguard", "blackrock", "jpmorgan", "goldman",
    "amundi", "ubs", "hsbc", "citibank", "nomura", "dws",
]


def extract_shareholders_from_pdf(
    pdf_bytes: bytes,
    filing_date: str = "",
    market_id: str = "",
) -> list[dict[str, Any]]:
    """Extract shareholder/ownership data from a PDF.

    Finds pages with shareholding pattern tables and extracts holder
    names with share counts and percentages.  Works with SEBI
    (India), IFRS annual reports, and other standard formats.

    Parameters
    ----------
    pdf_bytes:
        Raw PDF file content.
    filing_date:
        ISO date string of when the filing was published.
    market_id:
        Market identifier for logging.

    Returns
    -------
    List of dicts with: name, shares, percentage, holder_type, source.
    """
    holders: list[dict[str, Any]] = []

    try:
        import pdfplumber
    except ImportError:
        logger.debug("pdfplumber not installed, cannot extract shareholders from PDF")
        return holders

    # Build keyword list: base + market-specific
    keywords = list(_SHAREHOLDER_KEYWORDS)
    market_kw = _SHAREHOLDER_KEYWORDS_BY_MARKET.get(market_id, [])
    if market_kw:
        keywords = market_kw + keywords
        logger.debug("Shareholder parser: added %d market hints for %s", len(market_kw), market_id)

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
            # Find pages with shareholder content
            sh_pages = []
            for i, page in enumerate(doc.pages):
                text = (page.extract_text() or "").lower()
                score = sum(1 for kw in keywords if kw in text)
                if score >= 2:
                    sh_pages.append((i, score))

            sh_pages.sort(key=lambda x: -x[1])
            if not sh_pages:
                return holders

            logger.debug("Shareholder pages found: %d", len(sh_pages))

            seen_names: set[str] = set()

            for page_idx, _ in sh_pages[:5]:
                page = doc.pages[page_idx]
                tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # Try to identify columns: look for header row with
                    # "name", "shares", "%", "holding" keywords
                    header = table[0] if table[0] else []
                    header_lower = [str(h).lower().strip() if h else "" for h in header]

                    name_col = -1
                    shares_col = -1
                    pct_col = -1

                    for ci, h in enumerate(header_lower):
                        if any(w in h for w in ["name", "shareholder", "holder", "category"]):
                            name_col = ci
                        elif any(w in h for w in ["shares", "number", "quantity", "nos"]):
                            shares_col = ci
                        elif any(w in h for w in ["%", "percent", "holding", "proportion"]):
                            pct_col = ci

                    # If no clear header, assume col 0 = name, last cols = numbers
                    if name_col < 0:
                        name_col = 0

                    # Extract rows
                    for row in table[1:]:
                        if not row or not row[name_col]:
                            continue

                        name = str(row[name_col]).strip()
                        if not name or len(name) < 3:
                            continue

                        # Skip section headers and totals
                        name_lower = name.lower()
                        if any(w in name_lower for w in [
                            "total", "grand total", "sub-total", "subtotal",
                            "category", "particulars", "description",
                            "sl. no", "sr. no", "s.no",
                        ]):
                            continue

                        # Check if this looks like a holder name
                        is_holder = any(w in name_lower for w in _HOLDER_ROW_INDICATORS)
                        # Also accept if the row has numeric values
                        has_numbers = any(
                            _parse_indian_number(str(c).strip()) is not None
                            for c in row[1:] if c
                        )

                        if not is_holder and not has_numbers:
                            continue

                        if name in seen_names:
                            continue
                        seen_names.add(name)

                        # Extract shares and percentage
                        shares = 0
                        pct = 0.0

                        if shares_col >= 0 and shares_col < len(row) and row[shares_col]:
                            val = _parse_indian_number(str(row[shares_col]).strip())
                            if val is not None:
                                shares = int(val)

                        if pct_col >= 0 and pct_col < len(row) and row[pct_col]:
                            val = _parse_indian_number(str(row[pct_col]).strip())
                            if val is not None:
                                pct = round(val, 4)

                        # If no identified columns, try extracting from any cell
                        if shares == 0 and pct == 0.0:
                            for c in row[1:]:
                                if c is None:
                                    continue
                                val = _parse_indian_number(str(c).strip())
                                if val is not None:
                                    if val > 100:
                                        shares = int(val)
                                    elif 0 < val <= 100:
                                        pct = round(val, 4)

                        if shares > 0 or pct > 0:
                            # Determine holder type from name
                            holder_type = "institutional"
                            if any(w in name_lower for w in ["promoter", "director", "founder", "family"]):
                                holder_type = "promoter"
                            elif any(w in name_lower for w in ["public", "individual", "retail"]):
                                holder_type = "retail"
                            elif any(w in name_lower for w in ["fund", "mutual", "pension", "insurance"]):
                                holder_type = "fund"

                            holders.append({
                                "name": name,
                                "shares": shares,
                                "value": 0.0,
                                "percentage": pct,
                                "holder_type": holder_type,
                                "date_reported": filing_date,
                                "source": f"fuzzy_pdf_{market_id}" if market_id else "fuzzy_pdf",
                            })

        if holders:
            logger.info(
                "Fuzzy PDF shareholder extraction: %d holders from %d pages",
                len(holders), len(sh_pages),
            )

    except Exception as exc:
        logger.debug("Fuzzy PDF shareholder extraction failed: %s", exc)

    return holders


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
