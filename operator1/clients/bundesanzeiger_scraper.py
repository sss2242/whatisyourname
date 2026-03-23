"""Bundesanzeiger scraper -- German financial statement extraction.

The Bundesanzeiger (Federal Gazette) is the official publication platform
for German company financial statements (Jahresabschlüsse, Konzernabschlüsse).
Unlike other EU countries, German ESEF filings are NOT available on
filings.xbrl.org (0 DE filings there). The Bundesanzeiger is Germany's OAM
(Officially Appointed Mechanism).

**Key technical details:**

1. **Search**: GET with Wicket form URL pattern, ``area_select=22`` for
   Rechnungslegung/Finanzberichte (financial reporting category).
   The cookie ``cc=1628606977-805e172265bfdbde-10`` (consent cookie) must
   be set before any request.

2. **CAPTCHA**: Document access requires solving a simple image CAPTCHA
   (6 alphanumeric characters, 250x50 pixels). We use a community-trained
   ONNX neural network model (from ``dre808/bundesanzeiger-scraper``, MIT
   license) for automated solving. Accuracy is ~60-70% per attempt, so
   we retry up to 3 times per document.

3. **Financial data**: Published as HTML documents with ``<table>`` elements
   inside a ``publication_container`` div. Tables contain Bilanz (balance
   sheet), GuV (income statement / Gewinn- und Verlustrechnung), and
   Kapitalflussrechnung (cash flow statement) in German.

4. **Filing metadata**: Search results include company name, document type
   (Jahresabschluss/Konzernabschluss), fiscal period (from/to dates), and
   publication date (V.-Datum = Veröffentlichungsdatum).

Community references:
    - github.com/dre808/bundesanzeiger-scraper (ONNX captcha model, MIT)
    - The ONNX model file is stored in operator1/clients/assets/

No extra API keys required. No geo-blocking.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

_BA_BASE = "https://www.bundesanzeiger.de"
_CACHE_DIR = Path("cache/de_bundesanzeiger")

# ONNX model path (trained on Bundesanzeiger captchas)
_MODEL_PATH = Path(__file__).parent / "assets" / "bundesanzeiger_captcha.onnx"

# Captcha alphabet (a-z + 0-9 = 36 classes, 6 characters per captcha)
_ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789")

# Consent cookie required by Bundesanzeiger
_CONSENT_COOKIE = "1628606977-805e172265bfdbde-10"

# Max captcha solve attempts per document
_MAX_CAPTCHA_ATTEMPTS = 3

# Rate limiting between requests
_REQUEST_DELAY_S = 0.3

# German -> canonical field name mapping for Bilanz (Balance Sheet)
_BILANZ_MAP: dict[str, str] = {
    # Assets
    "immaterielle vermögensgegenstände": "intangible_assets",
    "sachanlagen": "property_plant_equipment",
    "finanzanlagen": "financial_assets",
    "anlagevermögen": "noncurrent_assets",
    "vorräte": "inventory",
    "forderungen": "receivables",
    "forderungen aus lieferungen und leistungen": "receivables",
    "kassenbestand": "cash_and_equivalents",
    "kassenbestand, bundesbankguthaben": "cash_and_equivalents",
    "flüssige mittel": "cash_and_equivalents",
    "zahlungsmittel und zahlungsmitteläquivalente": "cash_and_equivalents",
    "umlaufvermögen": "current_assets",
    "bilanzsumme": "total_assets",
    "summe aktiva": "total_assets",
    # Liabilities
    "eigenkapital": "total_equity",
    "gezeichnetes kapital": "share_capital",
    "gewinnrücklagen": "retained_earnings",
    "bilanzgewinn": "retained_earnings",
    "rückstellungen": "provisions",
    "verbindlichkeiten": "total_liabilities",
    "verbindlichkeiten gegenüber kreditinstituten": "long_term_debt",
    "kurzfristige verbindlichkeiten": "current_liabilities",
    "langfristige verbindlichkeiten": "noncurrent_liabilities",
    "summe passiva": "total_liabilities_and_equity",
    "bilanzsumme passiva": "total_liabilities_and_equity",
    # IFRS-style German terms
    "kurzfristige vermögenswerte": "current_assets",
    "langfristige vermögenswerte": "noncurrent_assets",
    "summe vermögenswerte": "total_assets",
    "kurzfristige schulden": "current_liabilities",
    "langfristige schulden": "noncurrent_liabilities",
    "summe schulden": "total_liabilities",
    "geschäfts- oder firmenwert": "goodwill",
}

# German -> canonical for GuV (Income Statement / P&L)
_GUV_MAP: dict[str, str] = {
    "umsatzerlöse": "revenue",
    "umsatz": "revenue",
    "erlöse": "revenue",
    "herstellungskosten": "cost_of_revenue",
    "materialaufwand": "cost_of_revenue",
    "bruttoergebnis": "gross_profit",
    "rohergebnis": "gross_profit",
    "personalaufwand": "sga_expenses",
    "abschreibungen": "depreciation_amortization",
    "sonstige betriebliche aufwendungen": "other_operating_expenses",
    "sonstige betriebliche erträge": "other_operating_income",
    "betriebsergebnis": "operating_income",
    "ergebnis der gewöhnlichen geschäftstätigkeit": "ebit",
    "ergebnis vor ertragsteuern": "ebit",
    "zinsen und ähnliche aufwendungen": "interest_expense",
    "zinsaufwand": "interest_expense",
    "zinsen und ähnliche erträge": "interest_income",
    "steuern vom einkommen und vom ertrag": "taxes",
    "ertragsteuern": "taxes",
    "jahresüberschuss": "net_income",
    "jahresfehlbetrag": "net_income",
    "konzernergebnis": "net_income",
    "ergebnis nach steuern": "net_income",
    "ergebnis je aktie": "eps",
    "unverwässertes ergebnis je aktie": "eps",
    "verwässertes ergebnis je aktie": "eps_diluted",
    # IFRS-style German terms
    "umsatzkosten": "cost_of_revenue",
    "forschungs- und entwicklungskosten": "rd_expenses",
    "vertriebskosten": "sga_expenses",
    "allgemeine verwaltungskosten": "admin_expenses",
}

# German -> canonical for Kapitalflussrechnung (Cash Flow Statement)
_CASHFLOW_MAP: dict[str, str] = {
    "cashflow aus betrieblicher tätigkeit": "operating_cash_flow",
    "cashflow aus der laufenden geschäftstätigkeit": "operating_cash_flow",
    "mittelzufluss aus laufender geschäftstätigkeit": "operating_cash_flow",
    "cashflow aus investitionstätigkeit": "investing_cf",
    "cashflow aus der investitionstätigkeit": "investing_cf",
    "mittelabfluss aus investitionstätigkeit": "investing_cf",
    "cashflow aus finanzierungstätigkeit": "financing_cf",
    "cashflow aus der finanzierungstätigkeit": "financing_cf",
    "mittelzu-/abfluss aus finanzierungstätigkeit": "financing_cf",
    "investitionen in sachanlagen": "capex",
    "auszahlungen für investitionen": "capex",
    "dividendenzahlungen": "dividends_paid",
    "gezahlte dividenden": "dividends_paid",
    # IFRS-style
    "zahlungsmittel aus betrieblicher tätigkeit": "operating_cash_flow",
    "zahlungsmittel aus investitionstätigkeit": "investing_cf",
    "zahlungsmittel aus finanzierungstätigkeit": "financing_cf",
}

# Statement type -> field map
_STATEMENT_MAPS: dict[str, dict[str, str]] = {
    "income": _GUV_MAP,
    "balance": _BILANZ_MAP,
    "cashflow": _CASHFLOW_MAP,
}

# German section keywords for detecting statement type in tables
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "balance": ["bilanz", "aktiva", "passiva", "vermögenswerte", "finanzlage",
                "financial position", "balance sheet"],
    "income": ["gewinn", "verlust", "ergebnis", "guv", "gesamtergebnis",
               "income statement", "profit", "loss"],
    "cashflow": ["kapitalfluss", "cashflow", "cash flow", "zahlungsmittel",
                 "mittelverwendung"],
}


# ---------------------------------------------------------------------------
# ONNX captcha solver
# ---------------------------------------------------------------------------

_onnx_model = None


def _load_captcha_model():
    """Load the ONNX captcha solver model (lazy, cached)."""
    global _onnx_model
    if _onnx_model is not None:
        return _onnx_model

    if not _MODEL_PATH.exists():
        logger.warning(
            "Bundesanzeiger ONNX captcha model not found at %s. "
            "Download from: github.com/dre808/bundesanzeiger-scraper/raw/main/assets/model.onnx",
            _MODEL_PATH,
        )
        return None

    try:
        from onnxruntime import InferenceSession
        _onnx_model = InferenceSession(str(_MODEL_PATH))
        logger.info("Bundesanzeiger captcha model loaded: %s", _MODEL_PATH)
        return _onnx_model
    except ImportError:
        logger.warning("onnxruntime not installed -- captcha solving unavailable")
        return None
    except Exception as exc:
        logger.warning("Failed to load captcha model: %s", exc)
        return None


def _solve_captcha(image_data: bytes) -> str:
    """Solve a Bundesanzeiger captcha image using the ONNX model.

    Parameters
    ----------
    image_data : bytes
        Raw JPEG/PNG bytes of the 250x50 captcha image.

    Returns
    -------
    str
        6-character uppercase solution, or empty string on failure.
    """
    model = _load_captcha_model()
    if model is None:
        return ""

    try:
        img = Image.open(BytesIO(image_data)).convert("L")
        arr = np.array(img, dtype=np.float32)
        arr = arr / 255.0 * 2.0 - 1.0  # normalize to [-1, 1]
        arr = arr.reshape((1, 50, 250, 1))

        prediction = model.run(None, {"captcha": arr})[0][0]
        char_indexes = np.argmax(prediction, axis=1)
        chars = np.array(_ALPHABET)[char_indexes]
        return "".join(chars).upper()
    except Exception as exc:
        logger.debug("Captcha solve failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _create_session() -> "requests.Session":
    """Create a requests session configured for Bundesanzeiger."""
    import requests as _requests

    sess = _requests.Session()
    sess.cookies["cc"] = _CONSENT_COOKIE
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.bundesanzeiger.de/",
        "DNT": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    })

    # Initialize session (required for Wicket state)
    try:
        sess.get(f"{_BA_BASE}", timeout=15)
        sess.get(f"{_BA_BASE}/pub/de/start?0", timeout=15)
    except Exception as exc:
        logger.debug("Session init failed: %s", exc)

    return sess


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class BundesanzeigerFiling:
    """A single filing from Bundesanzeiger search results."""
    company_name: str = ""
    title: str = ""
    publication_date: str = ""  # DD.MM.YYYY
    document_url: str = ""
    fiscal_year_start: str = ""  # from title parsing
    fiscal_year_end: str = ""  # from title parsing
    filing_type: str = ""  # jahresabschluss, konzernabschluss


def _parse_fiscal_dates(title: str) -> tuple[str, str]:
    """Extract fiscal year start/end from filing title.

    Titles look like:
        "Jahresabschluss zum Geschäftsjahr vom 01.10.2021 bis zum 30.09.2022"
    """
    match = re.search(
        r"vom\s+(\d{2}\.\d{2}\.\d{4})\s+bis\s+(?:zum\s+)?(\d{2}\.\d{2}\.\d{4})",
        title,
    )
    if match:
        start_de = match.group(1)  # DD.MM.YYYY
        end_de = match.group(2)
        # Convert to ISO
        start = f"{start_de[6:]}-{start_de[3:5]}-{start_de[:2]}"
        end = f"{end_de[6:]}-{end_de[3:5]}-{end_de[:2]}"
        return start, end
    return "", ""


def _classify_filing(title: str) -> str:
    """Classify filing type from title."""
    lower = title.lower()
    if "konzernabschluss" in lower:
        return "konzernabschluss"
    if "jahresabschluss" in lower:
        return "jahresabschluss"
    if "halbjahres" in lower:
        return "halbjahresbericht"
    return "other"


def search_bundesanzeiger(
    company_name: str,
    session: "requests.Session | None" = None,
) -> list[BundesanzeigerFiling]:
    """Search Bundesanzeiger for financial filings of a company.

    Parameters
    ----------
    company_name : str
        Company name to search for (e.g. "Siemens Aktiengesellschaft").
    session : requests.Session, optional
        Pre-configured session. Created if not provided.

    Returns
    -------
    List of BundesanzeigerFiling objects sorted by date (newest first).
    """
    if session is None:
        session = _create_session()

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 not installed -- Bundesanzeiger search unavailable")
        return []

    search_term = f"{company_name} Jahresabschluss"
    url = (
        f"{_BA_BASE}/pub/de/start?0-2."
        f"-top%7Econtent%7Epanel-left%7Ecard-form="
        f"&fulltext={search_term}"
        f"&area_select=22"
        f"&search_button=Suchen"
    )

    try:
        resp = session.get(url, timeout=30)
    except Exception as exc:
        logger.warning("Bundesanzeiger search failed: %s", exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    wrapper = soup.find("div", {"class": "result_container"})
    if not wrapper:
        return []

    filings: list[BundesanzeigerFiling] = []
    for row in wrapper.find_all("div", {"class": "row"}):
        info_el = row.find("div", {"class": "info"})
        if not info_el:
            continue
        link_el = info_el.find("a")
        if not link_el:
            continue

        title = link_el.text.strip()
        href = link_el.get("href", "")

        date_el = row.find("div", {"class": "date"})
        pub_date = date_el.text.strip() if date_el else ""

        company_el = row.find("div", {"class": "first"})
        name = company_el.text.strip().replace("\n", " ") if company_el else ""

        fy_start, fy_end = _parse_fiscal_dates(title)
        ftype = _classify_filing(title)

        if ftype in ("jahresabschluss", "konzernabschluss", "halbjahresbericht"):
            filings.append(BundesanzeigerFiling(
                company_name=name,
                title=title,
                publication_date=pub_date,
                document_url=href,
                fiscal_year_start=fy_start,
                fiscal_year_end=fy_end,
                filing_type=ftype,
            ))

    logger.info(
        "Bundesanzeiger search '%s': %d financial filings found",
        company_name, len(filings),
    )
    return filings


# ---------------------------------------------------------------------------
# Document fetching (with captcha solving)
# ---------------------------------------------------------------------------

def fetch_document(
    filing: BundesanzeigerFiling,
    session: "requests.Session | None" = None,
) -> str:
    """Fetch the full HTML content of a Bundesanzeiger filing.

    Handles the CAPTCHA challenge automatically using the ONNX model.
    Retries up to _MAX_CAPTCHA_ATTEMPTS times.

    Parameters
    ----------
    filing : BundesanzeigerFiling
        Filing metadata from search results.
    session : requests.Session, optional
        Pre-configured session.

    Returns
    -------
    str
        HTML content of the publication_container, or empty string on failure.
    """
    if session is None:
        session = _create_session()

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    for attempt in range(1, _MAX_CAPTCHA_ATTEMPTS + 1):
        try:
            resp = session.get(filing.document_url, timeout=30)
            soup = BeautifulSoup(resp.text, "html.parser")

            pub = soup.find("div", {"class": "publication_container"})
            if pub:
                return str(pub)

            # Need to solve captcha
            captcha_div = soup.find("div", {"class": "captcha_wrapper"})
            if not captcha_div:
                logger.debug("No publication_container and no captcha -- unexpected page")
                return ""

            img_el = captcha_div.find("img")
            if not img_el:
                return ""

            img_resp = session.get(img_el["src"], timeout=15)
            solution = _solve_captcha(img_resp.content)
            if not solution:
                logger.debug("Captcha solve returned empty (attempt %d)", attempt)
                continue

            forms = soup.find_all("form")
            form_action = forms[1].get("action", "") if len(forms) > 1 else ""
            if not form_action:
                return ""

            solved_resp = session.post(
                form_action,
                data={"solution": solution, "confirm-button": "OK"},
                timeout=30,
            )

            solved_soup = BeautifulSoup(solved_resp.text, "html.parser")
            pub = solved_soup.find("div", {"class": "publication_container"})
            if pub:
                logger.info(
                    "Bundesanzeiger captcha solved (attempt %d): %s",
                    attempt, filing.title[:60],
                )
                return str(pub)

            logger.debug(
                "Captcha solution '%s' rejected (attempt %d/%d)",
                solution, attempt, _MAX_CAPTCHA_ATTEMPTS,
            )
            time.sleep(_REQUEST_DELAY_S)

        except Exception as exc:
            logger.debug("Document fetch failed (attempt %d): %s", attempt, exc)

    logger.warning(
        "Failed to solve captcha after %d attempts for: %s",
        _MAX_CAPTCHA_ATTEMPTS, filing.title[:60],
    )
    return ""


# ---------------------------------------------------------------------------
# Financial data extraction from HTML tables
# ---------------------------------------------------------------------------

def _parse_german_number(text: str) -> float | None:
    """Parse a German-formatted number (e.g. '1.234.567,89' or '-123,45').

    German number format uses:
    - '.' as thousands separator
    - ',' as decimal separator
    - '-' or brackets for negative values
    """
    if not text:
        return None

    cleaned = text.strip()
    if not cleaned:
        return None

    # Handle parentheses for negative numbers
    is_negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        is_negative = True
        cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith("-"):
        is_negative = True
        cleaned = cleaned[1:].strip()

    # Remove thousands separators (dots in German)
    cleaned = cleaned.replace(".", "")
    # Replace decimal comma with dot
    cleaned = cleaned.replace(",", ".")
    # Remove any remaining non-numeric chars (like EUR, TEUR, etc.)
    cleaned = re.sub(r"[^\d.]", "", cleaned)

    if not cleaned:
        return None

    try:
        value = float(cleaned)
        return -value if is_negative else value
    except ValueError:
        return None


def _detect_unit_multiplier(html_text: str) -> float:
    """Detect if values are in thousands (TEUR/Tsd.) or millions (Mio.)."""
    lower = html_text.lower()
    if "mio" in lower or "millionen" in lower:
        return 1_000_000
    if "teur" in lower or "tsd" in lower or "in tausend" in lower:
        return 1_000
    return 1.0


def _detect_statement_type(table_text: str) -> str:
    """Detect which financial statement a table represents."""
    lower = table_text.lower()
    for stmt_type, keywords in _SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return stmt_type
    return ""


def _fuzzy_match_field(
    label: str,
    field_map: dict[str, str],
    threshold: float = 0.7,
) -> str:
    """Fuzzy match a German label to a canonical field name."""
    label_lower = label.lower().strip()

    # Exact match
    if label_lower in field_map:
        return field_map[label_lower]

    # Substring match
    for key, canonical in field_map.items():
        if key in label_lower or label_lower in key:
            return canonical

    # Fuzzy match (simple ratio)
    try:
        from difflib import SequenceMatcher
        best_score = 0.0
        best_match = ""
        for key, canonical in field_map.items():
            ratio = SequenceMatcher(None, label_lower, key).ratio()
            if ratio > best_score and ratio >= threshold:
                best_score = ratio
                best_match = canonical
        return best_match
    except Exception:
        return ""


def extract_financials_from_html(
    html_content: str,
    statement_type: str = "",
    filing_date: str = "",
    report_date: str = "",
) -> pd.DataFrame:
    """Extract structured financial data from a Bundesanzeiger HTML document.

    Parameters
    ----------
    html_content : str
        HTML content of the publication_container div.
    statement_type : str
        Filter to specific statement: "income", "balance", "cashflow",
        or "" for all.
    filing_date : str
        Publication date (ISO format) for PIT tracking.
    report_date : str
        Fiscal period end date (ISO format).

    Returns
    -------
    pd.DataFrame
        Canonical long-format DataFrame with columns:
        canonical_name, value, report_date, filing_date.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return pd.DataFrame()

    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return pd.DataFrame()

    multiplier = _detect_unit_multiplier(html_content)
    records: list[dict] = []

    for table in tables:
        table_text = table.get_text()
        detected_type = _detect_statement_type(table_text)

        if statement_type and detected_type and detected_type != statement_type:
            continue

        # Determine which field map to use
        if detected_type == "balance":
            field_map = _BILANZ_MAP
        elif detected_type == "income":
            field_map = _GUV_MAP
        elif detected_type == "cashflow":
            field_map = _CASHFLOW_MAP
        else:
            # Use combined map
            field_map = {**_BILANZ_MAP, **_GUV_MAP, **_CASHFLOW_MAP}

        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            # First cell is typically the label, second+ are values
            label = cells[0].get_text(strip=True)
            if not label or len(label) < 3:
                continue

            canonical = _fuzzy_match_field(label, field_map)
            if not canonical:
                continue

            # Try to parse the first numeric cell after the label
            for cell in cells[1:]:
                value = _parse_german_number(cell.get_text(strip=True))
                if value is not None:
                    records.append({
                        "canonical_name": canonical,
                        "value": value * multiplier,
                        "report_date": report_date,
                        "filing_date": filing_date,
                    })
                    break

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Deduplicate: keep first occurrence per canonical_name
    df = df.drop_duplicates(subset=["canonical_name"], keep="first")

    logger.info(
        "Bundesanzeiger extracted %d financial fields from HTML tables",
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# High-level API: search + fetch + extract
# ---------------------------------------------------------------------------

def fetch_german_financials(
    company_name: str,
    statement_type: str = "",
    max_filings: int = 3,
) -> pd.DataFrame:
    """Search, fetch, and extract financial data for a German company.

    This is the main entry point. Combines:
    1. Search Bundesanzeiger for Jahresabschluss/Konzernabschluss filings
    2. Solve captcha and download document HTML
    3. Extract structured financial data from HTML tables
    4. Map German field names to canonical English schema

    Parameters
    ----------
    company_name : str
        Company name (e.g. "Siemens Aktiengesellschaft", "BMW AG").
    statement_type : str
        Filter: "income", "balance", "cashflow", or "" for all.
    max_filings : int
        Maximum number of filings to extract (newest first).

    Returns
    -------
    pd.DataFrame
        Canonical long-format DataFrame with columns:
        canonical_name, value, report_date, filing_date.
    """
    session = _create_session()
    filings = search_bundesanzeiger(company_name, session)

    if not filings:
        logger.info("No Bundesanzeiger filings found for '%s'", company_name)
        return pd.DataFrame()

    # Filter to exact company name match (avoid Siemens Pensionsfonds etc.)
    target_lower = company_name.lower().strip()
    exact = [
        f for f in filings
        if target_lower in f.company_name.lower()
    ]
    if exact:
        filings = exact

    # Prefer Konzernabschluss (consolidated) over Jahresabschluss (standalone)
    konzern = [f for f in filings if f.filing_type == "konzernabschluss"]
    if konzern:
        filings = konzern + [f for f in filings if f.filing_type != "konzernabschluss"]

    all_records: list[pd.DataFrame] = []

    for filing in filings[:max_filings]:
        html = fetch_document(filing, session)
        if not html:
            continue

        # Convert German date (DD.MM.YYYY) to ISO
        pub_date_iso = ""
        if filing.publication_date:
            parts = filing.publication_date.split(".")
            if len(parts) == 3:
                pub_date_iso = f"{parts[2]}-{parts[1]}-{parts[0]}"

        df = extract_financials_from_html(
            html,
            statement_type=statement_type,
            filing_date=pub_date_iso,
            report_date=filing.fiscal_year_end,
        )

        if not df.empty:
            all_records.append(df)

        time.sleep(_REQUEST_DELAY_S)

    if not all_records:
        return pd.DataFrame()

    result = pd.concat(all_records, ignore_index=True)
    result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")
    result["filing_date"] = pd.to_datetime(result["filing_date"], errors="coerce")

    # Deduplicate: keep latest filing per (canonical_name, report_date)
    result = result.sort_values("filing_date", ascending=False)
    result = result.drop_duplicates(
        subset=["canonical_name", "report_date"], keep="first",
    )

    logger.info(
        "Bundesanzeiger total: %d records across %d periods for '%s'",
        len(result),
        result["report_date"].nunique() if "report_date" in result.columns else 0,
        company_name,
    )
    return result
