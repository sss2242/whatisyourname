"""Universal LLM-aided financial filing extractor.

Extracts structured financial data from PDF, HTML, and iXBRL filings
using the existing Gemini/Claude LLM clients. Works across all markets.

The extractor handles three input formats:
1. **iXBRL**: Machine-parseable tagged data (best quality, no LLM needed)
2. **HTML**: Semi-structured tables (light LLM validation)
3. **PDF**: Unstructured text (full LLM extraction)

The extracted data is validated using accounting identities before
being returned in canonical long format.

Usage:
    extractor = LLMFilingExtractor(llm_client)
    result = extractor.extract_from_pdf(pdf_bytes, market_id="au_asx")
    result = extractor.extract_from_html(html_text, market_id="hk_hkex")
    result = extractor.extract_from_ixbrl(ixbrl_text, market_id="ca_sedar")
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical financial fields the extractor should look for.
# These match STATEMENT_FIELDS in cache_builder.py.
EXTRACTION_FIELDS = [
    "revenue", "cost_of_revenue", "gross_profit",
    "operating_income", "ebit", "ebitda", "net_income",
    "interest_expense", "taxes",
    "total_assets", "total_liabilities", "total_equity",
    "current_assets", "current_liabilities",
    "cash_and_equivalents", "short_term_debt", "long_term_debt",
    "total_debt", "retained_earnings",
    "operating_cash_flow", "capex", "free_cash_flow",
    "investing_cf", "financing_cf", "dividends_paid",
    "stock_buybacks",
    "sga_expenses", "rd_expenses",
    "eps", "eps_diluted",
]

# The LLM prompt -- designed to extract ALL fields in a single call.
_EXTRACTION_PROMPT = """You are a financial data extraction expert. Extract the following financial data from this filing document.

Return ONLY a valid JSON object with these exact keys (use null if a value is not found):

INCOME STATEMENT:
- revenue: Total revenue/sales (number)
- cost_of_revenue: Cost of goods sold (number)
- gross_profit: Gross profit (number)
- operating_income: Operating income/profit (number)
- ebit: Earnings before interest and taxes (number)
- ebitda: EBITDA (number)
- net_income: Net income/profit (number)
- interest_expense: Interest expense (number)
- taxes: Income tax expense (number)
- sga_expenses: Selling, general and administrative expenses (number)
- rd_expenses: Research and development expenses (number)
- eps: Earnings per share basic (number)
- eps_diluted: Earnings per share diluted (number)

BALANCE SHEET:
- total_assets: Total assets (number)
- total_liabilities: Total liabilities (number)
- total_equity: Total shareholders equity (number)
- current_assets: Total current assets (number)
- current_liabilities: Total current liabilities (number)
- cash_and_equivalents: Cash and cash equivalents (number)
- short_term_debt: Short-term borrowings/debt (number)
- long_term_debt: Long-term debt (number)
- total_debt: Total debt (number)
- retained_earnings: Retained earnings (number)

CASH FLOW STATEMENT:
- operating_cash_flow: Net cash from operating activities (number)
- capex: Capital expenditures (number, usually negative)
- free_cash_flow: Free cash flow (number)
- investing_cf: Net cash from investing activities (number)
- financing_cf: Net cash from financing activities (number)
- dividends_paid: Dividends paid (number, usually negative)
- stock_buybacks: Share repurchases (number, usually negative)

METADATA:
- report_date: The period end date in YYYY-MM-DD format
- filing_date: The date the filing was published in YYYY-MM-DD format
- currency: 3-letter currency code (e.g. CAD, AUD, HKD)
- period_type: One of "annual", "quarterly", "semiannual"

All monetary values should be in the filing's native currency and unit.
If amounts are in thousands, multiply by 1000. If in millions, multiply by 1,000,000.
Return raw numbers, not formatted strings.

FILING TEXT:
{text}
"""

# Shorter prompt for validation/cross-check of existing data
_VALIDATION_PROMPT = """Verify this financial data extracted from a {market} filing.
Check for obvious errors (e.g., total_assets != total_liabilities + total_equity).
Return the corrected JSON if errors found, or the same JSON if correct.
Add a "validation_notes" field with any issues found.

Data to validate:
{data}

Filing text excerpt:
{text}
"""


@dataclass
class ExtractionResult:
    """Result of financial data extraction."""

    success: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    source_format: str = ""  # "ixbrl", "html", "pdf"
    validation_passed: bool = False
    validation_notes: list[str] = field(default_factory=list)
    error: str = ""
    tokens_used: int = 0


class LLMFilingExtractor:
    """Universal financial filing extractor using LLM."""

    def __init__(self, llm_client=None):
        """Initialize with an LLM client (Gemini or Claude).

        Parameters
        ----------
        llm_client:
            An LLM client from llm_factory (GeminiClient or ClaudeClient).
            If None, extraction from unstructured formats will fail gracefully.
        """
        self._llm = llm_client

    # ------------------------------------------------------------------
    # iXBRL extraction (structured, no LLM needed)
    # ------------------------------------------------------------------

    def extract_from_ixbrl(
        self,
        ixbrl_text: str,
        market_id: str = "",
    ) -> ExtractionResult:
        """Extract financials from iXBRL-tagged HTML.

        Uses the ixbrl-parse library (already installed) for structured
        extraction. No LLM call needed.
        """
        result = ExtractionResult(source_format="ixbrl")

        try:
            from ixbrl_parse import IXBRL
        except ImportError:
            result.error = "ixbrl-parse not installed"
            return result

        try:
            doc = IXBRL(io.StringIO(ixbrl_text))
            facts = {}
            for fact in doc.numeric:
                name = fact.get("name", "")
                value = fact.get("value")
                context = fact.get("context", {})
                if value is not None:
                    # Store by concept name
                    facts[name] = {
                        "value": float(value),
                        "context": context,
                    }

            if not facts:
                result.error = "No numeric facts found in iXBRL"
                return result

            # Map iXBRL concept names to canonical fields
            data = self._map_ixbrl_concepts(facts, market_id)
            result.data = data
            result.success = bool(data)
            result.validation_passed = self._validate_accounting_identities(data)
            return result

        except Exception as exc:
            result.error = f"iXBRL parsing failed: {exc}"
            return result

    # ------------------------------------------------------------------
    # HTML extraction (semi-structured)
    # ------------------------------------------------------------------

    def extract_from_html(
        self,
        html_text: str,
        market_id: str = "",
    ) -> ExtractionResult:
        """Extract financials from HTML filing (with tables)."""
        result = ExtractionResult(source_format="html")

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            result.error = "beautifulsoup4 not installed"
            return result

        try:
            soup = BeautifulSoup(html_text, "html.parser")

            # Extract all tables
            tables = soup.find_all("table")
            table_texts = []
            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    row_text = " | ".join(c.get_text(strip=True) for c in cells)
                    if row_text.strip():
                        table_texts.append(row_text)

            if not table_texts:
                # Fall back to full text
                text = soup.get_text(separator="\n", strip=True)
            else:
                text = "\n".join(table_texts)

            # Truncate to ~12K chars for LLM context
            text = text[:12000]

            return self._extract_via_llm(text, market_id, "html")

        except Exception as exc:
            result.error = f"HTML extraction failed: {exc}"
            return result

    # ------------------------------------------------------------------
    # PDF extraction (unstructured)
    # ------------------------------------------------------------------

    def extract_from_pdf(
        self,
        pdf_bytes: bytes,
        market_id: str = "",
    ) -> ExtractionResult:
        """Extract financials from a PDF filing."""
        result = ExtractionResult(source_format="pdf")

        try:
            import pdfplumber
        except ImportError:
            result.error = "pdfplumber not installed (pip install pdfplumber)"
            return result

        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                # Extract text from all pages, focusing on financial tables
                all_text = []
                for page in pdf.pages:
                    # Try table extraction first
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            for row in table:
                                if row:
                                    row_text = " | ".join(
                                        str(cell).strip() for cell in row if cell
                                    )
                                    if row_text:
                                        all_text.append(row_text)
                    else:
                        # Fall back to raw text
                        text = page.extract_text()
                        if text:
                            all_text.append(text)

                text = "\n".join(all_text)

            if not text.strip():
                result.error = "No text extracted from PDF"
                return result

            # Truncate to ~12K chars for LLM context
            text = text[:12000]

            return self._extract_via_llm(text, market_id, "pdf")

        except Exception as exc:
            result.error = f"PDF extraction failed: {exc}"
            return result

    # ------------------------------------------------------------------
    # LLM extraction core
    # ------------------------------------------------------------------

    def _extract_via_llm(
        self,
        text: str,
        market_id: str,
        source_format: str,
    ) -> ExtractionResult:
        """Send text to LLM for structured extraction."""
        result = ExtractionResult(source_format=source_format)

        if self._llm is None:
            result.error = "No LLM client available for extraction"
            return result

        prompt = _EXTRACTION_PROMPT.format(text=text)

        try:
            # Use the LLM's generate method
            response = self._llm.generate(prompt)
            if not response:
                result.error = "LLM returned empty response"
                return result

            # Parse JSON from response
            data = self._parse_llm_json(response)
            if not data:
                result.error = "Could not parse JSON from LLM response"
                return result

            # Clean and validate
            cleaned = self._clean_extracted_data(data)
            result.data = cleaned
            result.success = bool(cleaned)
            result.validation_passed = self._validate_accounting_identities(cleaned)

            if not result.validation_passed:
                result.validation_notes.append(
                    "Accounting identity check failed (assets != liabilities + equity)"
                )

            return result

        except Exception as exc:
            result.error = f"LLM extraction failed: {exc}"
            return result

    def validate_existing_data(
        self,
        data: dict[str, Any],
        filing_text: str = "",
        market_id: str = "",
    ) -> ExtractionResult:
        """Validate/cross-check existing extracted data using LLM.

        Can be used on data from any source (yfinance, EDGAR, etc.)
        to verify against the original filing text.
        """
        result = ExtractionResult(source_format="validation")

        if self._llm is None:
            # No LLM -- just do accounting identity check
            result.validation_passed = self._validate_accounting_identities(data)
            result.data = data
            result.success = True
            return result

        if filing_text:
            prompt = _VALIDATION_PROMPT.format(
                market=market_id,
                data=json.dumps(data, indent=2),
                text=filing_text[:8000],
            )
            try:
                response = self._llm.generate(prompt)
                validated = self._parse_llm_json(response)
                if validated:
                    notes = validated.pop("validation_notes", [])
                    if isinstance(notes, str):
                        notes = [notes]
                    result.validation_notes = notes
                    result.data = self._clean_extracted_data(validated)
                    result.success = True
                    result.validation_passed = not bool(notes) or all(
                        "correct" in n.lower() or "no issues" in n.lower()
                        for n in notes
                    )
                    return result
            except Exception as exc:
                logger.debug("LLM validation failed: %s", exc)

        # Fallback: accounting identity check only
        result.validation_passed = self._validate_accounting_identities(data)
        result.data = data
        result.success = True
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_llm_json(self, response: str) -> dict | None:
        """Extract JSON from LLM response (handles markdown fences)."""
        if not response:
            return None

        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fence
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _clean_extracted_data(self, data: dict) -> dict:
        """Clean and normalize extracted financial data."""
        cleaned = {}
        for key in EXTRACTION_FIELDS:
            val = data.get(key)
            if val is not None:
                try:
                    cleaned[key] = float(val)
                except (TypeError, ValueError):
                    pass

        # Metadata
        for meta_key in ("report_date", "filing_date", "currency", "period_type"):
            if meta_key in data and data[meta_key]:
                cleaned[meta_key] = str(data[meta_key])

        return cleaned

    def _validate_accounting_identities(self, data: dict) -> bool:
        """Check basic accounting identities."""
        issues = []

        # Balance sheet: assets = liabilities + equity
        assets = data.get("total_assets")
        liabilities = data.get("total_liabilities")
        equity = data.get("total_equity")

        if all(v is not None for v in [assets, liabilities, equity]):
            expected = liabilities + equity
            if assets > 0 and abs(assets - expected) / assets > 0.05:
                issues.append(
                    f"total_assets ({assets}) != total_liabilities ({liabilities}) + "
                    f"total_equity ({equity}) = {expected}"
                )

        # Cash flow: FCF ~= operating_cf - capex
        ocf = data.get("operating_cash_flow")
        capex = data.get("capex")
        fcf = data.get("free_cash_flow")

        if all(v is not None for v in [ocf, capex, fcf]):
            expected_fcf = ocf + capex  # capex is typically negative
            if abs(fcf) > 0 and abs(fcf - expected_fcf) / abs(fcf) > 0.1:
                issues.append(
                    f"free_cash_flow ({fcf}) != operating_cf ({ocf}) + capex ({capex}) = {expected_fcf}"
                )

        if issues:
            logger.debug("Accounting identity issues: %s", issues)

        return len(issues) == 0

    def _map_ixbrl_concepts(
        self,
        facts: dict[str, dict],
        market_id: str,
    ) -> dict:
        """Map iXBRL concept names to canonical financial fields."""
        # IFRS concept name patterns -> canonical names
        ifrs_map = {
            "Revenue": "revenue",
            "CostOfSales": "cost_of_revenue",
            "GrossProfit": "gross_profit",
            "ProfitLossFromOperatingActivities": "operating_income",
            "ProfitLossBeforeTax": "ebit",
            "ProfitLoss": "net_income",
            "IncomeTaxExpenseContinuingOperations": "taxes",
            "FinanceCosts": "interest_expense",
            "Assets": "total_assets",
            "Liabilities": "total_liabilities",
            "Equity": "total_equity",
            "CurrentAssets": "current_assets",
            "CurrentLiabilities": "current_liabilities",
            "CashAndCashEquivalents": "cash_and_equivalents",
            "NoncurrentBorrowings": "long_term_debt",
            "CurrentBorrowings": "short_term_debt",
            "RetainedEarnings": "retained_earnings",
            "CashFlowsFromUsedInOperatingActivities": "operating_cash_flow",
            "PurchaseOfPropertyPlantAndEquipment": "capex",
            "CashFlowsFromUsedInInvestingActivities": "investing_cf",
            "CashFlowsFromUsedInFinancingActivities": "financing_cf",
            "DividendsPaid": "dividends_paid",
            "EarningsPerShare": "eps",
            "DilutedEarningsPerShare": "eps_diluted",
        }

        result = {}
        for fact_name, fact_data in facts.items():
            # Try exact match first, then partial match
            canonical = None
            for pattern, canonical_name in ifrs_map.items():
                if pattern.lower() in fact_name.lower():
                    canonical = canonical_name
                    break

            if canonical and canonical not in result:
                result[canonical] = fact_data["value"]

        return result

    # ------------------------------------------------------------------
    # Convenience: extract to canonical long-format DataFrame
    # ------------------------------------------------------------------

    def to_canonical_dataframe(
        self,
        extraction: ExtractionResult,
        market_id: str = "",
    ) -> pd.DataFrame:
        """Convert extraction result to canonical long-format DataFrame.

        Returns DataFrame with columns: canonical_name, value, report_date, filing_date.
        """
        if not extraction.success or not extraction.data:
            return pd.DataFrame()

        data = extraction.data
        report_date = data.get("report_date", "")
        filing_date = data.get("filing_date", report_date)
        currency = data.get("currency", "")

        records = []
        for field_name in EXTRACTION_FIELDS:
            value = data.get(field_name)
            if value is not None:
                records.append({
                    "canonical_name": field_name,
                    "value": float(value),
                    "report_date": pd.Timestamp(report_date) if report_date else pd.NaT,
                    "filing_date": pd.Timestamp(filing_date) if filing_date else pd.NaT,
                    "currency": currency,
                    "source": f"llm_extract_{extraction.source_format}",
                    "market_id": market_id,
                })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)
