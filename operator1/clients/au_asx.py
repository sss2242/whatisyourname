"""Australia ASX PIT client -- uses ASX MarkitDigital API + filing discovery.

Primary profile: ASX MarkitDigital API (asx.api.markitdigital.com)
Primary financials: Filing discovery + LLM extraction from ASX PDFs
OHLCV: Handled by ohlcv_provider.py (yfinance .AX suffix)

Coverage: ~2,200+ listed companies, ~$1.8T market cap.

NOTE: The old ASX API (asx.com.au/asx/1/) returns 404 as of 2026.
The MarkitDigital API is the current ASX data provider.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ASX MarkitDigital API (same API used by the ASX website and filing discoverer)
_MARKIT_BASE = "https://asx.api.markitdigital.com/asx-research/1.0"
_MARKIT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Operator1/1.0",
}

_CACHE_DIR = Path("cache/au_asx")


class AUAsxClient:
    """PIT client for Australian ASX equities.

    Uses the ASX MarkitDigital API for profile and company search.
    Uses filing discovery + LLM extraction for financial statements.
    No yfinance dependency -- all data comes from ASX-native sources.
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)

    def _cache_path(self, identifier: str, fn: str) -> Path:
        return self._cache_dir / identifier.upper() / fn

    def _read_cache(self, identifier: str, fn: str) -> dict | None:
        p = self._cache_path(identifier, fn)
        if not p.exists():
            return None
        try:
            if fn == "profile.json" and (date.today() - date.fromtimestamp(p.stat().st_mtime)).days > 7:
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, fn: str, data: dict) -> None:
        p = self._cache_path(identifier, fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str:
        return "au_asx"

    @property
    def market_name(self) -> str:
        return "Australia (ASX)"

    # -- Company discovery ---------------------------------------------------

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """Search companies via ASX MarkitDigital directory API.

        The directory endpoint returns paginated results with up to 1,841+
        companies.  When a query is provided, we fetch the full list and
        filter client-side (the API has no search parameter).
        """
        try:
            # Fetch all companies (API supports up to ~2000 per page)
            url = f"{_MARKIT_BASE}/companies/directory"
            params = {"page": 0, "itemsPerPage": 2500}
            resp = requests.get(url, headers=_MARKIT_HEADERS, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            # Structure: {"data": {"items": [...], "count": N}}
            items = payload.get("data", {}).get("items", [])
            companies = [
                {
                    "ticker": i.get("symbol", ""),
                    "name": i.get("displayName", ""),
                    "cik": i.get("symbol", ""),
                    "sector": i.get("industry", ""),
                    "exchange": "ASX",
                    "country": "AU",
                    "market_id": self.market_id,
                }
                for i in items
            ]
            if query:
                q = query.lower()
                companies = [
                    c for c in companies
                    if q in c["ticker"].lower() or q in c["name"].lower()
                ]
            return companies
        except Exception as exc:
            logger.debug("ASX company list failed: %s", exc)
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    # -- Company profile -----------------------------------------------------

    def get_profile(self, identifier: str) -> dict[str, Any]:
        """Fetch company profile from ASX MarkitDigital header API.

        Returns sector, industry, market cap, and listing date directly
        from the exchange -- no yfinance dependency.
        """
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        raw: dict[str, Any] = {
            "name": "",
            "ticker": identifier.upper(),
            "isin": "",
            "country": "AU",
            "sector": "",
            "industry": "",
            "exchange": "ASX",
            "currency": "AUD",
            "cik": identifier,
        }

        # Primary: ASX MarkitDigital header endpoint
        try:
            url = f"{_MARKIT_BASE}/companies/{identifier.upper()}/header"
            resp = requests.get(url, headers=_MARKIT_HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            raw["name"] = data.get("displayName", "")
            raw["sector"] = data.get("sector", data.get("industryGroup", ""))
            raw["industry"] = data.get("industryGroup", "")
            if data.get("marketCap"):
                raw["market_cap"] = data["marketCap"]
            if data.get("dateListed"):
                raw["date_listed"] = data["dateListed"]
            logger.info(
                "ASX profile for %s: %s (sector=%s)",
                identifier, raw["name"], raw["sector"],
            )
        except Exception as exc:
            logger.info("ASX MarkitDigital profile failed for %s: %s", identifier, exc)

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_financials(identifier, "cashflow")

    def _fetch_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Fetch financials via ASX filing discovery only (PIT-compliant).

        yfinance is NOT used for financial statements because it does not
        provide true filing dates (sets filing_date = report_date).
        """
        try:
            from operator1.clients.filing_discoverer import try_filing_extraction
            df = try_filing_extraction(
                ticker=identifier,
                market_id=self.market_id,
                statement_type=statement_type,
                llm_client=None,
            )
            if df is not None and not df.empty:
                logger.info("ASX %s %s: %d rows from filing discovery",
                           identifier, statement_type, len(df))
                return df
        except Exception as exc:
            logger.debug("ASX filing discovery failed for %s: %s", identifier, exc)
        return pd.DataFrame()

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        """ASX does not provide OHLCV data. Handled by ohlcv_provider."""
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []

    # -- Institutional holders (yfinance .AX) --------------------------------

    def _yf_ticker(self, identifier: str) -> str:
        """Convert ASX ticker to yfinance format (e.g. 'BHP' -> 'BHP.AX')."""
        code = identifier.split(".")[0].strip().upper()
        return f"{code}.AX"

    def get_holders(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch institutional + mutual fund holders via yfinance (.AX suffix).

        ASX does not expose a public holder data API.  yfinance aggregates
        institutional ownership data from Yahoo Finance for major ASX-listed
        companies.  Coverage is good for ASX 200 constituents (BHP, CBA, CSL,
        etc.) but sparse for small-caps.

        Returns list of dicts with: name, shares, percentage, value,
        holder_type, date_reported, source.
        """
        holders: list[dict[str, Any]] = []
        try:
            import yfinance as yf
            tick = yf.Ticker(self._yf_ticker(identifier))

            # Institutional holders
            inst = tick.institutional_holders
            if inst is not None and not inst.empty:
                for _, row in inst.iterrows():
                    pct = row.get("pctHeld", 0) or row.get("% Out", 0) or 0
                    if isinstance(pct, (int, float)) and 0 < pct < 1:
                        pct = pct * 100
                    holders.append({
                        "name": str(row.get("Holder", "")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                        "percentage": round(float(pct), 2),
                        "holder_type": "institutional",
                        "date_reported": str(row.get("Date Reported", "")),
                        "source": "yfinance",
                    })

            # Mutual fund holders
            mf = tick.mutualfund_holders
            if mf is not None and not mf.empty:
                for _, row in mf.iterrows():
                    pct = row.get("pctHeld", 0) or row.get("% Out", 0) or 0
                    if isinstance(pct, (int, float)) and 0 < pct < 1:
                        pct = pct * 100
                    holders.append({
                        "name": str(row.get("Holder", "")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                        "percentage": round(float(pct), 2),
                        "holder_type": "mutualfund",
                        "date_reported": str(row.get("Date Reported", "")),
                        "source": "yfinance",
                    })

            # Major holders aggregate stats
            major = tick.major_holders
            if major is not None and not major.empty:
                for idx, row in major.iterrows():
                    breakdown = (
                        str(row.get("Breakdown", idx)).lower()
                        if "Breakdown" in major.columns
                        else str(idx).lower()
                    )
                    val = (
                        row.get("Value", row.iloc[-1])
                        if "Value" in major.columns
                        else row.iloc[-1]
                    )
                    try:
                        pct = float(val) * 100 if float(val) < 1 else float(val)
                    except (ValueError, TypeError):
                        continue

                    if "insider" in breakdown:
                        holders.append({
                            "name": "Insiders / Directors",
                            "shares": 0,
                            "percentage": round(pct, 2),
                            "holder_type": "insider_aggregate",
                            "date_reported": "",
                            "source": "yfinance_major",
                        })
                    elif "institution" in breakdown and "percent" in breakdown:
                        holders.append({
                            "name": "Institutional Investors",
                            "shares": 0,
                            "percentage": round(pct, 2),
                            "holder_type": "institutional_aggregate",
                            "date_reported": "",
                            "source": "yfinance_major",
                        })

            if holders:
                logger.info(
                    "ASX holders for %s: %d from yfinance (%s)",
                    identifier, len(holders), self._yf_ticker(identifier),
                )
        except Exception as exc:
            logger.debug("yfinance holders failed for ASX %s: %s", identifier, exc)

        return holders

    def get_holder_history(self, identifier: str, years: int = 2) -> pd.DataFrame:
        """Return institutional ownership metrics as a single-row snapshot.

        Uses yfinance major_holders for aggregate ownership percentages.
        ASX does not expose historical holder data via public APIs.
        """
        try:
            import yfinance as yf
            tick = yf.Ticker(self._yf_ticker(identifier))

            major = tick.major_holders
            if major is None or major.empty:
                return pd.DataFrame()

            inst_pct = 0.0
            holder_count = 0
            for idx, row in major.iterrows():
                breakdown = (
                    str(row.get("Breakdown", idx)).lower()
                    if "Breakdown" in major.columns
                    else str(idx).lower()
                )
                val = (
                    row.get("Value", row.iloc[-1])
                    if "Value" in major.columns
                    else row.iloc[-1]
                )
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    continue

                if "institution" in breakdown and "percent" in breakdown:
                    inst_pct = fval * 100 if fval < 1 else fval
                elif "institution" in breakdown and "count" in breakdown:
                    holder_count = int(fval)

            return pd.DataFrame([{
                "date_reported": pd.Timestamp.now(),
                "inst_ownership_pct": round(inst_pct, 2),
                "inst_top5_concentration": 0.0,
                "inst_holder_count": holder_count,
            }])
        except Exception as exc:
            logger.debug("yfinance holder history failed for ASX %s: %s", identifier, exc)
            return pd.DataFrame()

    def get_insider_transactions(self, identifier: str) -> list[dict[str, Any]]:
        """Fetch insider/director transactions via yfinance (.AX suffix).

        ASX requires directors to lodge Appendix 3Y (Change of Director's
        Interest Notice) within 5 business days of any change.  yfinance
        aggregates these into an insider_transactions table for major
        ASX-listed companies.

        Returns list of dicts with: insider_name, position, date,
        transaction, shares, value, source.
        """
        transactions: list[dict[str, Any]] = []
        try:
            import yfinance as yf
            tick = yf.Ticker(self._yf_ticker(identifier))

            insiders = tick.insider_transactions
            if insiders is not None and not insiders.empty:
                for _, row in insiders.iterrows():
                    name = str(row.get("Insider", ""))
                    text = str(row.get("Text", ""))
                    shares = row.get("Shares", 0)
                    start_date = row.get("Start Date", "")

                    # Classify transaction type from Text field
                    text_lower = text.lower()
                    if "sale" in text_lower or "sold" in text_lower:
                        tx_type = "Sale"
                    elif "purchase" in text_lower or "buy" in text_lower:
                        tx_type = "Purchase"
                    elif "exercise" in text_lower or "conversion" in text_lower:
                        tx_type = "Exercise"
                    elif "grant" in text_lower or "award" in text_lower or "vesting" in text_lower:
                        tx_type = "Vesting"
                    else:
                        tx_type = text[:30] if text else "Unknown"

                    try:
                        shares_int = int(shares)
                    except (ValueError, TypeError):
                        shares_int = 0

                    transactions.append({
                        "insider_name": name,
                        "position": "Director",
                        "date": str(start_date)[:10] if start_date else "",
                        "transaction": tx_type,
                        "shares": abs(shares_int),
                        "value": 0.0,  # yfinance doesn't provide value for AU
                        "source": "yfinance",
                    })

                logger.info(
                    "ASX insider transactions for %s: %d from yfinance",
                    identifier, len(transactions),
                )
        except Exception as exc:
            logger.debug("yfinance insider transactions failed for ASX %s: %s", identifier, exc)

        return transactions
