"""Taiwan MOPS PIT client -- scraper-based wrapper with disk caching.

Uses direct MOPS form POST requests (no official API exists) to fetch
financial statements for Taiwanese listed companies, with enhanced
disk caching and ROC date conversion.

Primary: MOPS form POST scraping (https://mops.twse.com.tw)
Fallback: TWSE company info for profile data

Coverage: ~1,700+ listed companies on TWSE/TPEX, ~$1.2T market cap.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from operator1.http_utils import cached_get, cached_post, HTTPError

logger = logging.getLogger(__name__)

_MOPS_BASE = "https://mops.twse.com.tw"
_TWSE_BASE = "https://www.twse.com.tw"
_CACHE_DIR = Path("cache/tw_mops")


class TWMopsError(Exception):
    def __init__(self, endpoint: str, detail: str = "") -> None:
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"TW MOPS error on {endpoint}: {detail}")


def _roc_to_ce(roc_year: int) -> int:
    """Convert ROC year to Common Era year. ROC 113 = CE 2024."""
    return roc_year + 1911


def _ce_to_roc(ce_year: int) -> int:
    """Convert Common Era year to ROC year."""
    return ce_year - 1911


class TWMopsClient:
    """Point-in-time client for MOPS (Taiwanese equities) with disk caching.

    Implements the ``PITClient`` protocol. Uses MOPS form POST scraping
    since there is no official REST API.

    Note: MOPS uses ROC calendar (year 113 = CE 2024).
    """

    def __init__(self, cache_dir: Path | str = _CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._session_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Operator1/1.0",
            "Referer": f"{_MOPS_BASE}/mops/web/index",
        }

    def _cache_path(self, identifier: str, filename: str) -> Path:
        safe_id = identifier.replace("/", "_").replace("\\", "_").upper()
        return self._cache_dir / safe_id / filename

    def _read_cache(self, identifier: str, filename: str) -> dict | None:
        path = self._cache_path(identifier, filename)
        if not path.exists():
            return None
        try:
            age_days = (date.today() - date.fromtimestamp(path.stat().st_mtime)).days
            if filename == "profile.json" and age_days > 7:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, identifier: str, filename: str, data: dict) -> None:
        path = self._cache_path(identifier, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")

    @property
    def market_id(self) -> str:
        return "tw_mops"

    @property
    def market_name(self) -> str:
        return "Taiwan (TWSE / TPEX) -- MOPS"

    def list_companies(self, query: str = "") -> list[dict[str, Any]]:
        """List Taiwanese companies from TWSE company list."""
        try:
            data = cached_get(
                f"{_TWSE_BASE}/exchangeReport/STOCK_DAY_ALL",
                params={"response": "json"},
                headers=self._session_headers,
            )
            items = data.get("data", []) if isinstance(data, dict) else []
            companies = []
            for item in items:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    companies.append({
                        "ticker": str(item[0]),
                        "name": str(item[1]),
                        "cik": str(item[0]),
                        "exchange": "TWSE",
                        "country": "TW",
                        "market_id": self.market_id,
                    })
            if query:
                q = query.lower()
                companies = [c for c in companies if q in c["ticker"].lower() or q in c["name"].lower()]
            return companies
        except Exception:
            return []

    def search_company(self, name: str) -> list[dict[str, Any]]:
        return self.list_companies(query=name)

    def get_profile(self, identifier: str) -> dict[str, Any]:
        cached = self._read_cache(identifier, "profile.json")
        if cached:
            return cached

        # Retry with backoff to handle TWSE rate limiting / transient
        # timeouts during concurrent batch runs.
        matches: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                matches = self.search_company(identifier)
                if matches:
                    break
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "MOPS get_profile attempt %d/3 for %s failed: %s",
                    attempt, identifier, exc,
                )
            # Backoff: 1s, 2s, 4s between retries.
            time.sleep(min(2 ** (attempt - 1), 4))

        raw_profile = {
            "name": matches[0]["name"] if matches else "",
            "ticker": identifier,
            "isin": "",
            "country": "TW",
            "sector": "",
            "industry": "",
            "exchange": "TWSE",
            "currency": "TWD",
            "cik": identifier,
        }

        from operator1.clients.canonical_translator import translate_profile
        profile = translate_profile(raw_profile, self.market_id)
        self._write_cache(identifier, "profile.json", profile)
        return profile

    def get_income_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_mops_financials(identifier, "income")

    def get_balance_sheet(self, identifier: str) -> pd.DataFrame:
        return self._fetch_mops_financials(identifier, "balance")

    def get_cashflow_statement(self, identifier: str) -> pd.DataFrame:
        return self._fetch_mops_financials(identifier, "cashflow")

    def _fetch_mops_financials(self, identifier: str, statement_type: str) -> pd.DataFrame:
        """Scrape financial statements from MOPS via form POST.

        MOPS endpoints use form POST with parameters:
        - encodeURIComponent: 1
        - step: 1
        - firstin: 1
        - off: 1
        - co_id: stock code
        - year: ROC year
        - season: 1-4 (quarter)
        """
        step_map = {
            "income": "t163sb04",  # Income statement
            "balance": "t163sb05", # Balance sheet
            "cashflow": "t163sb20", # Cash flow
        }
        step = step_map.get(statement_type, "t163sb04")

        rows: list[dict] = []
        current_year = date.today().year
        roc_current = _ce_to_roc(current_year)

        for roc_year in range(_ce_to_roc(current_year - 2), roc_current + 1):
            for season in range(1, 5):
                ce_year = _roc_to_ce(roc_year)
                # Skip future quarters
                if ce_year == current_year and season > (date.today().month - 1) // 3:
                    continue

                form_data = {
                    "encodeURIComponent": "1",
                    "step": "1",
                    "firstin": "1",
                    "off": "1",
                    "co_id": identifier,
                    "year": str(roc_year),
                    "season": str(season),
                }

                try:
                    result = cached_post(
                        f"{_MOPS_BASE}/mops/web/ajax_{step}",
                        form_data=form_data,
                        headers={
                            **self._session_headers,
                            "Content-Type": "application/x-www-form-urlencoded",
                        },
                    )

                    if isinstance(result, str):
                        # Parse HTML table response
                        parsed = self._parse_mops_html(result, statement_type)
                        month_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
                        period_end = f"{ce_year}-{month_end[season]}"
                        for concept, value in parsed.items():
                            rows.append({
                                "concept": concept,
                                "value": value,
                                "filing_date": period_end,
                                "report_date": period_end,
                                "period_type": "annual" if season == 4 else "quarterly",
                            })
                except Exception as exc:
                    logger.debug("MOPS %s scrape failed for %s/%s/Q%s: %s",
                               statement_type, identifier, roc_year, season, exc)

                time.sleep(1)  # Rate limit: 1 second between requests

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        for col in ("filing_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        from operator1.clients.canonical_translator import translate_financials
        return translate_financials(df, self.market_id, statement_type)

    def _parse_mops_html(self, html: str, statement_type: str) -> dict[str, float]:
        """Parse MOPS HTML table response into concept -> value dict."""
        from operator1.clients.canonical_translator import _TIFRS_MAP
        result: dict[str, float] = {}

        try:
            # Use pandas to parse HTML tables
            dfs = pd.read_html(html, encoding="utf-8")
            if not dfs:
                return result

            df = dfs[0]
            for _, row in df.iterrows():
                if len(row) < 2:
                    continue
                label = str(row.iloc[0]).strip()
                canonical = _TIFRS_MAP.get(label)
                if canonical:
                    try:
                        val_str = str(row.iloc[1]).replace(",", "").strip()
                        result[canonical] = float(val_str)
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass

        return result

    def get_quotes(self, identifier: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_peers(self, identifier: str) -> list[str]:
        return []

    def get_executives(self, identifier: str) -> list[dict[str, Any]]:
        return []
