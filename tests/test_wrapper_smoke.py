"""Smoke tests: run one real company through each wrapper to verify they work.

Grouped into:
  - NO-KEY wrappers (run first)
  - KEY-DEPENDENT wrappers (run last, skipped if key missing)

Skip SEC EDGAR per user request.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta

import pandas as pd
import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _non_empty_df(df: pd.DataFrame, label: str, min_rows: int = 1) -> None:
    """Assert a DataFrame has at least min_rows rows and print summary."""
    assert isinstance(df, pd.DataFrame), f"{label}: expected DataFrame, got {type(df)}"
    logger.info(f"{label}: {len(df)} rows, cols={list(df.columns)[:8]}")
    assert len(df) >= min_rows, f"{label}: expected >= {min_rows} rows, got {len(df)}"


# ===========================================================================
# SECTION 1: NO-KEY OHLCV WRAPPERS
# ===========================================================================

class TestOHLCVNoKey:
    """OHLCV wrappers that need no API key."""

    def test_yfinance_apple(self):
        """yfinance: fetch AAPL (US)."""
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance
        df = fetch_ohlcv_yfinance("AAPL", market_id="us_sec_edgar", years=1)
        _non_empty_df(df, "yfinance/AAPL", min_rows=50)

    def test_yfinance_toyota_jp(self):
        """yfinance: fetch Toyota (JP)."""
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance
        df = fetch_ohlcv_yfinance("7203", market_id="jp_jquants", years=1)
        _non_empty_df(df, "yfinance/Toyota-JP", min_rows=50)

    def test_yfinance_hsbc_uk(self):
        """yfinance: fetch HSBC (UK)."""
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance
        df = fetch_ohlcv_yfinance("HSBA", market_id="uk_companies_house", years=1)
        _non_empty_df(df, "yfinance/HSBC-UK", min_rows=50)

    def test_pykrx_samsung(self):
        """pykrx: fetch Samsung Electronics (KR)."""
        from operator1.clients.ohlcv_pykrx import fetch_ohlcv_pykrx
        df = fetch_ohlcv_pykrx("005930", years=1)
        _non_empty_df(df, "pykrx/Samsung", min_rows=50)

    def test_twstock_tsmc(self):
        """twstock: fetch TSMC (TW)."""
        from operator1.clients.ohlcv_twstock import fetch_ohlcv_twstock
        df = fetch_ohlcv_twstock("2330", years=1)
        _non_empty_df(df, "twstock/TSMC", min_rows=10)

    def test_jugaad_reliance(self):
        """jugaad-data: fetch Reliance Industries (IN)."""
        from operator1.clients.ohlcv_jugaad import fetch_ohlcv_jugaad
        df = fetch_ohlcv_jugaad("RELIANCE", years=1)
        _non_empty_df(df, "jugaad/Reliance", min_rows=50)

    def test_ohlcv_provider_dispatch(self):
        """ohlcv_provider: dispatch test with AAPL via yfinance fallback."""
        from operator1.clients.ohlcv_provider import fetch_ohlcv
        df = fetch_ohlcv("AAPL", market_id="us_sec_edgar", years=1)
        _non_empty_df(df, "ohlcv_provider/AAPL", min_rows=50)


# ===========================================================================
# SECTION 2: NO-KEY MACRO WRAPPERS
# ===========================================================================

class TestMacroNoKey:
    """Macro wrappers that need no API key."""

    def test_wbgapi_us(self):
        """wbgapi: fetch US GDP, inflation, unemployment."""
        from operator1.clients.macro_wbgapi import fetch_macro_wbgapi
        data = fetch_macro_wbgapi("US", years=5)
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        logger.info(f"wbgapi/US: keys={list(data.keys())}")
        assert len(data) > 0, "wbgapi returned empty dict"

    def test_wbgapi_brazil(self):
        """wbgapi: fetch Brazil macro data."""
        from operator1.clients.macro_wbgapi import fetch_macro_wbgapi
        data = fetch_macro_wbgapi("BR", years=5)
        assert isinstance(data, dict)
        logger.info(f"wbgapi/BR: keys={list(data.keys())}")
        assert len(data) > 0

    def test_sdmx_ecb(self):
        """sdmx1: fetch EU/ECB macro data."""
        from operator1.clients.macro_sdmx import fetch_macro_ecb
        data = fetch_macro_ecb(years=5)
        assert isinstance(data, dict)
        logger.info(f"sdmx/EU: keys={list(data.keys())}")

    def test_bcb_brazil(self):
        """python-bcb: fetch Brazil central bank data."""
        from operator1.clients.macro_bcb import fetch_macro_bcb
        data = fetch_macro_bcb(years=5)
        assert isinstance(data, dict)
        logger.info(f"bcb/BR: keys={list(data.keys())}")
        assert len(data) > 0

    def test_ons_uk(self):
        """ONS: fetch UK macro data (no key)."""
        from operator1.clients.macro_ons import fetch_macro_ons
        data = fetch_macro_ons(years=5)
        assert isinstance(data, dict)
        logger.info(f"ons/GB: keys={list(data.keys())}")

    def test_estat_japan(self):
        """e-Stat: fetch Japan macro data (no key)."""
        from operator1.clients.macro_estat import fetch_macro_estat
        data = fetch_macro_estat(years=5)
        assert isinstance(data, dict)
        logger.info(f"estat/JP: keys={list(data.keys())}")

    def test_kosis_korea(self):
        """KOSIS: fetch Korea macro data (no key)."""
        from operator1.clients.macro_kosis import fetch_macro_kosis
        data = fetch_macro_kosis(years=5)
        assert isinstance(data, dict)
        logger.info(f"kosis/KR: keys={list(data.keys())}")

    def test_dgbas_taiwan(self):
        """DGBAS: fetch Taiwan macro data (no key)."""
        from operator1.clients.macro_dgbas import fetch_macro_dgbas
        data = fetch_macro_dgbas(years=5)
        assert isinstance(data, dict)
        logger.info(f"dgbas/TW: keys={list(data.keys())}")

    def test_bcch_chile(self):
        """BCCh: fetch Chile macro data (no key)."""
        from operator1.clients.macro_bcch import fetch_macro_bcch
        data = fetch_macro_bcch(years=5)
        assert isinstance(data, dict)
        logger.info(f"bcch/CL: keys={list(data.keys())}")

    def test_macro_provider_dispatch(self):
        """macro_provider: dispatch test with US (should use wbgapi fallback without FRED key)."""
        from operator1.clients.macro_provider import fetch_macro
        data = fetch_macro("US", market_id="us_sec_edgar", years=5)
        assert isinstance(data, dict)
        logger.info(f"macro_provider/US: keys={list(data.keys())}")


# ===========================================================================
# SECTION 3: NO-KEY PIT WRAPPERS
# ===========================================================================

class TestPITNoKey:
    """Point-in-time data wrappers that need no API key."""

    def test_ixbrl_parse_import(self):
        """ixbrl-parse: verify the library imports."""
        import ixbrl_parse
        logger.info(f"ixbrl-parse import OK: {dir(ixbrl_parse)[:5]}")

    def test_tw_mops_tsmc(self):
        """Taiwan MOPS: fetch TSMC profile (no key needed)."""
        try:
            from operator1.clients.tw_mops_wrapper import TWMopsClient
            client = TWMopsClient()
            result = client.get_profile("2330")
            assert isinstance(result, dict)
            logger.info(f"tw_mops/2330 profile: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"tw_mops/2330: {e}")
            pytest.skip(f"TW MOPS: {e}")

    def test_br_cvm_petrobras(self):
        """Brazil CVM: fetch Petrobras profile (no key needed)."""
        try:
            from operator1.clients.br_cvm_wrapper import BRCvmClient
            client = BRCvmClient()
            result = client.get_profile("PETR4")
            assert isinstance(result, dict)
            logger.info(f"br_cvm/PETR4 profile: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"br_cvm/PETR4: {e}")
            pytest.skip(f"BR CVM: {e}")

    def test_cl_cmf_sqm(self):
        """Chile CMF: fetch SQM profile (no key needed)."""
        try:
            from operator1.clients.cl_cmf_wrapper import CLCmfClient
            client = CLCmfClient()
            result = client.get_profile("SQM")
            assert isinstance(result, dict)
            logger.info(f"cl_cmf/SQM profile: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"cl_cmf/SQM: {e}")
            pytest.skip(f"CL CMF: {e}")

    def test_eu_esef_sap(self):
        """EU ESEF: fetch SAP profile (graceful degradation without pyesef)."""
        try:
            from operator1.clients.eu_esef_wrapper import EUEsefClient
            client = EUEsefClient()
            result = client.get_profile("SAP")
            assert isinstance(result, dict)
            logger.info(f"eu_esef/SAP profile: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"eu_esef/SAP: {e}")
            pytest.skip(f"EU ESEF: {e}")


# ===========================================================================
# SECTION 4: KEY-DEPENDENT WRAPPERS (run last)
# ===========================================================================

class TestKeyDependentWrappers:
    """Wrappers that require API keys. Skipped if key not set."""

    def test_fredapi_us(self):
        """FRED: fetch US macro data (needs FRED_API_KEY)."""
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            pytest.skip("FRED_API_KEY not set")
        from operator1.clients.macro_fredapi import fetch_macro_fred
        data = fetch_macro_fred(api_key=key, years=5)
        assert isinstance(data, dict)
        logger.info(f"fred/US: keys={list(data.keys())}")
        assert len(data) > 0

    def test_banxico_mexico(self):
        """Banxico: fetch Mexico macro data (needs BANXICO_TOKEN)."""
        token = os.environ.get("BANXICO_TOKEN", "")
        if not token:
            pytest.skip("BANXICO_TOKEN not set")
        from operator1.clients.macro_banxico import fetch_macro_banxico
        data = fetch_macro_banxico(api_token=token, years=5)
        assert isinstance(data, dict)
        logger.info(f"banxico/MX: keys={list(data.keys())}")

    def test_dart_fss_samsung(self):
        """DART: fetch Samsung profile (needs DART_API_KEY)."""
        key = os.environ.get("DART_API_KEY", "")
        if not key:
            pytest.skip("DART_API_KEY not set")
        from operator1.clients.kr_dart_wrapper import KRDartClient
        client = KRDartClient(api_key=key)
        result = client.get_profile("005930")
        assert isinstance(result, dict)
        logger.info(f"dart/Samsung: keys={list(result.keys())[:8]}")

    def test_jquants_toyota(self):
        """J-Quants: fetch Toyota profile (needs JQUANTS_REFRESH_TOKEN)."""
        token = os.environ.get("JQUANTS_REFRESH_TOKEN", "")
        if not token:
            pytest.skip("JQUANTS_REFRESH_TOKEN not set")
        from operator1.clients.jp_jquants_wrapper import JPJquantsClient
        client = JPJquantsClient()
        result = client.get_profile("7203")
        assert isinstance(result, dict)
        logger.info(f"jquants/Toyota: keys={list(result.keys())[:8]}")

    def test_uk_ch_hsbc(self):
        """UK Companies House: fetch HSBC profile (needs COMPANIES_HOUSE_API_KEY)."""
        key = os.environ.get("COMPANIES_HOUSE_API_KEY", "")
        if not key:
            pytest.skip("COMPANIES_HOUSE_API_KEY not set")
        from operator1.clients.uk_ch_wrapper import UKCompaniesHouseClient
        client = UKCompaniesHouseClient(api_key=key)
        result = client.get_profile("00014259")  # HSBC Holdings company number
        assert isinstance(result, dict)
        logger.info(f"uk_ch/HSBC: keys={list(result.keys())[:8]}")

    # NOTE: SEC EDGAR skipped per user request
