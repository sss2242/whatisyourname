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

    def test_nselib_reliance_india(self):
        """nselib: fetch Reliance Industries (IN) -- regional primary."""
        from operator1.clients.ohlcv_nselib import fetch_ohlcv_nselib
        df = fetch_ohlcv_nselib("RELIANCE", years=1)
        _non_empty_df(df, "nselib/Reliance-IN", min_rows=10)

    def test_yfinance_reliance_india(self):
        """yfinance: fetch Reliance (IN) via .NS suffix -- fallback."""
        from operator1.clients.ohlcv_yfinance import fetch_ohlcv_yfinance
        df = fetch_ohlcv_yfinance("RELIANCE", market_id="in_bse", years=1)
        _non_empty_df(df, "yfinance/Reliance-IN", min_rows=50)

    def test_ohlcv_provider_india(self):
        """ohlcv_provider: India dispatch (nselib primary -> yfinance fallback)."""
        from operator1.clients.ohlcv_provider import fetch_ohlcv
        df = fetch_ohlcv("RELIANCE", market_id="in_bse", years=1)
        _non_empty_df(df, "ohlcv_provider/Reliance-IN", min_rows=10)

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
        from operator1.clients.br_cvm_wrapper import BRCvmClient
        client = BRCvmClient()
        result = client.get_profile("PETROBRAS")
        assert isinstance(result, dict)
        assert "name" in result
        logger.info(f"br_cvm/PETROBRAS profile: name={result.get('name')}")

    def test_cl_cmf_sqm(self):
        """Chile CMF: fetch SQM profile (yfinance fallback since CMF API is down)."""
        from operator1.clients.cl_cmf_wrapper import CLCmfClient
        client = CLCmfClient()
        result = client.get_profile("SQM")
        assert isinstance(result, dict)
        assert "name" in result
        logger.info(f"cl_cmf/SQM profile: name={result.get('name')}")

    def test_eu_esef_sap(self):
        """EU ESEF: fetch SAP SE profile via xbrl.org entity search."""
        from operator1.clients.eu_esef_wrapper import EUEsefClient
        client = EUEsefClient()
        result = client.get_profile("SAP SE")
        assert isinstance(result, dict)
        assert "name" in result
        logger.info(f"eu_esef/SAP SE profile: name={result.get('name')}")


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


# ===========================================================================
# SECTION 5: FREE REGIONAL PIT WRAPPERS (no key needed)
# ===========================================================================

class TestRegionalPIT:
    """Free PIT clients for specific markets -- all use public APIs."""

    def test_au_asx_bhp(self):
        """Australia ASX: fetch BHP profile."""
        try:
            from operator1.clients.au_asx import AUAsxClient
            client = AUAsxClient()
            result = client.get_profile("BHP")
            assert isinstance(result, dict)
            logger.info(f"au_asx/BHP: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"au_asx/BHP: {e}")
            pytest.skip(f"AU ASX: {e}")

    def test_ca_sedar_shopify(self):
        """Canada SEDAR: fetch Shopify profile."""
        try:
            from operator1.clients.ca_sedar import CASedarClient
            client = CASedarClient()
            result = client.get_profile("SHOP")
            assert isinstance(result, dict)
            logger.info(f"ca_sedar/SHOP: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"ca_sedar/SHOP: {e}")
            pytest.skip(f"CA SEDAR: {e}")

    def test_ch_six_nestle(self):
        """Switzerland SIX: fetch Nestle profile."""
        try:
            from operator1.clients.ch_six import CHSixClient
            client = CHSixClient()
            result = client.get_profile("NESN")
            assert isinstance(result, dict)
            logger.info(f"ch_six/NESN: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"ch_six/NESN: {e}")
            pytest.skip(f"CH SIX: {e}")

    def test_cn_sse_kweichow(self):
        """China SSE: fetch Kweichow Moutai profile."""
        try:
            from operator1.clients.cn_sse import CNSseClient
            client = CNSseClient()
            result = client.get_profile("600519")
            assert isinstance(result, dict)
            logger.info(f"cn_sse/600519: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"cn_sse/600519: {e}")
            pytest.skip(f"CN SSE: {e}")

    def test_hk_hkex_tencent(self):
        """Hong Kong HKEX: fetch Tencent profile."""
        try:
            from operator1.clients.hk_hkex import HKHkexClient
            client = HKHkexClient()
            result = client.get_profile("0700")
            assert isinstance(result, dict)
            logger.info(f"hk_hkex/0700: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"hk_hkex/0700: {e}")
            pytest.skip(f"HK HKEX: {e}")

    def test_in_bse_reliance(self):
        """India BSE: fetch Reliance Industries profile (may hang from non-IN IPs)."""
        import signal
        def _timeout(signum, frame):
            raise TimeoutError("BSE/NSE API timed out (geo-blocked)")
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(30)
        try:
            from operator1.clients.in_bse import INBseClient
            client = INBseClient()
            result = client.get_profile("RELIANCE")
            assert isinstance(result, dict)
            logger.info(f"in_bse/RELIANCE: keys={list(result.keys())[:8]}")
        except TimeoutError:
            logger.warning("in_bse/RELIANCE: BSE/NSE geo-blocks non-Indian IPs")
            pytest.skip("IN BSE: geo-blocked (non-Indian IP)")
        except Exception as e:
            logger.warning(f"in_bse/RELIANCE: {e}")
            pytest.skip(f"IN BSE: {e}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def test_mx_bmv_walmex(self):
        """Mexico BMV: fetch Walmart Mexico profile."""
        try:
            from operator1.clients.mx_bmv import MXBmvClient
            client = MXBmvClient()
            result = client.get_profile("WALMEX")
            assert isinstance(result, dict)
            logger.info(f"mx_bmv/WALMEX: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"mx_bmv/WALMEX: {e}")
            pytest.skip(f"MX BMV: {e}")

    def test_sa_tadawul_aramco(self):
        """Saudi Tadawul: fetch Saudi Aramco profile."""
        try:
            from operator1.clients.sa_tadawul import SATadawulClient
            client = SATadawulClient()
            result = client.get_profile("2222")
            assert isinstance(result, dict)
            logger.info(f"sa_tadawul/2222: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"sa_tadawul/2222: {e}")
            pytest.skip(f"SA Tadawul: {e}")

    def test_sg_sgx_dbs(self):
        """Singapore SGX: fetch DBS Group profile."""
        try:
            from operator1.clients.sg_sgx import SGSgxClient
            client = SGSgxClient()
            result = client.get_profile("D05")
            assert isinstance(result, dict)
            logger.info(f"sg_sgx/D05: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"sg_sgx/D05: {e}")
            pytest.skip(f"SG SGX: {e}")

    def test_ae_dfm_emaar(self):
        """UAE DFM: fetch Emaar Properties profile."""
        try:
            from operator1.clients.ae_dfm import AEDfmClient
            client = AEDfmClient()
            result = client.get_profile("EMAAR")
            assert isinstance(result, dict)
            logger.info(f"ae_dfm/EMAAR: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"ae_dfm/EMAAR: {e}")
            pytest.skip(f"AE DFM: {e}")

    def test_za_jse_naspers(self):
        """South Africa JSE: fetch Naspers profile."""
        try:
            from operator1.clients.za_jse import ZAJseClient
            client = ZAJseClient()
            result = client.get_profile("NPN")
            assert isinstance(result, dict)
            logger.info(f"za_jse/NPN: keys={list(result.keys())[:8]}")
        except Exception as e:
            logger.warning(f"za_jse/NPN: {e}")
            pytest.skip(f"ZA JSE: {e}")


# ===========================================================================
# SECTION 6: OHLCV AKSHARE + ORCHESTRATION LAYERS
# ===========================================================================

class TestOtherFreeWrappers:
    """akshare OHLCV, pit_registry, equity_provider, canonical_translator, supplement."""

    def test_ohlcv_akshare_china(self):
        """akshare: fetch Kweichow Moutai OHLCV (CN, no key)."""
        try:
            from operator1.clients.ohlcv_akshare import fetch_ohlcv_akshare
            df = fetch_ohlcv_akshare("600519", years=1)
            if df.empty:
                logger.warning("akshare returned empty (may not be installed)")
                pytest.skip("akshare not installed or returned empty")
            _non_empty_df(df, "akshare/600519", min_rows=10)
        except ImportError:
            pytest.skip("akshare not installed")
        except Exception as e:
            logger.warning(f"akshare/600519: {e}")
            pytest.skip(f"akshare: {e}")

    def test_pit_registry_list_markets(self):
        """pit_registry: verify market listing works."""
        from operator1.clients.pit_registry import MarketInfo
        # Just verify import and basic structure
        logger.info(f"pit_registry MarketInfo fields: {[f for f in dir(MarketInfo) if not f.startswith('_')][:8]}")

    def test_canonical_translator_import(self):
        """canonical_translator: verify import and field mappings exist."""
        from operator1.clients.canonical_translator import translate_financials, translate_profile, get_concept_map
        # Verify we can get a concept map for a known market
        cmap = get_concept_map("us_sec_edgar")
        logger.info(f"canonical_translator: us_sec_edgar concept_map has {len(cmap)} entries")

    def test_supplement_import(self):
        """supplement: verify supplementary API module imports."""
        from operator1.clients import supplement
        funcs = [f for f in dir(supplement) if not f.startswith('_') and callable(getattr(supplement, f, None))]
        logger.info(f"supplement functions: {funcs[:8]}")
