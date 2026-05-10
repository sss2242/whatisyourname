"""Global Point-in-Time (PIT) API registry.

Maps regions and markets to their respective free, immutable PIT data
sources.  The user selects a region, then picks a company from the
available exchange listings.

Supported regions (Tier 1 -- all free, immutable PIT APIs):

  Region          | Country      | Exchange         | Market Cap | PIT API
  ----------------+--------------+------------------+------------+-----------
  North America   | US           | NYSE/NASDAQ      | $50T       | SEC EDGAR
  Europe          | UK           | LSE              | $3.18T     | Companies House
  Europe          | EU (pan-EU)  | ESEF             | $8-9T      | ESEF/XBRL
  Europe          | France       | Paris/Euronext   | $3.13T     | ESEF
  Europe          | Germany      | Frankfurt/XETRA  | $2.04T     | ESEF
  Asia            | Japan        | Tokyo (JPX)      | $6.5T      | EDINET
  Asia            | South Korea  | KOSPI/KOSDAQ     | $2.5T      | DART
  Asia            | Taiwan       | TWSE/TPEX        | ~$1.2T     | MOPS
  South America   | Brazil       | B3               | $2.2T      | CVM
  South America   | Chile        | CMF              | $0.4T      | CMF API

Total Tier 1 coverage: $91+ trillion in market cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketInfo:
    """Metadata for a single supported market / exchange."""

    market_id: str          # unique key, e.g. "us_sec_edgar"
    country: str            # display name, e.g. "United States"
    country_code: str       # ISO-2, e.g. "US"
    region: str             # continent/group, e.g. "North America"
    exchange: str           # exchange name(s), e.g. "NYSE / NASDAQ"
    market_cap: str         # human-readable, e.g. "$50T"
    pit_api_name: str       # data source label, e.g. "SEC EDGAR"
    pit_api_url: str        # base URL of the PIT API
    requires_api_key: bool  # whether an API key is needed
    tier: int = 1           # 1 = launch, 2 = stretch
    phase: int = 1          # implementation phase (1 or 2)
    client_module: str = "" # dotted path to the client class module
    personal_data_level: str = "none"  # none | low | medium | high
    input_requirements: str = ""       # human-readable API registration requirements


@dataclass
class CompanyListing:
    """A single company entry returned by a PIT API's company list."""

    ticker: str
    name: str
    cik: str = ""           # SEC CIK or equivalent filing ID
    isin: str = ""
    exchange: str = ""
    sector: str = ""
    industry: str = ""
    country: str = ""
    market_id: str = ""     # back-reference to MarketInfo.market_id


# ---------------------------------------------------------------------------
# Registry of all supported PIT markets
# ---------------------------------------------------------------------------

MARKETS: dict[str, MarketInfo] = {
    # --- North America ---
    "us_sec_edgar": MarketInfo(
        market_id="us_sec_edgar",
        country="United States",
        country_code="US",
        region="North America",
        exchange="NYSE / NASDAQ",
        market_cap="$50T",
        pit_api_name="SEC EDGAR",
        pit_api_url="https://efts.sec.gov/LATEST",
        requires_api_key=False,
        tier=1,
        phase=1,
        client_module="operator1.clients.us_edgar",
        personal_data_level="low",
        input_requirements="Email address (as User-Agent header)",
    ),

    # --- Europe ---
    "uk_companies_house": MarketInfo(
        market_id="uk_companies_house",
        country="United Kingdom",
        country_code="GB",
        region="Europe",
        exchange="LSE",
        market_cap="$3.18T",
        pit_api_name="Companies House",
        pit_api_url="https://api.company-information.service.gov.uk",
        requires_api_key=True,
        tier=1,
        phase=1,
        client_module="operator1.clients.uk_ch_wrapper",
        personal_data_level="low",
        input_requirements="Email, name, application description",
    ),
    "eu_esef": MarketInfo(
        market_id="eu_esef",
        country="European Union",
        country_code="EU",
        region="Europe",
        exchange="ESEF (pan-EU: Euronext, Frankfurt, etc.)",
        market_cap="$8-9T",
        pit_api_name="ESEF / XBRL Europe",
        pit_api_url="https://filings.xbrl.org/api",
        requires_api_key=False,
        tier=1,
        phase=1,
        client_module="operator1.clients.eu_esef_wrapper",
    ),
    "fr_esef": MarketInfo(
        market_id="fr_esef",
        country="France",
        country_code="FR",
        region="Europe",
        exchange="Paris / Euronext",
        market_cap="$3.13T",
        pit_api_name="ESEF (France)",
        pit_api_url="https://filings.xbrl.org/api",
        requires_api_key=False,
        tier=1,
        phase=1,
        client_module="operator1.clients.eu_esef_wrapper",
    ),
    "de_esef": MarketInfo(
        market_id="de_esef",
        country="Germany",
        country_code="DE",
        region="Europe",
        exchange="Frankfurt / XETRA",
        market_cap="$2.04T",
        pit_api_name="ESEF (Germany)",
        pit_api_url="https://filings.xbrl.org/api",
        requires_api_key=False,
        tier=1,
        phase=1,
        client_module="operator1.clients.eu_esef_wrapper",
    ),

    # --- Asia ---
    "jp_jquants": MarketInfo(
        market_id="jp_jquants",
        country="Japan",
        country_code="JP",
        region="Asia",
        exchange="Tokyo Stock Exchange (JPX)",
        market_cap="$6.5T",
        pit_api_name="J-Quants",
        pit_api_url="https://jpx-jquants.com/",
        requires_api_key=True,
        tier=1,
        phase=1,
        client_module="operator1.clients.jp_jquants_wrapper",
        personal_data_level="low",
        input_requirements="Email registration only at jpx-jquants.com (free plan)",
    ),
    "kr_dart": MarketInfo(
        market_id="kr_dart",
        country="South Korea",
        country_code="KR",
        region="Asia",
        exchange="KOSPI / KOSDAQ",
        market_cap="$2.5T",
        pit_api_name="DART",
        pit_api_url="https://opendart.fss.or.kr/api",
        requires_api_key=True,
        tier=1,
        phase=1,
        client_module="operator1.clients.kr_dart_wrapper",
        personal_data_level="medium",
        input_requirements="Email, name, usage purpose, IP address (if corporate)",
    ),
    "tw_mops": MarketInfo(
        market_id="tw_mops",
        country="Taiwan",
        country_code="TW",
        region="Asia",
        exchange="TWSE / TPEX",
        market_cap="~$1.2T",
        pit_api_name="MOPS",
        pit_api_url="https://mops.twse.com.tw",
        requires_api_key=False,
        tier=1,
        phase=2,
        client_module="operator1.clients.tw_mops_wrapper",
    ),

    # --- South America ---
    "br_cvm": MarketInfo(
        market_id="br_cvm",
        country="Brazil",
        country_code="BR",
        region="South America",
        exchange="B3",
        market_cap="$2.2T",
        pit_api_name="CVM",
        pit_api_url="https://dados.cvm.gov.br/api/v1",
        requires_api_key=False,
        tier=1,
        phase=2,
        client_module="operator1.clients.br_cvm_wrapper",
    ),
    "cl_cmf": MarketInfo(
        market_id="cl_cmf",
        country="Chile",
        country_code="CL",
        region="South America",
        exchange="Santiago Stock Exchange",
        market_cap="$0.4T",
        pit_api_name="CMF",
        pit_api_url="https://www.cmfchile.cl/api",
        requires_api_key=False,
        tier=1,
        phase=2,
        client_module="operator1.clients.cl_cmf_wrapper",
    ),

    # --- Phase 2: New markets (Tier 2) ---

    # North America
    "ca_sedar": MarketInfo(
        market_id="ca_sedar",
        country="Canada",
        country_code="CA",
        region="North America",
        exchange="TSX / TSXV",
        market_cap="~$3T",
        pit_api_name="SEDAR+",
        pit_api_url="https://www.sedarplus.ca",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.ca_sedar",
    ),

    # Oceania
    "au_asx": MarketInfo(
        market_id="au_asx",
        country="Australia",
        country_code="AU",
        region="Oceania",
        exchange="ASX",
        market_cap="~$1.8T",
        pit_api_name="ASX API",
        pit_api_url="https://www.asx.com.au/asx/1",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.au_asx",
    ),

    # Asia (additional)
    "in_bse": MarketInfo(
        market_id="in_bse",
        country="India",
        country_code="IN",
        region="Asia",
        exchange="BSE / NSE",
        market_cap="~$4T",
        pit_api_name="BSE India",
        pit_api_url="https://api.bseindia.com",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.in_bse",
    ),
    "cn_sse": MarketInfo(
        market_id="cn_sse",
        country="China",
        country_code="CN",
        region="Asia",
        exchange="SSE / SZSE",
        market_cap="~$10T",
        pit_api_name="SSE / CSRC",
        pit_api_url="http://www.sse.com.cn",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.cn_sse",
    ),
    "hk_hkex": MarketInfo(
        market_id="hk_hkex",
        country="Hong Kong",
        country_code="HK",
        region="Asia",
        exchange="HKEX",
        market_cap="~$4.5T",
        pit_api_name="HKEX",
        pit_api_url="https://www.hkexnews.hk",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.hk_hkex",
    ),
    "sg_sgx": MarketInfo(
        market_id="sg_sgx",
        country="Singapore",
        country_code="SG",
        region="Asia",
        exchange="SGX",
        market_cap="~$0.6T",
        pit_api_name="SGX",
        pit_api_url="https://www.sgx.com",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.sg_sgx",
    ),

    # Latin America (additional)
    "mx_bmv": MarketInfo(
        market_id="mx_bmv",
        country="Mexico",
        country_code="MX",
        region="Latin America",
        exchange="BMV",
        market_cap="~$0.5T",
        pit_api_name="BMV",
        pit_api_url="https://www.bmv.com.mx",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.mx_bmv",
    ),

    # Africa
    "za_jse": MarketInfo(
        market_id="za_jse",
        country="South Africa",
        country_code="ZA",
        region="Africa",
        exchange="JSE",
        market_cap="~$1T",
        pit_api_name="JSE",
        pit_api_url="https://www.jse.co.za",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.za_jse",
    ),

    # Europe (additional -- non-ESEF)
    "ch_six": MarketInfo(
        market_id="ch_six",
        country="Switzerland",
        country_code="CH",
        region="Europe",
        exchange="SIX",
        market_cap="~$1.8T",
        pit_api_name="SIX",
        pit_api_url="https://www.six-group.com",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.ch_six",
    ),

    # Europe (ESEF-based -- reuse eu_esef_wrapper with country filter)
    "nl_esef": MarketInfo(
        market_id="nl_esef",
        country="Netherlands",
        country_code="NL",
        region="Europe",
        exchange="Euronext Amsterdam",
        market_cap="~$1.2T",
        pit_api_name="ESEF (Netherlands)",
        pit_api_url="https://filings.xbrl.org/api",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.eu_esef_wrapper",
    ),
    "es_esef": MarketInfo(
        market_id="es_esef",
        country="Spain",
        country_code="ES",
        region="Europe",
        exchange="BME (Madrid)",
        market_cap="~$0.7T",
        pit_api_name="ESEF (Spain)",
        pit_api_url="https://filings.xbrl.org/api",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.eu_esef_wrapper",
    ),
    "it_esef": MarketInfo(
        market_id="it_esef",
        country="Italy",
        country_code="IT",
        region="Europe",
        exchange="Borsa Italiana",
        market_cap="~$0.8T",
        pit_api_name="ESEF (Italy)",
        pit_api_url="https://filings.xbrl.org/api",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.eu_esef_wrapper",
    ),
    "se_esef": MarketInfo(
        market_id="se_esef",
        country="Sweden",
        country_code="SE",
        region="Europe",
        exchange="Nasdaq Stockholm",
        market_cap="~$0.9T",
        pit_api_name="ESEF (Sweden)",
        pit_api_url="https://filings.xbrl.org/api",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.eu_esef_wrapper",
    ),

    # Middle East
    "sa_tadawul": MarketInfo(
        market_id="sa_tadawul",
        country="Saudi Arabia",
        country_code="SA",
        region="Middle East",
        exchange="Tadawul",
        market_cap="~$2.7T",
        pit_api_name="Tadawul",
        pit_api_url="https://www.saudiexchange.sa",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.sa_tadawul",
    ),
    "ae_dfm": MarketInfo(
        market_id="ae_dfm",
        country="UAE",
        country_code="AE",
        region="Middle East",
        exchange="DFM / ADX",
        market_cap="~$0.8T",
        pit_api_name="DFM / ADX",
        pit_api_url="https://www.dfm.ae",
        requires_api_key=False,
        tier=2,
        phase=2,
        client_module="operator1.clients.ae_dfm",
    ),
}


# ---------------------------------------------------------------------------
# Macro economic data API registry (free, per-region)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MacroAPIInfo:
    """Metadata for a free macroeconomic data API."""

    macro_id: str           # unique key, e.g. "us_fred"
    country: str            # display name
    country_code: str       # ISO-2
    region: str             # continent/group
    api_name: str           # e.g. "FRED"
    api_url: str            # base URL
    requires_api_key: bool
    description: str        # what it provides
    # Key series IDs for survival mode analysis
    series_gdp: str = ""           # GDP series ID
    series_inflation: str = ""     # CPI / inflation rate
    series_interest_rate: str = "" # policy / benchmark rate
    series_unemployment: str = ""  # unemployment rate
    series_currency: str = ""      # exchange rate vs USD


# Maps each market's country_code to its macro data source.
# Every region gets at least one free macro API for survival analysis.
MACRO_APIS: dict[str, MacroAPIInfo] = {

    # --- North America ---
    "us_fred": MacroAPIInfo(
        macro_id="us_fred",
        country="United States",
        country_code="US",
        region="North America",
        api_name="FRED (Federal Reserve Economic Data)",
        api_url="https://api.stlouisfed.org/fred",
        requires_api_key=True,  # free registration
        description="US Federal Reserve: GDP, CPI, rates, employment, yield curves",
        series_gdp="GDP",
        series_inflation="CPIAUCSL",
        series_interest_rate="FEDFUNDS",
        series_unemployment="UNRATE",
        series_currency="DTWEXBGS",
    ),

    # --- Europe ---
    "eu_ecb": MacroAPIInfo(
        macro_id="eu_ecb",
        country="European Union",
        country_code="EU",
        region="Europe",
        api_name="ECB SDW (Statistical Data Warehouse)",
        api_url="https://sdw-wsrest.ecb.europa.eu/service",
        requires_api_key=False,
        description="European Central Bank: eurozone GDP, HICP inflation, ECB rates, M3",
        series_gdp="MNA.Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N",
        series_inflation="ICP.M.U2.N.000000.4.ANR",
        series_interest_rate="FM.B.U2.EUR.4F.KR.MRR_FR.LEV",
        series_unemployment="STS.M.I8.S.UNEH.RTT000.4.000",
        series_currency="EXR.D.USD.EUR.SP00.A",
    ),
    "uk_ons": MacroAPIInfo(
        macro_id="uk_ons",
        country="United Kingdom",
        country_code="GB",
        region="Europe",
        api_name="ONS (Office for National Statistics)",
        api_url="https://api.ons.gov.uk/dataset",
        requires_api_key=False,
        description="UK ONS: GDP, CPI, employment, trade balance, housing",
        series_gdp="ABMI",
        series_inflation="D7G7",
        series_interest_rate="IUMABEDR",
        series_unemployment="MGSX",
        series_currency="XUMAUSS",
    ),
    "de_bundesbank": MacroAPIInfo(
        macro_id="de_bundesbank",
        country="Germany",
        country_code="DE",
        region="Europe",
        api_name="Deutsche Bundesbank API",
        api_url="https://api.statistiken.bundesbank.de/rest",
        requires_api_key=False,
        description="Bundesbank: German GDP, Ifo index, industrial production, rates",
        series_gdp="BBDP1.Q.DE.Y.A.AG1.CA010.A.A.A.A.A.V.EUR.B",
        series_inflation="BBDP1.M.DE.Y.VPI.VPI000.R.I15.A",
        series_interest_rate="BBK01.SU0202",
        series_unemployment="BBNZ1.Q.DE.Y.UNQ.AG1.A0000.TT.NT",
        series_currency="BBEX3.D.USD.EUR.BB.AC.000",
    ),
    "fr_insee": MacroAPIInfo(
        macro_id="fr_insee",
        country="France",
        country_code="FR",
        region="Europe",
        api_name="INSEE (Institut National de la Statistique)",
        api_url="https://api.insee.fr/series/BDM/V1",
        requires_api_key=True,  # free registration
        description="French INSEE: GDP, CPI, industrial production, employment",
        series_gdp="010565692",
        series_inflation="001759970",
        series_interest_rate="",  # uses ECB rate
        series_unemployment="001688527",
        series_currency="",  # uses ECB EUR/USD
    ),

    # --- Asia ---
    "jp_estat": MacroAPIInfo(
        macro_id="jp_estat",
        country="Japan",
        country_code="JP",
        region="Asia",
        api_name="e-Stat (Statistics Bureau of Japan)",
        api_url="https://api.e-stat.go.jp/rest/3.0/app",
        requires_api_key=True,  # free registration
        description="Japan e-Stat: GDP, CPI, Tankan, industrial production, labor",
        series_gdp="0003109741",
        series_inflation="0003143513",
        series_interest_rate="",  # BOJ publishes separately
        series_unemployment="0003009482",
        series_currency="",  # use FRED DEXJPUS
    ),
    "kr_kosis": MacroAPIInfo(
        macro_id="kr_kosis",
        country="South Korea",
        country_code="KR",
        region="Asia",
        api_name="KOSIS (Korean Statistical Information)",
        api_url="https://kosis.kr/openapi/Param/statisticsParameterData.do",
        requires_api_key=True,  # free registration
        description="Korean KOSIS: GDP, CPI, BOK base rate, employment, trade",
        series_gdp="101_DT_1B01",
        series_inflation="101_DT_1J17",
        series_interest_rate="731Y001",
        series_unemployment="101_DT_1D07",
        series_currency="731Y003",
    ),
    "tw_dgbas": MacroAPIInfo(
        macro_id="tw_dgbas",
        country="Taiwan",
        country_code="TW",
        region="Asia",
        api_name="DGBAS (Directorate-General of Budget, Taiwan)",
        api_url="https://nstatdb.dgbas.gov.tw/dgbasAll/webMain.aspx",
        requires_api_key=False,
        description="Taiwan DGBAS: GDP, CPI, industrial production. Also uses FRED for rates.",
        series_gdp="NA8101A1Q",
        series_inflation="PR0101A1M",
        series_interest_rate="",  # use FRED
        series_unemployment="LF01A1S",
        series_currency="",  # use FRED
    ),

    # --- South America ---
    "br_bcb": MacroAPIInfo(
        macro_id="br_bcb",
        country="Brazil",
        country_code="BR",
        region="South America",
        api_name="BCB (Banco Central do Brasil)",
        api_url="https://api.bcb.gov.br/dados/serie/bcdata.sgs",
        requires_api_key=False,
        description="Brazilian Central Bank: SELIC rate, IPCA inflation, GDP, FX, employment",
        series_gdp="4380",
        series_inflation="433",
        series_interest_rate="432",
        series_unemployment="24369",
        series_currency="1",
    ),
    "cl_bcch": MacroAPIInfo(
        macro_id="cl_bcch",
        country="Chile",
        country_code="CL",
        region="South America",
        api_name="BCCh (Banco Central de Chile)",
        api_url="https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx",
        requires_api_key=True,  # free registration
        description="Chilean Central Bank: GDP, IPC inflation, TPM rate, copper price, FX",
        series_gdp="PIB_ACTIVIDAD",
        series_inflation="IPC_GENERAL",
        series_interest_rate="TPM",
        series_unemployment="TASA_DESOCUPACION",
        series_currency="TCO_DOLAR",
    ),

    # --- Latin America (additional) ---
    "mx_banxico": MacroAPIInfo(
        macro_id="mx_banxico",
        country="Mexico",
        country_code="MX",
        region="Latin America",
        api_name="Banxico (Banco de Mexico)",
        api_url="https://www.banxico.org.mx/SieAPIRest/",
        requires_api_key=True,  # free registration at banxico.org.mx
        description="Banxico central bank: interest rate, inflation, FX; wbgapi fallback for GDP/unemployment",
        series_gdp="",           # GDP from wbgapi fallback
        series_inflation="SP74665",
        series_interest_rate="SF61745",
        series_unemployment="",  # unemployment from wbgapi fallback
        series_currency="SF63528",
    ),
}

# Map country_code -> macro_id for quick lookup from market selection
_COUNTRY_TO_MACRO: dict[str, str] = {
    api.country_code: api.macro_id
    for api in MACRO_APIS.values()
}
# EU markets (FR, DE) should also map to eu_ecb as a fallback
_COUNTRY_TO_MACRO.setdefault("FR", "eu_ecb")
_COUNTRY_TO_MACRO.setdefault("DE", "eu_ecb")


def get_macro_api_for_market(market_id: str) -> MacroAPIInfo | None:
    """Return the macro API that corresponds to a given market.

    Looks up by country_code, falling back to the EU ECB for
    European markets without a country-specific macro source.
    """
    market = MARKETS.get(market_id)
    if market is None:
        return None

    # Direct country match first
    cc = market.country_code
    macro_id = _COUNTRY_TO_MACRO.get(cc)

    # Fallback: EU markets -> ECB
    if macro_id is None and market.region == "Europe":
        macro_id = "eu_ecb"

    if macro_id is None:
        return None

    return MACRO_APIS.get(macro_id)


def get_macro_api(macro_id: str) -> MacroAPIInfo | None:
    """Look up a macro API by its unique ID."""
    return MACRO_APIS.get(macro_id)


def get_all_macro_apis() -> list[MacroAPIInfo]:
    """Return all registered macro APIs."""
    return sorted(MACRO_APIS.values(), key=lambda m: (m.region, m.country))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_regions() -> list[str]:
    """Return a sorted list of unique region names."""
    regions = sorted({m.region for m in MARKETS.values()})
    return regions


def get_markets_by_region(region: str) -> list[MarketInfo]:
    """Return all markets belonging to *region*, sorted by market_id."""
    return sorted(
        [m for m in MARKETS.values() if m.region.lower() == region.lower()],
        key=lambda m: m.market_id,
    )


def get_market(market_id: str) -> MarketInfo | None:
    """Look up a market by its unique ID."""
    return MARKETS.get(market_id)


def get_all_markets() -> list[MarketInfo]:
    """Return all registered markets, sorted by tier then market cap desc."""
    return sorted(MARKETS.values(), key=lambda m: (m.tier, m.market_id))


def get_tier1_markets() -> list[MarketInfo]:
    """Return only Tier 1 (launch-ready) markets."""
    return [m for m in get_all_markets() if m.tier == 1]


def get_market_summary_for_llm() -> str:
    """Build compact market list for LLM entity discovery prompt.

    Returns a string listing all available markets with their market_id,
    country, and exchange.  Injected into the entity discovery prompt so
    the LLM can route entities to the correct PIT wrapper in a single
    call (no brute-force cross-region search needed).
    """
    lines = []
    for m in sorted(MARKETS.values(), key=lambda x: (x.tier, x.market_id)):
        lines.append(f"  {m.market_id}: {m.country} ({m.exchange})")
    return "\n".join(lines)


def format_region_menu() -> str:
    """Build a human-readable menu of regions for CLI display."""
    lines = ["", "Available Regions:", ""]
    for idx, region in enumerate(get_regions(), 1):
        markets = get_markets_by_region(region)
        total_markets = len(markets)
        lines.append(f"  {idx}. {region} ({total_markets} market{'s' if total_markets != 1 else ''})")
        for m in markets:
            lines.append(f"       - {m.country} ({m.exchange}) -- {m.pit_api_name}")
    lines.append("")
    return "\n".join(lines)


def format_market_menu(region: str) -> str:
    """Build a human-readable menu of markets within a region."""
    markets = get_markets_by_region(region)
    if not markets:
        return f"\nNo markets available for region: {region}\n"

    lines = ["", f"Markets in {region}:", ""]
    for idx, m in enumerate(markets, 1):
        key_note = " (API key required)" if m.requires_api_key else " (no key needed)"
        lines.append(
            f"  {idx}. {m.country} -- {m.exchange}"
            f"\n       Data source: {m.pit_api_name}{key_note}"
            f"\n       Market cap: {m.market_cap}"
        )
    lines.append("")
    return "\n".join(lines)
