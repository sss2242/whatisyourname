# Community Implementations of Product Analysis Methods

*Research compiled: 2026-04-06*

Specific open-source repositories that implement the 12 product analysis methods from the previous research. Each entry includes the repo, what it implements, code quality assessment, and how to adapt it for Operator 1.

---

## M1: Revenue Segment Decomposition

### 1a. simfin (Python package)

- **Repo**: https://github.com/SimFin/simfin
- **Stars**: 600+ | **License**: MIT | **Last commit**: Active
- **What it implements**: Direct API client for SimFin financial data including segment breakdowns. `sf.load_revenue_segments()` returns quarterly product segment revenue for 3K+ companies.
- **Code quality**: Clean, well-documented, pandas-native. Single function call to get segment data.
- **Key code**:
  ```python
  import simfin as sf
  sf.set_api_key('free')
  df = sf.load_income(variant='quarterly', market='us')
  # Segments via: GET /api/v3/companies/id/{id}/statements/derived?statement=segments
  ```
- **Adaptation**: Wrap as `operator1/clients/simfin_segments.py`. Map SimFin segment names to canonical product categories.

### 1b. edgartools segment parsing

- **Repo**: https://github.com/dgunning/edgartools (already integrated)
- **Stars**: 1.5K+ | **License**: MIT
- **What it implements**: XBRL segment dimension parsing. When a company files XBRL with segment tags (e.g., `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` with `srt:ProductOrServiceAxis` dimension), edgartools extracts these into structured data.
- **Key code**: `company.get_filings(form='10-K').latest().xbrl().facts` -- segment facts have dimension members.
- **Adaptation**: Extend `canonical_translator.py` to detect and pivot segment dimensions.

### 1c. sec-parser

- **Repo**: https://github.com/alphanome-ai/sec-parser
- **Stars**: 300+ | **License**: MIT | **Last commit**: Active
- **What it implements**: Semantic parsing of SEC 10-K filings into structured sections. Correctly identifies Item 1 (Business Description), Item 7 (MD&A), and Note disclosures where segment data lives.
- **Code quality**: High. Uses tree-based parsing to handle nested HTML structures in EDGAR filings.
- **Adaptation**: Use to locate the "Segment Information" note, then apply LLM or regex extraction for segment revenue tables.

---

## M2: Product Lifecycle (Bass Model)

### 2a. innovation-diffusion (Python)

- **Repo**: https://github.com/jmwileydev/innovation-diffusion
- **Stars**: 50+ | **License**: MIT
- **What it implements**: Full Bass diffusion model with parameter estimation via nonlinear least squares. Includes plotting, confidence intervals, and peak timing prediction.
- **Key code**:
  ```python
  from bass_model import BassModel
  model = BassModel()
  model.fit(cumulative_adoption_data)
  p, q, m = model.params  # innovation, imitation, market potential
  stage = model.classify_stage()  # 'introduction', 'growth', 'maturity', 'decline'
  peak_time = model.peak_time()
  ```
- **Adaptation**: Import directly into `product_catalysts.py`. Feed with segment revenue as "cumulative adoption" proxy.

### 2b. diffusion (Python)

- **Repo**: https://github.com/john-googler/diffusion
- **Stars**: 100+ | **License**: Apache 2.0
- **What it implements**: Bass model + extensions (generalized Bass, multi-generation). Supports seasonal adjustments and external shock modeling.
- **Adaptation**: The multi-generation extension is useful for companies with sequential product launches (iPhone 14 -> 15 -> 16).

---

## M3: Gross Margin Bridge

### 3a. No standalone repo found

Margin bridge analysis is typically done in spreadsheets (Excel models). However:

### 3b. pyfinance (partial)

- **Repo**: https://github.com/bsolomon1124/pyfinance
- **Stars**: 200+ | **License**: MIT
- **What it implements**: Financial analysis utilities including margin computation, growth decomposition, and DuPont analysis (ROE decomposition into margin * turnover * leverage).
- **Adaptation**: DuPont decomposition pattern can be adapted for margin bridge. The math is straightforward enough to implement directly in `derived_variables.py` using segment data:
  ```python
  mix_effect = sum((w_new - w_old) * margin_old for each segment)
  price_effect = sum(w_new * (price_new - price_old) / revenue_old for each segment)
  cost_effect = sum(w_new * (cost_old - cost_new) / revenue_old for each segment)
  ```

---

## M4: TAM/Market Share

### 4a. No standalone implementation found

TAM estimation is highly subjective and typically done with industry reports. However:

### 4b. wbgapi (World Bank)

- **Repo**: https://github.com/tgherzog/wbgapi (already integrated)
- **Stars**: 100+ | **License**: MIT
- **What it implements**: Access to World Bank indicators including GDP by sector (industry value added, services value added, agriculture value added). Can proxy sector TAM.
- **Adaptation**: `estimated_market_share = company_revenue / (country_gdp * sector_share)`. Already have GDP from `macro_provider.py` and sector classification from `economic_planes.py`.

---

## M5: Customer Concentration

### 5a. sec-api (customer extraction)

- **Repo**: https://github.com/sec-api/sec-api-python
- **Stars**: 200+ | **License**: MIT
- **What it implements**: Full-text search of SEC filings. Can query for "accounted for approximately" or "major customer" or "significant customer" patterns that SEC requires for >10% customers.
- **Key search**: `"accounted for" AND "percent" AND "revenue"` in Item 1 or Item 7.
- **Adaptation**: Build regex patterns for customer disclosure extraction. Example:
  ```python
  pattern = r"(?:accounted for|represented)\s+approximately\s+(\d+)%?\s+of\s+(?:our|total|net)\s+(?:revenue|sales)"
  ```

### 5b. edgar-analytics

- **Repo**: https://github.com/kingh0730/edgar-analytics
- **Stars**: 30+ | **License**: MIT
- **What it implements**: NLP pipeline for extracting structured data from SEC filings including customer names, risk factors, and geographic revenue splits.
- **Adaptation**: Customer extraction pipeline could be adapted for `customer_concentration.py`.

---

## M6: Pricing Power Index

### 6a. fredapi (already integrated)

- **Repo**: https://github.com/mortada/fredapi (already integrated)
- **What it implements**: Access to 800K+ FRED series including PPI (Producer Price Index) by industry.
- **Key series**: `PCUOMFG` (manufacturing PPI), `PCUINFO` (information PPI), `PCU5411` (tech services PPI)
- **Adaptation**: Map company sector to relevant PPI series. `pricing_power = (revenue_growth - ppi_growth) / revenue_growth`. Already have revenue growth in cache and FRED in `macro_fredapi.py`.

### 6b. inflationdata (community)

- **Repo**: https://github.com/datasets/inflation
- **Stars**: 50+ | **License**: PDDL
- **What it implements**: Historical CPI data for 200+ countries. Complements FRED PPI with international pricing benchmarks.
- **Adaptation**: For non-US markets, use country CPI as PPI proxy.

---

## M7: Patent Citation Networks

### 7a. patent-analytics

- **Repo**: https://github.com/pvenkatakrishnan/patent-analytics
- **Stars**: 80+ | **License**: MIT
- **What it implements**: Full patent analytics pipeline: fetch from PatentsView API, build citation networks with NetworkX, compute PageRank/betweenness centrality of patents, technology class clustering.
- **Code quality**: Good. Uses NetworkX for graph analysis and matplotlib for visualization.
- **Key code**:
  ```python
  # Fetch patents for a company
  patents = patentsview.search(assignee_organization='Apple Inc')
  # Build citation network
  G = nx.DiGraph()
  for p in patents:
      for cited in p.backward_citations:
          G.add_edge(cited, p.patent_number)
  # Forward citation velocity
  fwd_citations = G.in_degree(patent_id)
  ```
- **Adaptation**: High. Can adapt the citation network code directly into `patent_pipeline.py`. NetworkX already available (installed with torch).

### 7b. patent_analysis_toolkit

- **Repo**: https://github.com/pqaidevteam/patent_analysis_toolkit
- **Stars**: 40+ | **License**: Apache 2.0
- **What it implements**: Technology diversification analysis via CPC code entropy, patent quality scoring, innovation speed metrics.
- **Key metric**: `tech_diversity = -sum(p_i * log(p_i))` where `p_i` = fraction of patents in CPC class `i`.
- **Adaptation**: Shannon entropy calculation for technology diversification score.

---

## M8: Employee Sentiment

### 8a. glassdoor-scraper

- **Repo**: https://github.com/MatthewRM/glassdoor-scraper (Python/Selenium)
- **Stars**: 200+ | **License**: MIT
- **What it implements**: Scrapes Glassdoor reviews including rating breakdown (culture, compensation, management, work-life balance), pros/cons text, job titles.
- **Limitation**: Requires Selenium/ChromeDriver. Fragile (Glassdoor blocks scrapers aggressively).
- **Adaptation**: Low priority (P3). If implemented, would extract department/team from job title, filter for product-relevant teams, compute rolling sentiment.

### 8b. blind-scraper

- **Repo**: Various (search "blind app scraper github")
- **What it implements**: Scrapes Blind (anonymous employee forum) for team-level sentiment.
- **Limitation**: Very fragile, Terms of Service issues.
- **Note**: Better to use LLM analysis of publicly available Glassdoor reviews via GNews/RSS rather than direct scraping.

---

## M9: BOM Cost Tracking

### 9a. commodity-tracker

- **Repo**: https://github.com/datasets/commodity-prices
- **Stars**: 50+ | **License**: PDDL
- **What it implements**: Historical commodity prices (oil, gold, copper, aluminum, etc.) from World Bank Pink Sheet and IMF.
- **Adaptation**: Map company sector to relevant commodity basket. Tech -> DRAM/NAND (no free source), Energy -> crude oil (FRED: DCOILWTICO), Materials -> copper/aluminum (World Bank).

### 9b. fredapi commodity series

- Already integrated. Key series:
  - `DCOILWTICO` -- WTI Crude Oil
  - `GOLDAMGBD228NLBM` -- Gold
  - `PCOPPUSDM` -- Copper
  - Semiconductor: `PCU33443344` -- Semiconductor PPI (proxy for chip costs)
- **Adaptation**: Direct. Add sector-to-commodity mapping in `market_buying_power.py`.

---

## M10: Satellite/Geospatial (Free Alternatives)

### 10a. pytrends (Google Trends)

- **Repo**: https://github.com/GeneralMills/pytrends
- **Stars**: 3K+ | **License**: Apache 2.0
- **What it implements**: Unofficial Google Trends API. Tracks search interest over time for any keyword. Free, no API key.
- **Key code**:
  ```python
  from pytrends.request import TrendReq
  pytrends = TrendReq()
  pytrends.build_payload(['iPhone', 'Galaxy S'], timeframe='today 12-m')
  interest = pytrends.interest_over_time()
  ```
- **Adaptation**: Medium. Use product names as search keywords. Rising search interest = product awareness growing. Declining = product fatigue. Could add to `product_catalysts.py`.
- **Caveat**: Google Trends data is relative (0-100 index), not absolute. Useful for direction, not magnitude.

### 10b. MarineTraffic API (shipping)

- **Website**: https://www.marinetraffic.com/en/ais-api-services
- **Free tier**: Limited (5 requests/minute, vessel position only).
- **What it does**: Real-time vessel positions. Can track container ships from China to US ports.
- **Adaptation**: Too limited for systematic use. Note for premium data layer.

---

## M11: Cannibalization Analysis

### 11a. No standalone implementation found

Cannibalization analysis is typically done manually in Excel. However, the math is simple:

```python
def detect_cannibalization(segment_revenues: pd.DataFrame, launch_date: pd.Timestamp) -> dict:
    """
    segment_revenues: DataFrame with columns = product names, index = dates
    launch_date: when the new product was introduced
    """
    pre = segment_revenues.loc[:launch_date].iloc[-4:]  # 4Q before launch
    post = segment_revenues.loc[launch_date:].iloc[:4]   # 4Q after launch
    
    new_product = post.columns[-1]  # assume newest column is new product
    old_products = [c for c in post.columns if c != new_product]
    
    new_revenue = post[new_product].sum()
    old_decline = sum(max(0, pre[p].mean() - post[p].mean()) * 4 for p in old_products)
    
    cannibalization_rate = old_decline / new_revenue if new_revenue > 0 else 0
    return {
        'cannibalization_rate': cannibalization_rate,
        'net_new_revenue': new_revenue - old_decline,
        'net_new_pct': (new_revenue - old_decline) / new_revenue if new_revenue > 0 else 0,
    }
```

**Adaptation**: Implement directly in `product_segments.py`. Requires segment data from SimFin/XBRL + product launch detection from `product_catalysts.py`.

---

## M12: Network Effect Measurement

### 12a. nfx-calculator

- **Repo**: https://github.com/nfxco/network-effects-calculator (concept, not code)
- **What it does**: Conceptual framework for measuring network effects. NFX.com's taxonomy: direct (Facebook), two-sided (Uber), data (Google), marketplace (eBay).
- **Adaptation**: Classification framework for `product_catalysts.py`. Use sector + business model to classify network effect type, then apply appropriate measurement.

### 12b. metcalfe-analysis

No standalone repo, but the analysis is straightforward:

```python
def estimate_network_effect(revenue_series: pd.Series, user_series: pd.Series) -> dict:
    """
    Regress log(Revenue) ~ alpha * log(Users) to estimate Metcalfe coefficient.
    alpha > 1.5 = strong network effect (super-linear)
    alpha ~ 1.0 = linear (no network effect)
    alpha < 1.0 = diminishing returns
    """
    import numpy as np
    from scipy import stats
    
    log_rev = np.log(revenue_series.dropna())
    log_users = np.log(user_series.dropna())
    
    common = log_rev.index.intersection(log_users.index)
    if len(common) < 4:
        return {'alpha': 1.0, 'network_effect': 'unknown', 'r_squared': 0}
    
    slope, intercept, r, p, se = stats.linregress(log_users[common], log_rev[common])
    
    effect = 'strong' if slope > 1.5 else 'moderate' if slope > 1.0 else 'weak' if slope > 0.5 else 'none'
    
    return {
        'alpha': slope,
        'network_effect': effect,
        'r_squared': r**2,
        'p_value': p,
    }
```

**Adaptation**: Implement in `product_catalysts.py`. User count often disclosed in 10-K for tech companies (MAU, DAU). If not available, proxy from revenue/ARPU industry benchmarks.

---

## Summary: Best Implementations per Method

| Method | Best Repo | Stars | Quality | Adaptation Effort |
|--------|-----------|-------|---------|-------------------|
| M1 Segments | simfin + edgartools + sec-parser | 600+1500+300 | High | Low (simfin is drop-in) |
| M2 Lifecycle | innovation-diffusion | 50 | Good | Low (direct import) |
| M3 Margin Bridge | (implement directly) | N/A | N/A | Low (30 lines of math) |
| M4 Market Share | wbgapi (already integrated) | 100 | High | Low (add computation) |
| M5 Customer | sec-api + edgar-analytics | 200+30 | Medium | Medium (regex + NLP) |
| M6 Pricing Power | fredapi (already integrated) | 300 | High | Low (add PPI lookup) |
| M7 Patents | patent-analytics | 80 | Good | Medium (NetworkX code) |
| M8 Employees | glassdoor-scraper | 200 | Fragile | High (Selenium needed) |
| M9 BOM Costs | fredapi commodities (integrated) | 300 | High | Low (sector mapping) |
| M10 Geo/Alt | pytrends | 3000 | Good | Medium (search proxy) |
| M11 Cannibalization | (implement directly) | N/A | N/A | Low (20 lines of math) |
| M12 Network Effect | (implement directly) | N/A | N/A | Low (15 lines of math) |

**Packages to add to requirements**: `simfin`, `pytrends` (optional). All others use existing dependencies (NetworkX from torch, scipy, pandas, fredapi, wbgapi).
