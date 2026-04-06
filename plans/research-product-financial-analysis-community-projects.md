# Community Projects: Financial Product Analysis of Companies

*Research compiled: 2026-04-06*

Research into open-source and community projects that analyze a company's **products** from a financial perspective -- revenue per product line, product pipeline valuation, product market share, product lifecycle stage, and product-level risk assessment.

---

## Category 1: SEC Filing Product Segment Extraction

These projects extract product-level financial data from regulatory filings (10-K, 20-F) where companies are required to disclose segment revenue.

### 1.1 sec-edgar-downloader + NLP Pipeline

- **Repo**: https://github.com/jadchaar/sec-edgar-downloader
- **What it does**: Downloads any SEC filing type (10-K, 10-Q, 8-K, etc.) as raw text/HTML. Community pipelines layer NLP on top to extract "Operating Segments" and "Revenue by Product" tables from Item 7 (MD&A) and Note disclosures.
- **Product relevance**: FASB ASC 280 requires public companies to disclose revenue by reportable segment. This is where Apple breaks out iPhone vs Mac vs Services vs Wearables revenue.
- **Integration potential**: Could feed into Operator 1's product_catalysts module to get actual product revenue splits instead of proxying from aggregate R&D/SGA ratios.
- **License**: MIT
- **Status**: Active, 1.2K+ stars

### 1.2 EDGAR Full-Text Search (EFTS)

- **API**: https://efts.sec.gov/LATEST
- **What it does**: SEC's own full-text search across all filings. Query for product names, brand names, or segment labels across any company's filings.
- **Product relevance**: Search for "iPhone revenue" or "AWS segment" across all filings to build cross-company product revenue time series.
- **Integration potential**: Already used by Operator 1's us_edgar.py for company search. Could extend to product-level search.
- **License**: Public domain (US government data)

### 1.3 edgartools (Python)

- **Repo**: https://github.com/dgunning/edgartools
- **What it does**: Structured XBRL parsing of SEC filings. Already extracts segment data when available in XBRL format.
- **Product relevance**: XBRL tags like `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax` with segment dimensions contain product-level revenue. edgartools v5+ parses these dimensions.
- **Integration potential**: Already integrated in Operator 1. The segment dimension parsing could be extended in the canonical_translator to extract product-level data.
- **License**: MIT
- **Status**: Active, Operator 1 already uses v5.19.1

---

## Category 2: Financial Data Platforms with Product Breakdowns

### 2.1 SimFin (Simplified Finance)

- **Website**: https://simfin.com
- **Repo**: https://github.com/SimFin/simfin
- **What it does**: Free financial data including **segment-level revenue breakdowns** for 3,000+ US companies. Provides quarterly product segment data extracted from SEC filings.
- **Product relevance**: Direct product revenue time series (e.g., Apple iPhone revenue Q1 2020 - Q4 2024).
- **Data fields**: Segment name, segment revenue, segment operating income, segment assets.
- **API**: Free tier (2,000 API calls/day). Bulk CSV download available.
- **Integration potential**: High. Could provide the product-level revenue data that product_catalysts.py currently proxies from aggregate R&D patterns. A new `simfin_product_segments.py` client could fetch quarterly product revenue splits.
- **License**: SimFin API Terms (free for non-commercial, commercial license available)
- **Status**: Active

### 2.2 Financial Modeling Prep (FMP) -- Revenue by Segment

- **Website**: https://financialmodelingprep.com
- **API**: `GET /api/v3/revenue-product-segmentation/{symbol}`
- **What it does**: Returns product-level revenue breakdown for US companies. Example: Apple shows iPhone, Mac, iPad, Wearables, Services separately.
- **Product relevance**: Direct product revenue splits with historical data.
- **Pricing**: Free tier (250 requests/day). Note: Operator 1 removed FMP dependency in earlier refactor, but the product segmentation endpoint is unique.
- **Integration potential**: Medium. Would require re-adding FMP as an optional data source specifically for product segmentation.

### 2.3 Wisesheets / StockAnalysis.com

- **Website**: https://stockanalysis.com/stocks/aapl/revenue/
- **What it does**: Free product revenue breakdown visualization. Shows historical product segment revenue with growth rates.
- **Product relevance**: Good reference data but no API. Would need scraping.
- **Integration potential**: Low (no API).

---

## Category 3: Patent & R&D Pipeline Analysis

These projects analyze a company's product pipeline by examining patent filings, R&D spending patterns, and clinical trial data (pharma).

### 3.1 PatentsView (USPTO)

- **Website**: https://patentsview.org
- **API**: https://patentsview.org/apis/api-endpoints
- **What it does**: Free API to all 12M+ US patents. Query by assignee (company), CPC class (technology area), filing date, grant date. Returns patent counts, citation networks, inventor teams.
- **Product relevance**: Patent filing acceleration in a CPC class signals upcoming product launches. A spike in "G06F" (computing) patents from Apple signals new product features.
- **Metrics**: Patent count trend, citation velocity (how fast new patents cite prior art), technology diversification (Shannon entropy of CPC classes).
- **Integration potential**: High. A `patent_pipeline.py` feature module could compute: (1) Patent acceleration score (are they filing faster?), (2) Technology diversification (new product categories?), (3) Citation impact (are their innovations being built upon?).
- **License**: Public domain
- **Status**: Active, maintained by USPTO

### 3.2 Google Patents Public Datasets (BigQuery)

- **Dataset**: `patents-public-data` on Google BigQuery
- **What it does**: 120M+ patent publications worldwide with full text, CPC codes, citations, and BERT-based embeddings.
- **Product relevance**: Track R&D pipeline across geographies. Japanese pharmaceutical patents signal pipeline products 5-10 years before launch.
- **Integration potential**: Medium. Requires BigQuery access (free 1TB/month). More useful for deep product pipeline analysis than quick screening.

### 3.3 ClinicalTrials.gov API (Pharma/Biotech)

- **API**: https://clinicaltrials.gov/api/v2/studies
- **What it does**: All 400K+ clinical trials. Query by sponsor (company), condition, phase, status.
- **Product relevance**: For pharma companies, the clinical trial pipeline IS the product pipeline. Phase 3 success rates and trial timelines directly impact revenue forecasts.
- **Metrics**: Pipeline value (sum of probability-weighted peak revenue by phase), Time-to-market, Competitive landscape (how many trials targeting same condition).
- **Integration potential**: High for healthcare sector. A `clinical_pipeline.py` module could compute pipeline value estimates for pharma companies.
- **License**: Public domain
- **Status**: Active, v2 API launched 2024

---

## Category 4: Alternative Data for Product Performance

### 4.1 SimilarWeb (Community Scrapers)

- **Projects**: Various GitHub scrapers for SimilarWeb traffic data
- **What it does**: Website traffic estimates for any domain. Monthly visits, bounce rate, pages/visit, traffic sources.
- **Product relevance**: For digital products, web traffic IS product adoption. A decline in traffic signals product deterioration before it shows in quarterly revenue.
- **Integration potential**: Medium. Useful for technology/e-commerce companies. Traffic trend as a leading indicator of revenue.
- **Note**: SimilarWeb official API is paid ($). Community scrapers may break.

### 4.2 App Store Analytics (appfigures, data.ai alternatives)

- **Open alternatives**: AppTweak has limited free tier. Google Play public data can be scraped.
- **What it does**: App download estimates, ratings trends, review sentiment.
- **Product relevance**: For mobile-first companies (Meta, Snap, Uber), app download trends lead revenue by 1-2 quarters.
- **Integration potential**: Low-Medium. Would require per-company app identification.

### 4.3 Steam/Epic Games Store (Gaming)

- **Projects**: SteamSpy (https://steamspy.com), SteamDB (https://steamdb.info)
- **What it does**: Player count estimates, concurrent users, ownership estimates for PC games.
- **Product relevance**: For gaming companies (EA, Activision, Valve), player engagement directly predicts DLC/microtransaction revenue.
- **Integration potential**: Niche (gaming sector only).

### 4.4 Glassdoor/Indeed Job Postings (Product Team Signals)

- **Projects**: Various job posting scrapers on GitHub
- **What it does**: Track hiring patterns by role type and product area.
- **Product relevance**: A surge in hiring for "AR/VR engineers" at Apple signals investment in Vision Pro product line. A reduction in hiring for a product team signals deprioritization.
- **Integration potential**: Medium. Hiring velocity by product area as a forward-looking investment signal.

---

## Category 5: Product Sentiment & Review Analysis

### 5.1 Amazon Product Reviews (academic datasets)

- **Dataset**: Amazon Review Data (2023) -- McAuley lab
- **What it does**: 570M+ product reviews across all Amazon categories with ratings, review text, and product metadata.
- **Product relevance**: For consumer product companies, review sentiment trends predict future sales. Rating decline signals product quality issues.
- **Integration potential**: Low-Medium. Very useful for consumer goods companies but requires matching company products to Amazon ASINs.

### 5.2 Trustpilot/G2/Capterra Reviews

- **Projects**: Various review scraping tools
- **What it does**: B2B software product reviews with detailed feature ratings.
- **Product relevance**: For SaaS companies, G2 review trends (NPS, feature satisfaction) lead churn by 2-3 quarters.
- **Integration potential**: Medium for technology/SaaS sector.

---

## Category 6: Integrated Financial Product Analysis Platforms

### 6.1 OpenBB Terminal

- **Repo**: https://github.com/OpenBB-finance/OpenBBTerminal
- **What it does**: Open-source Bloomberg terminal alternative. 30+ data sources including product segment data from FMP, revenue geographic/product splits, patent data integration.
- **Product relevance**: Has a `/stocks/fa/product` command that shows product revenue breakdown. Also has patent analysis and clinical trial lookup.
- **Integration potential**: Could use OpenBB's data aggregation layer rather than building individual API clients.
- **License**: AGPL-3.0
- **Status**: Active, 35K+ stars, very well maintained

### 6.2 FinRL (Reinforcement Learning for Finance)

- **Repo**: https://github.com/AI4Finance-Foundation/FinRL
- **What it does**: Deep RL framework for trading. Includes data preprocessing that can incorporate product-level features.
- **Product relevance**: Indirectly useful -- can train RL agents on product segment data as features.
- **License**: MIT
- **Status**: Active, 10K+ stars

### 6.3 Stocksera

- **Repo**: https://github.com/guanquann/Stocksera
- **What it does**: Dashboard for alternative data including government contracts, lobbying data, FDA calendar, patent grants -- all product-relevant signals.
- **License**: MIT
- **Status**: Active

---

## Category 7: Product Revenue Forecasting Models

### 7.1 Bass Diffusion Model (Product Lifecycle)

- **Implementations**: Multiple Python packages (`diffusion`, `pyBassModel`)
- **What it does**: Models product adoption using the Bass (1969) innovation diffusion curve: S-shaped adoption driven by innovators (p) and imitators (q).
- **Product relevance**: Estimates where a product is in its lifecycle (introduction, growth, maturity, decline). iPhone is in late maturity; Apple Vision Pro is in early introduction.
- **Integration potential**: High. Could be added to product_catalysts.py to classify each product segment's lifecycle stage and estimate remaining growth runway.
- **Academic basis**: Bass, F.M. (1969). "A New Product Growth for Model Consumer Durables." Management Science.

### 7.2 Product Market Fit Scoring (PMF)

- **Projects**: Various GitHub projects implementing Superhuman's PMF survey methodology
- **What it does**: Scores how well a product meets market needs. "How disappointed would you be if this product no longer existed?" >40% "very disappointed" = PMF.
- **Product relevance**: For private/pre-IPO companies, PMF score is the strongest predictor of future revenue growth.
- **Integration potential**: Low (requires survey data, not available from public filings).

---

## Recommendations for Operator 1 Integration

### Priority 1 (High Impact, Low Effort)
1. **SimFin product segments** -- Direct quarterly product revenue splits for 3K+ US companies. New client `simfin_product_segments.py` in `operator1/clients/`.
2. **PatentsView patent pipeline** -- Patent filing acceleration as a product catalyst signal. Enhance `product_catalysts.py` with patent velocity data.
3. **edgartools segment extraction** -- Already integrated. Extend `canonical_translator.py` to parse XBRL segment dimensions for product-level revenue.

### Priority 2 (Medium Impact, Medium Effort)
4. **ClinicalTrials.gov** -- Pipeline valuation for pharma/biotech. New sector-specific module.
5. **Bass Diffusion Model** -- Product lifecycle classification from segment revenue trajectory. Add to `product_catalysts.py`.
6. **SEC EFTS product search** -- Cross-filing product mention frequency as innovation signal.

### Priority 3 (Niche/Experimental)
7. **SimilarWeb traffic proxies** -- Digital product adoption leading indicator.
8. **Job posting analysis** -- Hiring patterns as product investment signal.
9. **Review sentiment** -- Consumer product quality degradation detection.

---

## Summary Table

| # | Project | Data Type | Free API | Product Revenue | Product Pipeline | Sector Coverage |
|---|---------|-----------|----------|----------------|-----------------|-----------------|
| 1 | SimFin | Segment financials | Yes (2K/day) | Direct quarterly | No | US (3K+ companies) |
| 2 | edgartools | XBRL segments | Yes (SEC) | Direct quarterly | No | US (all SEC filers) |
| 3 | PatentsView | Patent filings | Yes (unlimited) | No | R&D pipeline | Global (US patents) |
| 4 | ClinicalTrials.gov | Clinical trials | Yes (unlimited) | No | Drug pipeline | Global pharma |
| 5 | OpenBB | Aggregated | Yes | Via FMP | Via patents | Global |
| 6 | SEC EFTS | Full-text search | Yes | Mentions | Innovation signals | US (all filers) |
| 7 | FMP | Segment financials | Yes (250/day) | Direct quarterly | No | US (10K+ companies) |
| 8 | Bass Model | Statistical | N/A (local) | Lifecycle forecast | Adoption curve | Any (needs data) |
| 9 | SimilarWeb | Web traffic | Scrapers only | Traffic proxy | Digital adoption | Global (web companies) |
| 10 | Glassdoor/Indeed | Job postings | Scrapers only | No | Team investment | Global |
