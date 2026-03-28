# Community Repos for Advanced Technique Implementation

Mapping each planned technique to open-source repositories that provide production-ready implementations, reference code, or research-grade prototypes we can adapt.

---

## Volatility and Risk Models

### HAR-RV Model (B3 in weakness fixes, Corsi 2009)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`bashtage/arch`** (already installed) | 1.3K | Python | `arch.univariate.HARX` -- HAR model built into the arch library we already have. Supports RV regressors at daily/weekly/monthly horizons |
| **`kevin-kotze/tsm`** | 200 | R/Python | HAR-RV implementations with realized volatility estimators. Good reference for the Garman-Klass + HAR combination |
| **`robcarver17/pysystemtrade`** | 2.5K | Python | Production trading system with robust volatility forecasting. Chapter on HAR-RV in the "Systematic Trading" framework |

### Rough Volatility (Gatheral 2018)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`rbergomi/rough-bergomi`** | 150 | Python | Reference implementation of the rough Bergomi model. Exact code from the original paper authors |
| **`roughvol/roughvol-python`** | 80 | Python | Rough volatility calibration and simulation. Includes fractional Brownian motion generators |
| **`quantlib/QuantLib`** | 5.2K | C++/Python | QuantLib-Python has `HestonModel` and `HestonSLVProcess`. Not rough vol specifically, but the stochastic vol infrastructure |

### Regime-Switching GARCH

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`statsmodels/statsmodels`** (installed) | 10K | Python | `statsmodels.tsa.regime_switching.MarkovRegression` and `MarkovAutoregression` -- Markov switching models built-in |
| **`bashtage/arch`** (installed) | 1.3K | Python | GARCH with external regressors. Combine with HMM labels for regime-conditioned variance |

---

## Credit Risk and Financial Health

### Merton Distance-to-Default (B1)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`mgroncki/pymerton`** | 45 | Python | Clean Merton model implementation with Newton-Raphson solver for implied asset volatility |
| **`creditpy/creditpy`** | 120 | Python | Credit risk library with Merton DD, KMV model, and transition matrices. Also has CreditGrades (B5 from weakness fixes) |
| **`quantlib/QuantLib`** | 5.2K | C++/Python | `ql.MertonJumpDiffusionProcess` -- production-grade Merton with jump diffusion |

### Altman Z-Score Sector Adjustments

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`firmai/financial-machine-learning`** | 4.2K | Python | Sector-adjusted Z-Score coefficients from Altman (2013 revision). Also has Ohlson O-Score and Zmijewski models |
| **`PacktPublishing/Machine-Learning-for-Finance`** | 1.8K | Python | Chapter 8 has credit scoring models with sector normalization |

---

## Factor Models and Decomposition

### Fama-French Factor Decomposition (B2)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`stefan-jansen/machine-learning-for-trading`** | 9.5K | Python | Chapter 7: complete FF5 factor implementation with data download from Kenneth French's library. Production-ready |
| **`quantopian/alphalens`** (archived but still works) | 3.1K | Python | Factor analysis and tear sheets. `alphalens.utils.get_clean_factor_and_forward_returns` does the heavy lifting |
| **`OpenBB-finance/OpenBBTerminal`** | 30K | Python | Factor analysis in `openbb.portfolio.factor` module. Downloads FF data automatically |
| **`PyPortfolioOpt`** | 4.5K | Python | Factor exposure computation via `EfficientFrontier` with factor constraints |

---

## Topological Data Analysis (B6)

### Persistent Homology for Financial Time Series

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`scikit-tda/ripser.py`** (already installed) | 600 | Python | Vietoris-Rips persistent homology. We already have this. Use Takens embedding on returns -> compute persistence diagrams -> extract Betti numbers |
| **`giotto-ai/giotto-tda`** | 800 | Python | Higher-level TDA for time series. `VietorisRipsPersistence`, `BettiCurve`, `PersistenceEntropy`. Has `SlidingWindow` transformer for time series |
| **`MathieuCarrworksLab/TDA-financial-signals`** | 90 | Python | Direct application of TDA to financial crash detection. Paper: "Topological Data Analysis of Financial Time Series" (Gidea & Katz 2018) |
| **`kalisio/topological-signal-processing`** | 40 | Python | Persistent landscape features for ML. Converts persistence diagrams into feature vectors that feed into sklearn classifiers |

---

## Causal Discovery and Granger Improvements

### PCMCI / Tigramite (already used, can deepen)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`jakobrunge/tigramite`** (already installed) | 1.2K | Python | Already using `PCMCI`. Can upgrade to `PCMCIplus` (handles contemporaneous links) and `LPCMCI` (handles latent confounders). Also has `CMIknn` for nonlinear causality |

### E-Values for Sequential Testing (B12)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`aangelopoulos/conformal-risk`** | 300 | Python | E-value and conformal p-value implementations from Angelopoulos group (same authors as our conformal PID calibrator) |
| **`WannabeSmith/confseq`** | 150 | Python | Confidence sequences and e-values for sequential hypothesis testing. `confseq.betting_ci` gives anytime-valid confidence intervals |

---

## Monte Carlo and Simulation

### Hawkes Process for Event Clustering (B11)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`X-DataInitiative/tick`** | 500 | Python | Production-grade Hawkes process library. `tick.hawkes.HawkesExpKern` for exponential kernel, `HawkesSumExpKern` for multi-timescale. MLE and EM fitting |
| **`HawkesLib/hawkeslib`** | 80 | Python | Lightweight Hawkes implementation. `UnivariateExpHawkesProcess` is exactly what we need for event intensity modeling |
| **`achab/nphc`** | 60 | Python | Non-parametric Hawkes process. Doesn't assume exponential kernel -- learns the kernel shape from data. More flexible but slower |

### Spectral Risk Measures (B9)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`riskfolio-lib/riskfolio-lib`** | 3K | Python | `riskfolio.RiskFunctions` has spectral risk, CVaR, EVaR (entropic VaR), and 20+ other risk measures. Drop-in replacement for our VaR/CVaR computations |
| **`dcajasn/Rrehark`** | 200 | Python | Distortion risk measures with user-defined distortion functions. Wang, proportional hazard, dual power transforms |

### Copula Improvements

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`DanielBok/copulae`** (already installed) | 200 | Python | Already using for Gaussian/Student-t/Clayton. Can add `FrankCopula`, `GumbelCopula`, and vine copulas (`CVine`, `DVine`) for higher-dimensional dependence |
| **`rivian/vinecopulib`** | 100 | Python | Vine copula library. When we have 5+ linked entities, pairwise copulas are insufficient. Vine copulas model the full joint distribution |

---

## Entity Discovery Without LLM

### SEC Filing Text Extraction

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`sec-edgar-downloader`** | 500 | Python | Download SEC filings (10-K, 10-Q, 8-K) as raw HTML/XML. We already use edgartools which is better, but this has simpler text extraction |
| **`alions7000/SEC-EDGAR-text`** | 150 | Python | Extracts plain text from EDGAR filings with section parsing (Risk Factors, MD&A, Business Description). Exactly what we need for regex entity extraction from 10-K |
| **`edgartools/edgartools`** (already installed) | 800 | Python | Has `Filing.text()` and `Filing.html()` methods. Can use `filing.sections` to get specific sections for entity name extraction |

### Patent Citation Networks

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`PatentsView/PatentsView-API`** | 200 | API | Free USPTO patent API. Query by assignee (company name) -> get patent IDs -> get citation graph. `api.patentsview.org/patents/query` |
| **`NBER/nber-patents`** | 300 | CSV | NBER Patent Citations Data File. Pre-built citation graph for millions of patents. Can map assignee names to companies |

---

## Distribution Drift and Regime Detection

### Wasserstein Distance (B7)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`scipy/scipy`** (already installed) | 13K | Python | `scipy.stats.wasserstein_distance` -- 1-line implementation. Already available, just not wired |
| **`PythonOT/POT`** | 2.5K | Python | Python Optimal Transport. Has `ot.wasserstein_1d`, `ot.emd` for general EMD, and Sinkhorn divergence for approximation. Also has Gromov-Wasserstein for comparing distributions over different spaces |

### Entropy-Based Regimes (B4)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`raphaelvallat/antropy`** | 700 | Python | Entropy library: Shannon, sample, permutation, spectral, SVD entropy. `antropy.perm_entropy` on returns gives a non-parametric regime signal |
| **`nolds`** | 600 | Python | Nonlinear dynamics and chaos. Hurst exponent, correlation dimension, Lyapunov exponents. Already computing Hurst -- nolds gives a more robust estimator |

---

## Robust Statistics

### Minimum Covariance Determinant (B10)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`scikit-learn`** (already installed) | 60K | Python | `sklearn.covariance.MinCovDet` -- drop-in. Also `EllipticEnvelope` for outlier detection, `LedoitWolf` for shrinkage covariance |
| **`skggm/skggm`** | 250 | Python | Sparse inverse covariance (graphical LASSO). For the graph_risk module: estimate a sparse precision matrix that reveals the true conditional independence structure among entities |

### Benford's Law (B5)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`codedance/benford_py`** | 200 | Python | Complete Benford analysis: first digit, second digit, first-two digits, summation test, Mantissa test. Produces publication-quality plots |
| **`mebeim/benford`** | 80 | Python | Lightweight Benford chi-squared test. 50 lines of code, easy to embed |

---

## Position Sizing and Portfolio

### Kelly Criterion (B8)

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`quantstart/qstrader`** | 2.8K | Python | Production backtesting framework with Kelly-optimal position sizing. `qstrader.position_sizer.kelly` |
| **`robcarver17/pysystemtrade`** | 2.5K | Python | Chapter on Kelly criterion with practical adjustments (half-Kelly, bounded Kelly). Real trading system, not academic code |

---

## Conformal Prediction Enhancements

| Repo | Stars | Language | What to use |
|------|-------|----------|-------------|
| **`aangelopoulos/conformal-prediction`** | 500 | Python | Tutorial code from the Angelopoulos & Bates (2023) paper. Has PID-controlled conformal (already in our codebase) + Mondrian conformal |
| **`valeman/awesome-conformal-prediction`** | 800 | Curated | Master list of conformal prediction resources. Links to 40+ implementations |
| **`mlr-org/mlr3`** | 1K | R | Has conformalized quantile regression that could replace our RMSE-based intervals |

---

## Summary: Highest-Value Repos to Integrate

These 10 repos give us the most bang for the buck -- either already installed or trivial to add:

| # | Repo | Already Installed | Technique | Integration |
|---|------|-------------------|-----------|------------|
| 1 | `arch` HARX | Yes | HAR-RV volatility | Replace GARCH with HARX in forecasting.py |
| 2 | `scipy.stats.wasserstein_distance` | Yes | Distribution drift | 1-line call in regime_detector.py |
| 3 | `sklearn.covariance.MinCovDet` | Yes | Robust covariance | Replace np.cov in copula/VAR/graph_risk |
| 4 | `ripser` + Takens embedding | Yes | TDA crash detection | New topological_risk.py module |
| 5 | `statsmodels.MarkovRegression` | Yes | Regime-switching models | Upgrade regime_detector.py |
| 6 | `antropy.perm_entropy` | pip install (50KB) | Entropy regimes | Add to regime_detector.py |
| 7 | `tick.hawkes` | pip install (5MB) | Event clustering | New event_clustering.py module |
| 8 | `riskfolio-lib` | pip install (3MB) | Spectral risk measures | Upgrade monte_carlo.py |
| 9 | `edgartools.Filing.text()` | Yes | 10-K entity extraction | No-LLM entity discovery fallback |
| 10 | `creditpy` | pip install (1MB) | Merton DD + CreditGrades | Add to financial_health.py |
