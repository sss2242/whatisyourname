# Community Repos and PyPI Packages for All 42 Analysis Models

Comprehensive survey of packages and repos that could enhance each model category beyond conformal prediction and walk-forward (which are already covered in the previous plan).

Organized by pipeline layer and module group.

---

## Layer 1: Feature Engineering

### 1. Derived Variables (`derived_variables.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **tsfresh** | 0.21.1 | Automatic extraction of 800+ time series features (mean, variance, entropy, autocorrelation, wavelet coefficients, etc.) with relevance filtering | Could replace our manual 25 derived variables with 800+ auto-extracted features. `tsfresh.extract_relevant_features()` filters to only statistically significant ones. Risk: feature explosion. |
| **ta** | 0.11.0 | Technical analysis library: 90+ indicators (RSI, MACD, Bollinger Bands, ADX, etc.) | Adds standard technical indicators our derived_variables module currently lacks. RSI and MACD are commonly requested by report readers. |
| **pandas-ta** | 0.3.14 | 130+ technical indicators as pandas extension. `df.ta.rsi()` syntax. | Same as `ta` but nicer pandas integration. Includes Ichimoku Cloud, Keltner Channels, OBV, VWAP. |
| **feature-engine** | 1.8.2 | Feature engineering transformers compatible with sklearn pipelines. Handles outliers, missing data, encoding, discretization. | Could replace our manual `safe_ratio()` and `is_missing_*` flag logic with standardized transforms. |

**Best pick:** `ta` or `pandas-ta` for technical indicators (RSI, MACD, Bollinger Bands). These are the #1 missing features users would notice in reports.

### 2. Conflict Risk (`conflict_risk.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **acled-api** | 1.0.0 | ACLED (Armed Conflict Location & Event Data) API client. More comprehensive than UCDP for recent events. | Alternative to UCDP GED (which now requires auth). ACLED has free academic access. |
| **country-converter** | 1.3.0 | Converts between country name formats, ISO codes, continent regions. | Simplifies our static sanctions/conflict country list management. Handles edge cases (Taiwan, Kosovo). |

**Best pick:** `acled-api` as a replacement for UCDP GED (which returns 401 since 2025).

### 3. News Sentiment (`news_sentiment.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **transformers** | 4.x | Hugging Face transformers. FinBERT, DistilBERT for sentiment classification. | Replace keyword-based fallback with actual NLP sentiment model. FinBERT (`ProsusAI/finbert`) is trained on financial text. |
| **vaderSentiment** | 3.3.2 | Rule-based sentiment analyzer tuned for social media/news. No model download needed. | Better keyword fallback than our custom positive/negative word lists. Handles negation, capitalization, emojis. |
| **textblob** | 0.18.0 | Simple NLP: sentiment polarity and subjectivity. | Lightweight alternative to keyword matching. Not financial-domain-specific. |

**Best pick:** `vaderSentiment` for immediate improvement to keyword fallback (no model download). `transformers` + FinBERT for institutional-grade sentiment (large download though).

### 4. Filing Calendar (`filing_calendar.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **exchange-calendars** | 4.x | Market-specific trading calendars for 50+ exchanges. Knows holidays, half-days, early closes. | Already installed. Could enhance filing staleness detection with market-specific holiday awareness. |

### 5. Linked Aggregates (`linked_aggregates.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **networkx** | 3.x | Community detection algorithms (Louvain, Label Propagation, Girvan-Newman). | Could group linked entities into communities and compute per-community aggregates instead of per-relationship-type. Identifies clusters of co-moving entities. |

### 6. Macro Quadrant (`macro_quadrant.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **pycountry** | 24.6.1 | ISO country/currency/language database. | Maps country codes to full names, currencies, and subdivisions. Better than our hardcoded mappings. |
| **cpi** | 1.1.0 | Consumer Price Index data from BLS. | Direct CPI data for inflation-adjusted calculations in the macro module. US-only but precise. |

---

## Layer 2: Analysis Modules

### 6. Survival Mode (`survival_mode.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **lifelines** | 0.29.0 | Survival analysis: Kaplan-Meier, Cox Proportional Hazards, Weibull AFT, time-varying covariates. | **High value.** Could replace our threshold-based survival flag with a proper survival curve. Cox PH gives hazard ratios for each financial variable. `CoxTimeVaryingFitter` handles our daily-varying ratios. |
| **scikit-survival** | 0.24.0 | Machine learning for survival analysis: Random Survival Forests, Gradient Boosted Survival. | Survival prediction using ML models. Random Survival Forest can learn non-linear survival boundaries instead of hard thresholds. |

**Best pick:** `lifelines` for Cox PH survival curves -- transforms our binary survival flag into a proper time-to-event analysis with interpretable hazard ratios.

### 7. Vanity Scoring (`vanity.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **pyod** | 2.0.2 | Outlier detection: Isolation Forest, LOF, COPOD, ECOD, AutoEncoder-based. 40+ algorithms. | Could detect "vanity outliers" -- companies whose spending patterns are anomalous relative to peers. `COPOD` (copula-based) identifies multivariate outliers efficiently. |
| **alibi-detect** | 0.12.1 | Drift detection: KS test, MMD, LSDD, learned kernel. | Detect when vanity metrics drift from historical norms -- a signal that capital allocation quality is changing. |

**Best pick:** `pyod` with COPOD for multivariate outlier detection of vanity spending patterns.

### 8. Economic Planes (`economic_planes.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **industry-classification** | 0.1.0 | Maps company descriptions to GICS/ICB sector/industry codes. | Could replace or enhance our YAML-based sector-to-plane mapping with a more comprehensive classification. |

### 9. Ethical Filters (`ethical_filters.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **pyaml-env** | 1.2.1 | YAML config with environment variable interpolation. | Minor: could enhance ethical filter threshold config. |

No major packages found. The ethical filters are custom domain logic (Islamic finance concepts: Gharar, purchasing power, solvency assessment).

---

## Layer 3: Temporal Models

### 10. Regime Detection (`regime_detector.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **hmmlearn** | 0.3.3 | Gaussian HMM for regime detection. | Already installed and used. |
| **pomegranate** | 1.1.0 | Probabilistic models: HMM with more flexible emission distributions (GMM, Poisson, multivariate). Semi-Markov models with explicit regime duration distributions. | Alternative to hmmlearn with richer emission models. Semi-Markov models let you model "how long does a crisis last?" directly. |
| **ruptures** | 1.1.10 | Change point detection: PELT, binary segmentation, bottom-up. | Already installed and used. |
| **changefinder** | 0.3 | Online change point detection using SDAR (Sequentially Discounting AR). Real-time, no look-ahead. | Unlike PELT (batch), ChangeFinder detects regime changes in real-time during the forward pass. 500 bytes memory per variable. |
| **bayesian-changepoint** | 0.2.0 | Bayesian online change point detection (Adams & MacKay 2007). Maintains a run-length distribution online. | More principled than our simplified BCP. Provides a posterior probability of "a change point happened at this exact day." |

**Best pick:** `changefinder` for real-time online detection during forward pass. `pomegranate` for semi-Markov regime duration modeling.

### 11. Forecasting (`forecasting.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **darts** | 0.32.0 | Unified forecasting: TFT, N-BEATS, DeepAR, TCN, ARIMA, Prophet, all with a single API. Supports probabilistic forecasts natively. | **High value.** Could replace our custom LSTM/Transformer with production-grade implementations. TFT produces calibrated intervals. N-BEATS needs no feature engineering. |
| **neuralforecast** | 1.7.5 | Neural forecasting: NHITS, PatchTST, TimesNet, FEDformer. Nixtla project. | State-of-the-art neural forecasters. PatchTST outperforms TFT on many benchmarks. Very fast training. |
| **statsforecast** | 2.0.1 | Statistical forecasting: AutoARIMA, AutoETS, AutoTheta, CrostonClassic. Blazing fast (Rust backend). | Could replace our Kalman/GARCH/VAR with auto-tuned statistical models. 100x faster than statsmodels for large batches. |
| **prophet** | 1.1.6 | Facebook's time series forecasting with trend, seasonality, holidays. | Good for variables with strong seasonal patterns (quarterly earnings). Not great for daily returns. |
| **sktime** | 0.35.0 | Unified time series ML: classification, regression, forecasting, transformation. sklearn-compatible. | Pipeline framework that unifies all our forecasting models under one interface. Supports walk-forward natively. |
| **gluonts** | 0.16.0 | Amazon's probabilistic time series library. DeepAR, Transformer, SimpleFeedForward with probabilistic outputs. | Amazon's production forecasting. All models output distributions, not just point forecasts. |

**Best pick:** `statsforecast` for 100x faster statistical models (AutoARIMA, AutoETS). Consider `darts` for TFT/N-BEATS if we want to upgrade the deep learning path.

### 12. Monte Carlo (`monte_carlo.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **riskfolio-lib** | 7.2.1 | Portfolio optimization: CVaR, drawdown constraints, risk parity, Black-Litterman. | Risk metrics that complement our Monte Carlo: CVaR, Expected Shortfall, Drawdown-at-Risk. |
| **quantlib** | 1.35 | QuantLib Python bindings. Monte Carlo engines, yield curves, option pricing. | Industrial-grade MC engines with variance reduction techniques (antithetic variates, control variates). |
| **pymc** | 5.28.0 | Bayesian probabilistic programming. | Already installed. Could replace frequentist MC with Bayesian posterior simulation using NUTS sampler. |
| **chaospy** | 4.3.18 | Uncertainty quantification via polynomial chaos expansion. | Alternative to brute-force MC: PCE achieves same accuracy with 100x fewer samples for smooth functions. |

**Best pick:** `riskfolio-lib` for CVaR/Expected Shortfall risk metrics on simulated paths. `chaospy` for sample-efficient uncertainty quantification.

### 13. Causality (`granger_causality.py`, `causality.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **dowhy** | 0.11.1 | Microsoft's causal inference library. DAG-based causal reasoning, do-calculus, causal effect estimation. | **High value.** Could upgrade our Granger (correlation-based) to true causal inference with confounders. Identifies spurious correlations. |
| **causalml** | 0.15.1 | Uber's causal ML. Uplift modeling, CATE estimation, meta-learners (S-learner, T-learner, X-learner). | Treatment effect estimation: "what is the causal effect of a rate cut on this company's survival probability?" |
| **tigramite** | 5.2.6 | Time series causal discovery: PCMCI, PCMCI+, LPCMCI. Handles autocorrelation and hidden confounders. | **Best for time series causality.** PCMCI is specifically designed for discovering causal links in time series with autocorrelated variables -- exactly our use case. |
| **lingam** | 1.9.0 | Linear Non-Gaussian Acyclic Model. Discovers causal direction from observational data using non-Gaussianity. | Can determine which direction the causality flows (does revenue cause stock price, or vice versa?) without experiments. |
| **pgmpy** | 0.1.26 | Probabilistic Graphical Models: Bayesian Networks, Markov Networks, causal inference. | Could build a Bayesian Network of financial variables where the structure encodes causal relationships. Used for what-if scenario analysis. |

**Best pick:** `tigramite` for PCMCI time-series causal discovery. `pgmpy` for Bayesian Network scenario analysis.

### 14. SHAP Explainability (`explainability.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **shap** | 0.50.0 | SHAP values for model explainability. | Already installed and used. |
| **lime** | 0.2.0 | Local Interpretable Model-Agnostic Explanations. Alternative to SHAP for local explanations. | Could provide a second explainability method for cross-validation of SHAP results. |
| **alibi** | 0.9.6 | Seldon's explainability toolkit: ALE plots, partial dependence, counterfactual explanations, anchors. | **Counterfactual explanations:** "what would need to change for this company to exit survival mode?" Produces actionable insights. |
| **interpret** | 0.6.4 | Microsoft's interpretable ML: EBM (Explainable Boosting Machine), glass-box models, SHAP integration. | EBM is a glass-box model that is both accurate AND inherently interpretable. Could replace our tree models where explainability is critical. |
| **dalex** | 1.7.0 | Model-agnostic explainability: feature importance, partial dependence, breakdown plots, Shapley values. | Unified explainability interface. Produces publication-quality explanation plots. |

**Best pick:** `alibi` for counterfactual explanations. `interpret` for EBM glass-box models.

### 15. Graph Risk (`graph_risk.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **networkx** | 3.x | Graph algorithms: betweenness centrality, community detection, shortest paths, Katz centrality. | Would add community detection (Louvain: is the target in a vulnerable sub-cluster?) and Katz centrality (accounts for indirect connections through the network). |
| **igraph** | 0.11.8 | Fast graph algorithms (C backend). 10-100x faster than networkx for large graphs. | Better for large entity networks (50+ nodes). Faster PageRank, community detection. |
| **graph-tool** | 2.x | C++ graph library. Stochastic block models, network inference, minimum description length. | SBMs can discover hidden community structure without specifying the number of communities. |
| **stellargraph** | 1.2.1 | Graph neural networks: GraphSAGE, GCN, GAT for link prediction and node classification. | Could predict which entity relationships are most likely to transmit contagion using learned graph embeddings. |
| **node2vec** | 0.4.6 | Graph embedding: converts graph structure to dense vectors. | Entity embeddings from the relationship graph. Similar entities cluster in embedding space. |

**Best pick:** `networkx` for community detection and Katz centrality (lightweight). `node2vec` for graph embeddings if we want ML on the entity network.

### 16. Game Theory (`game_theory.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **nashpy** | 0.0.41 | Compute Nash equilibria for 2-player games. Support enumeration and vertex enumeration methods. | Could compute Nash equilibria for the Cournot/Bertrand games we model, instead of our analytical approximations. |
| **gambit** | 16.2.0 | Full game theory toolkit: N-player games, Nash equilibria, extensive form games, quantal response equilibria. | Handles N-player competitive dynamics (our pipeline often has 3+ competitors). Extensive form games model sequential moves. |
| **axelrod** | 4.12.0 | Iterated Prisoner's Dilemma tournament framework. 200+ strategies. | Model repeated competitive interactions between firms. Identify whether competitors are cooperative or defecting. |

**Best pick:** `nashpy` for exact 2-player Nash equilibria. `gambit` for N-player games.

### 17. Genetic Optimizer (`genetic_optimizer.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **deap** | 1.4.1 | Evolutionary computation framework: GA, GP, PSO, CMA-ES. Multi-objective via NSGA-II. | More flexible than our custom GA. CMA-ES is better for continuous weight optimization. Multi-objective: optimize RMSE + stability + cost. |
| **pymoo** | 0.6.1 | Multi-objective optimization: NSGA-II, NSGA-III, MOEA/D, reference point methods. | Multi-objective Pareto optimization. Trade off prediction accuracy against computational cost. |
| **optuna** | 4.1.0 | Hyperparameter optimization with pruning: TPE, CMA-ES, random search, grid search. | Could replace GA for ensemble weight optimization. TPE often converges faster. Built-in pruning stops bad trials early. |
| **nevergrad** | 1.0.4 | Facebook's gradient-free optimization: OnePlusOne, CMA, PSO, DE, portfolio optimization. | Automatically selects the best optimizer for the problem type. Portfolio mode runs multiple optimizers in parallel. |
| **scipy.optimize** | (builtin) | `differential_evolution()`, `dual_annealing()`, `shgo()` for global optimization. | Already available. `differential_evolution` is a well-tested alternative to our custom GA with fewer parameters to tune. |

**Best pick:** `optuna` for TPE-based optimization with pruning. `scipy.optimize.differential_evolution` as a zero-dependency GA replacement.

### 18. Transformer Forecaster (`transformer_forecaster.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **pytorch-forecasting** | 1.1.1 | Production-grade TFT, N-BEATS, DeepAR on PyTorch. Variable selection, quantile heads, static covariates. | **High value.** Full TFT architecture with variable selection network that our basic transformer lacks. Produces calibrated quantile forecasts natively. |
| **tsai** | 0.3.9 | Time series AI: InceptionTime, ROCKET, MiniRocket, TST (Time Series Transformer), TSTPlus. fastai-based. | ROCKET/MiniRocket are extremely fast feature extractors. TST is a purpose-built time series transformer. |
| **neuralforecast** | 1.7.5 | NHITS, PatchTST, TimesNet, FEDformer, Autoformer. Nixtla project. | PatchTST achieves SOTA on many time series benchmarks. Very efficient training. |

**Best pick:** `pytorch-forecasting` for the full TFT architecture. `neuralforecast` for PatchTST.

### 19. Particle Filter (`particle_filter.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **particles** | 0.4 | Sequential Monte Carlo / particle filtering library. SMC^2, PMCMC, Rao-Blackwellization, tempering. | More sophisticated PF with variance reduction. Rao-Blackwellized PF shares state with Kalman filter. |
| **filterpy** | 1.4.5 | Kalman filters (standard, extended, unscented), particle filters, IMM (Interacting Multiple Model). | **IMM filter** runs multiple models (e.g., random walk + mean reversion) simultaneously and blends based on which fits recent data. Natural regime switching. |
| **pykalman** | 0.9.7 | Kalman filter and smoother. EM algorithm for parameter estimation. | EM-based parameter learning for the Kalman filter state-space model. |

**Best pick:** `filterpy` for IMM (Interacting Multiple Model) filter -- runs Kalman and particle filter in parallel, blends based on current fit. Natural regime-aware filtering.

### 20. Sensitivity Analysis (`sensitivity.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **SALib** | 1.5.1 | Sensitivity analysis: Sobol, Morris, FAST, Delta, PAWN. | Already used (optional). We could add PAWN (distribution-based) sensitivity which is more robust than Sobol for non-linear models. |
| **uncertainpy** | 1.2.3 | Uncertainty quantification + sensitivity analysis for computational models. | Combines MC uncertainty propagation with Sobol sensitivity in one framework. |

### 21. Model Synergies (`model_synergies.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **feature-engine** | 1.8.2 | Feature engineering pipelines with sklearn compatibility. | Could formalize the synergy feature injection (cycle phases, causal features) as a sklearn Pipeline. |

### 22. PID Controller (`pid_controller.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **simple-pid** | 2.0.1 | Simple PID controller implementation with auto-tuning. Anti-windup, output clamping, sample time. | Production-grade PID with anti-windup. Could replace our custom PID with a well-tested implementation. |
| **python-control** | 0.10.1 | Control systems library: transfer functions, state-space models, frequency response, PID tuning. | Provides Ziegler-Nichols auto-tuning for PID gains. Our PID gains are currently hand-tuned. |

**Best pick:** `simple-pid` for a robust PID with auto-tuning and anti-windup.

### 23. Estimation Engine (`estimator.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **miceforest** | 5.7.0 | Fast MICE imputation using LightGBM. Handles mixed types, supports predictive mean matching. | Faster and more flexible than our sklearn-based MICE. LightGBM handles non-linearities in the imputation model. |
| **missingno** | 0.5.2 | Missing data visualization: matrix plots, bar charts, heatmaps, dendrograms. | Diagnostic tool to visualize the missingness patterns in our financial data. Helps validate MAR vs MNAR classification. |
| **autoimpute** | 0.13.0 | Automated imputation: MICE, predictive mean matching, Bayesian regression, time series interpolation. | Comprehensive imputation library with time-series-aware methods. |
| **fancyimpute** | 0.7.0 | Matrix completion: SoftImpute, IterativeSVD, nuclear norm minimization. | We already implement SoftImpute manually. This provides a tested, optimized version. |
| **hyperimpute** | 0.1.20 | AutoML for imputation: automatically selects the best imputation method per column. | Meta-learning for imputation -- automatically picks the best method (MICE, KNN, MissForest) per variable. |

**Best pick:** `miceforest` for faster LightGBM-based MICE. `hyperimpute` for automatic method selection.

### 24. Frequency Interpolator (`frequency_interpolator.py`)

| Package | Version | What It Does | How It Helps Us |
|---------|---------|-------------|-----------------|
| **tsmoothie** | 1.0.5 | Time series smoothing: exponential, Kalman, Lowess, Gaussian, polynomial. With confidence intervals. | Could enhance our linear interpolation of stock variables with Kalman smoothing (better handles noisy quarterly data). |

---

## Priority Ranking: Top 15 Packages Across All Modules

| Rank | Package | Module | Impact | Size | Already Installed? |
|------|---------|--------|--------|------|-------------------|
| 1 | **lifelines** | survival_mode | Cox PH survival curves | ~5MB | No |
| 2 | **tigramite** | granger_causality | PCMCI causal discovery | ~3MB | No |
| 3 | **statsforecast** | forecasting | 100x faster AutoARIMA | ~5MB | No |
| 4 | **pytorch-forecasting** | transformer | Full TFT architecture | ~10MB | No |
| 5 | **ta** | derived_variables | 90+ technical indicators | ~3MB | No |
| 6 | **vaderSentiment** | news_sentiment | Better keyword sentiment | ~1MB | No |
| 7 | **optuna** | genetic_optimizer | TPE optimization | ~5MB | No |
| 8 | **miceforest** | estimator | LightGBM-based MICE | ~3MB | No |
| 9 | **filterpy** | particle_filter | IMM + Kalman-PF fusion | ~2MB | No |
| 10 | **nashpy** | game_theory | Exact Nash equilibria | ~1MB | No |
| 11 | **alibi** | explainability | Counterfactual explanations | ~5MB | No |
| 12 | **changefinder** | regime_detector | Online change points | ~1MB | No |
| 13 | **riskfolio-lib** | monte_carlo | CVaR, Expected Shortfall | ~5MB | No |
| 14 | **simple-pid** | pid_controller | Auto-tuning PID | ~100KB | No |
| 15 | **pyod** | vanity | Multivariate outlier detection | ~3MB | No |

---

## GitHub Repos Worth Investigating

| Repo | Stars | Relevant Module | What It Offers |
|------|-------|----------------|---------------|
| **uber/causalml** | 5k | granger_causality | Treatment effect estimation for policy analysis |
| **py-why/dowhy** | 7k | granger_causality | DAG-based causal reasoning with do-calculus |
| **jakobrunge/tigramite** | 1.2k | granger_causality | PCMCI for time series causal discovery |
| **Nixtla/statsforecast** | 4k | forecasting | Blazing fast statistical forecasting |
| **Nixtla/neuralforecast** | 3k | transformer | PatchTST, NHITS, TimesNet |
| **jdb78/pytorch-forecasting** | 3.8k | transformer | Full TFT, N-BEATS, DeepAR |
| **CamDavidsonPilon/lifelines** | 2.3k | survival_mode | Survival analysis with Cox PH |
| **optuna/optuna** | 11k | genetic_optimizer | Hyperparameter optimization with pruning |
| **bukosabino/ta** | 4.5k | derived_variables | Technical indicators library |
| **cjhutto/vaderSentiment** | 4.3k | news_sentiment | Rule-based sentiment analysis |
| **rlabbe/filterpy** | 3.2k | particle_filter | Kalman-particle fusion |
| **SeldonIO/alibi** | 2.5k | explainability | Counterfactual explanations |
| **yzhao062/pyod** | 8.5k | vanity | Outlier detection (40+ algorithms) |
| **AnotherSamWilson/miceforest** | 400 | estimator | Fast LightGBM-based MICE |
| **MaxHalford/prince** | 1.2k | derived_variables | Dimensionality reduction (PCA, MCA, FAMD) |
| **timeseriesAI/tsai** | 5k | transformer | Time series deep learning (ROCKET, TST) |
| **microsoft/interpret** | 6.3k | explainability | EBM glass-box interpretable ML |
