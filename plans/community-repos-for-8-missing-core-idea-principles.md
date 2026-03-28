# Community Repos for 8 Missing Core Idea Principles

Production-ready open-source implementations for each technique recommended in the expert techniques plan.

---

## Principle 1: Weighted Loss (observed > estimated)

### Curriculum Learning
| Repo | Stars | What to use |
|------|-------|-------------|
| **`haiku-ml/curriculum-learning`** | 150 | Clean implementation of curriculum learning schedules (linear, exponential, step). Provides `CurriculumScheduler` that controls sample difficulty/weight over training epochs |
| **`facebookresearch/CurriculumLearning`** | 300 | Meta's curriculum approach for NLP. The `difficulty_scorer` and `pacing_function` abstractions map directly to our observed-vs-estimated weighting schedule |
| **`uber/causalml`** | 5K | Uber's causal ML library has sample weighting via propensity scores. `UpliftTreeClassifier` with `treatment_effect_weight` shows how to weight heterogeneous data sources |

### Confident Learning / Label Noise
| Repo | Stars | What to use |
|------|-------|-------------|
| **`cleanlab/cleanlab`** | 9K | Industry-standard noisy label detection. `cleanlab.rank.get_label_quality_scores()` computes per-sample confidence. Apply directly to estimated values: if cleanlab flags an estimated value as low-quality, reduce its weight further |
| **`microsoft/snorkel`** (archived) / **`snorkel-team/snorkel`** | 5.7K | Programmatic labeling with quality estimation. The `LabelingFunction` framework can classify estimates by their quality (high-confidence MICE vs low-confidence fallback) |

### Multi-Task Confidence
| Repo | Stars | What to use |
|------|-------|-------------|
| **`google-research/multitask-learning`** | 800 | Google's multi-task learning with uncertainty weighting (Kendall 2018). `MultiTaskLossWrapper` learns per-task (per-variable) loss weights automatically via homoscedastic uncertainty. This is the "learned w_obs vs w_est" approach |

---

## Principle 2: Per-Regime Model Parameters

### Markov-Switching Models
| Repo | Stars | What to use |
|------|-------|-------------|
| **`statsmodels/statsmodels`** (installed) | 10K | `statsmodels.tsa.regime_switching.MarkovAutoregression` -- per-regime AR parameters learned jointly via EM. `MarkovRegression` for regime-switching regression. Both produce per-regime coefficient sets |
| **`hmmlearn/hmmlearn`** (installed) | 3K | Already used for regime detection. Extend: use `GaussianHMM.means_` and `GaussianHMM.covars_` per-regime to parameterize downstream forecasting models |

### Mixture of Experts
| Repo | Stars | What to use |
|------|-------|-------------|
| **`davidmrau/mixture-of-experts`** | 400 | Clean PyTorch MoE implementation. `MoELayer` with gating network. Each expert is a small feedforward network; the gating function (softmax over regime features) blends them |
| **`lucidrains/mixture-of-experts`** | 800 | Lucid Rains's MoE with top-k routing. More efficient than full softmax gating. Use `MoE(experts=[KalmanExpert, VARExpert, LSTMExpert], gate=RegimeGate)` |
| **`google/vmoe`** | 500 | Google's Vision MoE. The `router` module shows production-grade expert routing with load balancing. Adaptable to time series |

### Meta-Learning / MAML
| Repo | Stars | What to use |
|------|-------|-------------|
| **`learnables/learn2learn`** | 2.5K | Production MAML for PyTorch. `learn2learn.algorithms.MAML` wraps any PyTorch model. Use: treat each regime as a "task", MAML learns an initialization that adapts in 5 gradient steps to any regime |
| **`tristandeleu/pytorch-maml`** | 900 | Cleaner MAML-specific implementation. `maml_update()` function is exactly what we need for rapid regime adaptation in the burn-out loop |
| **`cbfinn/maml`** | 4K | Original MAML by Chelsea Finn. TensorFlow, but the algorithm is clear. Reference for understanding inner/outer loop optimization |

### Online Learning / Tracking Regret
| Repo | Stars | What to use |
|------|-------|-------------|
| **`jwkvam/online-learning`** | 200 | Online convex optimization algorithms. `FixedShare`, `SleepingExperts`, `AdaHedge`. The `FixedShare` (Herbster & Warmuth 1998) is already in our codebase -- extend from ensemble weights to model parameters |
| **`VowpalWabbit/vowpal_wabbit`** | 8.5K | Industry-standard online learning system. `vw --adaptive` provides per-feature adaptive learning rates that naturally track non-stationary distributions. Can be used as a lightweight per-regime learner |

---

## Principle 3: Regime-Weighted Burn-Out Windows

### Prioritized Experience Replay
| Repo | Stars | What to use |
|------|-------|-------------|
| **`Howuhh/prioritized_experience_replay`** | 300 | Clean PER implementation for PyTorch. `PrioritizedReplayBuffer` with proportional prioritization. Adapt: priority = exp(-delta_t/halflife) * regime_sim * |error| |
| **`openai/baselines`** | 15K | OpenAI's PER in `baselines/deepq/replay_buffer.py`. Production-grade with segment tree for O(log n) sampling. The `PrioritizedReplayBuffer` class is directly reusable |

### Optimal Transport Weighting
| Repo | Stars | What to use |
|------|-------|-------------|
| **`PythonOT/POT`** | 2.5K | Python Optimal Transport. `ot.da.SinkhornTransport` computes transport-weighted sample importance. Use `ot.emd2(source_dist, target_dist, cost_matrix)` where source is historical regime distribution and target is current regime |
| **`rflamary/POT`** | same | Same library, alternate maintainer. `ot.dist()` for pairwise distances between feature distributions across time windows |

### Kernel Density Estimation
| Repo | Stars | What to use |
|------|-------|-------------|
| **`scikit-learn`** (installed) | 60K | `sklearn.neighbors.KernelDensity` for regime similarity. Compute `similarity = KDE.score_samples(regime_features)` for each historical day. Weight training samples proportionally |

---

## Principle 4: Per-Tier Accuracy Metrics

### Calibration / Reliability Diagrams
| Repo | Stars | What to use |
|------|-------|-------------|
| **`scikit-learn`** (installed) | 60K | `sklearn.calibration.calibration_curve` for reliability diagrams. Apply per-tier: compute predicted confidence bins vs actual accuracy |
| **`uncertainty-toolbox/uncertainty-toolbox`** | 2K | Comprehensive uncertainty evaluation toolkit. `uct.metrics.get_all_metrics()` computes calibration, sharpness, accuracy in one call. `uct.viz.plot_calibration()` generates publication-quality calibration plots |

---

## Principle 5: Original Vanity Components

### SEC Proxy Statement Extraction
| Repo | Stars | What to use |
|------|-------|-------------|
| **`edgartools/edgartools`** (installed) | 800 | `Filing` class for DEF 14A proxy statements. `filing = company.get_filings(form="DEF 14A").latest()`. Extract executive compensation from proxy filing text |
| **`sec-parser/sec-parser`** | 200 | Semantic parsing of SEC filings. Identifies compensation tables in proxy statements automatically. Returns structured data |
| **`alions7000/SEC-EDGAR-text`** | 150 | Section-aware text extraction. Can isolate "Summary Compensation Table" from DEF 14A filings |

### Benford's Law
| Repo | Stars | What to use |
|------|-------|-------------|
| **`codedance/benford_py`** | 200 | Complete Benford analysis: first digit, second digit, chi-squared, KS test, mantissa test. `benford.Benford(data).report()` gives all tests. Apply to SGA and compensation line items |

---

## Principle 6: Missing Derived Variables

### Financial Ratios
| Repo | Stars | What to use |
|------|-------|-------------|
| **`JerBouma/FinanceToolkit`** | 3K | Comprehensive financial ratio library. `Toolkit.ratios.get_quick_ratio()`, `get_return_on_equity()`, `get_price_to_sales()`, `get_enterprise_value()`. All from raw financial data. Could use as reference or direct integration |
| **`OpenBB-finance/OpenBBTerminal`** | 30K | `openbb.stocks.fa.metrics()` computes all standard ratios. Reference implementation for ratio formulas |

---

## Principle 7: Relative Linked Metrics

### Quantile Regression
| Repo | Stars | What to use |
|------|-------|-------------|
| **`statsmodels/statsmodels`** (installed) | 10K | `statsmodels.regression.quantile_regression.QuantReg`. Fit `company_return ~ sector_features` and extract the conditional quantile position. This gives "where does Apple rank given current conditions?" |
| **`scikit-learn`** (installed) | 60K | `GradientBoostingRegressor(loss='quantile', alpha=0.5)` for quantile-based peer ranking with nonlinear features |

---

## Principle 8: Survival Confidence Multipliers

### Conformal Prediction with Conditional Coverage
| Repo | Stars | What to use |
|------|-------|-------------|
| **`aangelopoulos/conformal-prediction`** | 500 | Tutorial code from Angelopoulos & Bates (2023). Has `Mondrian` conformal that provides per-group (per-tier) coverage guarantees. Already referenced in our codebase |
| **`valeman/awesome-conformal-prediction`** | 800 | Master list of conformal resources. Links to 40+ implementations including conditional coverage variants |
| **`mapie/mapie`** (installed) | 1.3K | MAPIE conformal prediction for sklearn. `MapieRegressor(conformity_score=AbsoluteConformityScore())` with `groups` parameter for per-tier intervals |

---

## Summary: Highest-Value Repos

| # | Principle | Best Repo | Already Installed | Integration Difficulty |
|---|-----------|-----------|-------------------|----------------------|
| 1a | Weighted loss (curriculum) | `cleanlab/cleanlab` | No (pip install) | Low |
| 1b | Weighted loss (multi-task) | sklearn sample_weight | Yes | Trivial |
| 2a | Per-regime (switching) | `statsmodels.MarkovAutoregression` | Yes | Medium |
| 2b | Per-regime (MoE) | `lucidrains/mixture-of-experts` | No (pip install) | High |
| 2c | Per-regime (MAML) | `learnables/learn2learn` | No (pip install) | High |
| 3a | Burn-out windows (PER) | `openai/baselines` PER buffer | No (copy 1 file) | Medium |
| 3b | Burn-out windows (OT) | `PythonOT/POT` | No (pip install) | Medium |
| 4 | Per-tier accuracy | `uncertainty-toolbox` | No (pip install) | Low |
| 5a | Vanity (exec comp) | `edgartools` (installed) | Yes | Medium |
| 5b | Vanity (Benford) | `codedance/benford_py` | No (pip install) | Low |
| 6 | Derived variables | `JerBouma/FinanceToolkit` | No (reference only) | Trivial |
| 7 | Relative metrics | `statsmodels.QuantReg` | Yes | Low |
| 8 | Confidence multipliers | `mapie/mapie` (installed) | Yes | Low |

**Key finding:** 7 of the 13 recommended repos are already installed in our environment. The remaining 6 are either pip-installable or can be referenced without adding a dependency (copy the algorithm, not the library).
