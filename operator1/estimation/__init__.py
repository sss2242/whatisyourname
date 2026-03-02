"""Post-cache estimation and imputation engine.

Supports three imputer backends (configured via ``global_config.yml``
key ``estimation_imputer``):

  - ``"split"`` (default): Classifies missingness as MAR or MNAR,
    then routes to specialized estimators:
      * MAR: MICE + Gaussian Process + Matrix Completion ensemble
      * MNAR: Heckman Selection + Pattern-Mixture + Sensitivity Bounds + GAIN
  - ``"bayesian_ridge"``: Legacy per-variable BayesianRidge (linear)
  - ``"vae"``: Legacy Variational Autoencoder (nonlinear, requires torch)
"""
