"""Hedge Fund Normal Analysis Pipeline.

Parallel analytical track that runs alongside the existing 50-module
pipeline.  Answers "Can I make money on this, when, and how much?"
by computing 20 investment-grade metrics across 5 tiers:

  Tier 1 -- Earnings Forensics (FCF quality, accruals, smoothing)
  Tier 2 -- Cash Flow Stress Test (dividend burn, CROA/ROIC, leverage)
  Tier 3 -- Balance Sheet Risk (OBS, asset quality, stress scenarios)
  Tier 4 -- Inflection Detection (momentum, growth quality, surprise)
  Tier 5 -- Valuation Engine (DCF MC, quality-value matrix, PEG)

Integration layer fuses multi-frequency insights and produces an
Investment Thesis Scorecard with position signal.

Primary data source: raw quarterly statement DataFrames (8-24 rows),
NOT the 504-row daily cache.  Daily cache used only for price signals.
"""
