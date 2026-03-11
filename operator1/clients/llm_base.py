"""Abstract base class for LLM clients.

Defines the interface that both GeminiClient and ClaudeClient implement.
All methods return sensible fallbacks on failure -- the LLM is never a
hard dependency for the pipeline.

Includes shared retry logic with exponential backoff and per-host rate
limiting, consistent with the project's http_utils pattern.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

from operator1.config_loader import get_global_config

logger = logging.getLogger(__name__)


class LLMNonRetryableError(Exception):
    """Raised for non-retryable LLM API errors (e.g. 400 invalid key).

    This intentionally does NOT inherit from ``requests.RequestException``
    so that the retry catch block will not intercept it.
    """

    def __init__(self, provider: str, status_code: int, detail: str) -> None:
        self.provider = provider
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{provider} API error {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Per-host rate limiting (mirrors http_utils._rate_limit_sleep)
# ---------------------------------------------------------------------------

_last_request_time_by_host: dict[str, float] = {}

# LLM-specific rate limits (requests per second).
# Gemini free tier: 15 RPM = 0.25/s.  Paid: 1000 RPM = ~16/s.
# Claude: Tier 1 = 50 RPM = ~0.83/s.  Higher tiers scale up.
# These are conservative defaults; override via global_config.yml.
_LLM_HOST_RATE_LIMITS: dict[str, float] = {
    "generativelanguage.googleapis.com": 0.25,  # Gemini free tier
    "api.anthropic.com": 0.8,                   # Claude Tier 1
}


from operator1.http_utils import _extract_host, _sanitise_url  # shared utilities


def _rate_limit_sleep(host: str, calls_per_second: float | None = None) -> None:
    """Per-host rate limiter: sleep if requests are too fast."""
    effective_rate = calls_per_second or _LLM_HOST_RATE_LIMITS.get(host, 1.0)
    if effective_rate <= 0:
        return

    min_interval = 1.0 / effective_rate
    now = time.time()
    last_time = _last_request_time_by_host.get(host, 0.0)
    elapsed = now - last_time
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        logger.debug("LLM rate limiting [%s]: sleeping %.2fs", host, sleep_time)
        time.sleep(sleep_time)
    _last_request_time_by_host[host] = time.time()


# _sanitise_url imported from operator1.http_utils above


# ---------------------------------------------------------------------------
# Model capability definitions
# ---------------------------------------------------------------------------

# Each entry: (model_name, max_output_tokens, context_window, report_capable)
# "report_capable" means it can generate 8000-12000 word reports (~16k tokens)
GEMINI_MODELS: dict[str, dict[str, Any]] = {
    "gemini-2.5-flash-preview-05-20": {
        "max_output_tokens": 65536,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "preview",
    },
    "gemini-2.5-pro-preview-05-06": {
        "max_output_tokens": 65536,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "preview",
    },
    "gemini-2.0-flash": {
        "max_output_tokens": 8192,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "stable",
    },
    "gemini-1.5-pro": {
        "max_output_tokens": 8192,
        "context_window": 2097152,
        "report_capable": True,
        "tier": "stable",
    },
    "gemini-1.5-flash": {
        "max_output_tokens": 8192,
        "context_window": 1048576,
        "report_capable": True,
        "tier": "stable",
    },
}

CLAUDE_MODELS: dict[str, dict[str, Any]] = {
    "claude-opus-4-20250514": {
        "max_output_tokens": 32000,
        "context_window": 200000,
        "report_capable": True,
        "tier": "flagship",
    },
    "claude-sonnet-4-20250514": {
        "max_output_tokens": 64000,
        "context_window": 200000,
        "report_capable": True,
        "tier": "balanced",
    },
    "claude-3-5-sonnet-20241022": {
        "max_output_tokens": 8192,
        "context_window": 200000,
        "report_capable": True,
        "tier": "stable",
    },
    "claude-3-5-haiku-20241022": {
        "max_output_tokens": 8192,
        "context_window": 200000,
        "report_capable": True,
        "tier": "fast",
    },
}


def get_best_model(provider: str, model_registry: dict[str, dict[str, Any]]) -> str:
    """Pick the best report-capable model from a provider's registry.

    Prefers models with higher max_output_tokens and stable tier.
    Returns the model name string.
    """
    # Sort by: report_capable (True first), then max_output_tokens desc
    candidates = [
        (name, info)
        for name, info in model_registry.items()
        if info.get("report_capable", False)
    ]
    if not candidates:
        # Fallback: just return the first model
        return next(iter(model_registry))

    # Prefer stable/balanced tiers, then by max_output_tokens
    tier_priority = {"flagship": 0, "balanced": 1, "stable": 2, "fast": 3, "preview": 4}
    candidates.sort(
        key=lambda x: (
            tier_priority.get(x[1].get("tier", "stable"), 5),
            -x[1].get("max_output_tokens", 0),
        )
    )
    return candidates[0][0]


def validate_model(
    model: str,
    model_registry: dict[str, dict[str, Any]],
    provider: str,
) -> str:
    """Validate a model name against the registry.

    If the model is not in the registry, log a warning and return the
    best available model instead.
    """
    if model in model_registry:
        info = model_registry[model]
        logger.info(
            "%s model '%s': max_output=%d, context=%d, report_capable=%s",
            provider, model,
            info["max_output_tokens"],
            info["context_window"],
            info["report_capable"],
        )
        return model

    best = get_best_model(provider, model_registry)
    logger.warning(
        "%s model '%s' not in known registry. Using '%s' instead. "
        "Known models: %s",
        provider, model, best, ", ".join(model_registry.keys()),
    )
    return best


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract LLM client interface.

    Subclasses must implement ``_do_request`` (the actual HTTP call)
    and ``_build_request_args`` (provider-specific request construction).
    All higher-level methods (entity discovery, report generation,
    sentiment scoring) are implemented here using shared prompts.

    Includes built-in retry logic with exponential backoff and per-host
    rate limiting, following the project's existing ``http_utils`` pattern.
    """

    # ------------------------------------------------------------------
    # Abstract methods -- subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_request_args(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Build the request kwargs for requests.post().

        Must return a dict with keys: url, json, headers (optional),
        and any other kwargs for requests.post().
        """

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> str:
        """Extract the text content from a provider's JSON response."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'Gemini', 'Claude')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier currently in use."""

    @property
    @abstractmethod
    def max_output_tokens(self) -> int:
        """Maximum output tokens the current model supports."""

    # ------------------------------------------------------------------
    # Shared request execution with retry + rate limiting
    # ------------------------------------------------------------------

    def _execute_request(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 8192,
        temperature: float = 0.7,
        timeout: int = 60,
    ) -> str:
        """Execute an LLM request with retry logic and rate limiting.

        This is the core execution method that handles:
        - Per-host rate limiting
        - Exponential backoff on 429/5xx errors
        - Retry-After header respect
        - Request logging
        """
        cfg = get_global_config()
        max_retries: int = cfg.get("max_retries", 5)
        backoff: float = cfg.get("backoff_factor", 2.0)
        retryable_codes: set = set(cfg.get("retry_on_status", [429, 500, 502, 503, 504]))

        req_args = self._build_request_args(
            prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        url = req_args.pop("url")
        host = _extract_host(url)

        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            # Rate limit before each attempt
            _rate_limit_sleep(host)

            t0 = time.time()
            try:
                resp = requests.post(url, timeout=timeout, **req_args)
                elapsed = time.time() - t0

                if resp.status_code == 200:
                    logger.debug(
                        "%s request OK (%.1fs, attempt %d)",
                        self.provider_name, elapsed, attempt,
                    )
                    data = resp.json()
                    return self._parse_response(data)

                # Check if retryable
                if resp.status_code not in retryable_codes:
                    # Non-retryable error -- raise immediately
                    error_detail = resp.text[:500]
                    logger.error(
                        "%s API error %d (non-retryable): %s",
                        self.provider_name, resp.status_code, error_detail,
                    )
                    raise LLMNonRetryableError(
                        self.provider_name, resp.status_code, error_detail,
                    )

                # Retryable error -- backoff
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff ** attempt
                else:
                    wait = backoff ** attempt

                logger.warning(
                    "%s HTTP %d on attempt %d/%d for %s -- retrying in %.1fs",
                    self.provider_name, resp.status_code, attempt, max_retries,
                    _sanitise_url(url), wait,
                )
                last_exc = requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp,
                )
                time.sleep(wait)

            except requests.RequestException as exc:
                elapsed = time.time() - t0
                last_exc = exc
                wait = backoff ** attempt
                logger.warning(
                    "%s request error on attempt %d/%d (%.1fs): %s -- retrying in %.1fs",
                    self.provider_name, attempt, max_retries, elapsed, exc, wait,
                )
                time.sleep(wait)

        # All retries exhausted
        raise RuntimeError(
            f"{self.provider_name}: all {max_retries} retries exhausted. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # High-level generate methods (use _execute_request internally)
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Send a prompt and return the raw text response."""
        cfg = get_global_config()
        timeout = cfg.get("timeout_s", 30)
        return self._execute_request(prompt, timeout=timeout)

    def _generate_with_config(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 8192,
        temperature: float = 0.7,
        timeout: int = 60,
    ) -> str:
        """Send a prompt with generation config and return the raw text."""
        return self._execute_request(
            prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(text: str) -> Any:
        """Best-effort extraction of JSON from an LLM response.

        Handles responses that wrap JSON in markdown code fences.
        """
        cleaned = text.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response; returning raw text")
            return None

    # ------------------------------------------------------------------
    # Linked entity discovery (Sec 5)
    # ------------------------------------------------------------------

    _LINKED_ENTITIES_PROMPT = """\
You are a financial analyst. Given the following company profile, suggest
related entities grouped by relationship type.

Company profile:
{profile_json}

Sector hints: {sector_hints}

Return a JSON object with these keys. Each value is a list of objects
with the entity name and temporal context:

- competitors: direct competitors in the same industry
- suppliers: known major suppliers
- customers: known major customers
- financial_institutions: primary banks or lenders
- logistics: key logistics or distribution partners
- regulators: relevant regulatory bodies (if publicly listed)

Each entity should be an object with:
  "name": "Company Name" (publicly traded, not a ticker),
  "relationship_start": "YYYY" or "ongoing" or "unknown",
  "relationship_end": "current" or "YYYY" or "unknown",
  "stability": "stable" or "volatile" or "new"

Example format:
{{"competitors": [{{"name": "Example Corp", "relationship_start": "2020", "relationship_end": "current", "stability": "stable"}}]}}

IMPORTANT: Relationships change over time. A supplier from 3 years ago
may no longer be a supplier. Only include entities with CURRENT or
RECENT relationships (within the last 2 years). Mark ended relationships
with their approximate end date.

Only include companies you are reasonably confident about.
Return valid JSON only, no markdown.
"""

    def propose_linked_entities(
        self,
        target_profile: dict[str, Any],
        sector_hints: str = "",
    ) -> dict[str, list[str]]:
        """Ask LLM to propose linked entities for a target company.

        Returns dict mapping relationship_group -> list of company names.
        Returns empty dict on any failure.
        """
        try:
            prompt = self._LINKED_ENTITIES_PROMPT.format(
                profile_json=json.dumps(target_profile, indent=2),
                sector_hints=sector_hints or "none",
            )
            text = self._generate(prompt)
            parsed = self._parse_json_response(text)
            if isinstance(parsed, dict):
                # Handle both new format (list of objects) and old format (list of strings)
                result: dict[str, list[str]] = {}
                for k, vs in parsed.items():
                    if not isinstance(vs, list):
                        continue
                    names: list[str] = []
                    for v in vs:
                        if isinstance(v, dict):
                            # New temporal format: extract name
                            name = v.get("name", "")
                            if name:
                                names.append(str(name))
                        elif isinstance(v, str):
                            # Old format: plain string
                            names.append(v)
                    if names:
                        result[k] = names
                # Store the raw parsed data for temporal extraction
                self._last_entity_proposals_raw = parsed
                return result
            return {}
        except Exception as exc:
            logger.warning("%s linked-entity proposal failed: %s", self.provider_name, exc)
            return {}

    # ------------------------------------------------------------------
    # World Bank mapping suggestions (Sec 4)
    # ------------------------------------------------------------------

    _WB_MAPPING_PROMPT = """\
You are a macroeconomic data analyst. For the country "{country}" and
sector "{sector}", suggest the most relevant macroeconomic indicator codes
for these canonical variables:

- inflation_rate_yoy
- cpi_index
- unemployment_rate
- gdp_growth
- gdp_current_usd
- official_exchange_rate_lcu_per_usd
- current_account_balance_pct_gdp
- reserves_months_of_imports
- real_interest_rate
- lending_interest_rate
- deposit_interest_rate

Return a JSON object mapping each variable name to a macro indicator
code string. If unsure, omit the key.
Return valid JSON only, no markdown.
"""

    def propose_macro_indicator_mappings(
        self,
        country: str,
        sector: str = "",
    ) -> dict[str, str]:
        """Ask LLM to suggest macro indicator mappings.

        These are *suggestions only* -- not used at runtime without
        human review.  Returns empty dict on failure.
        """
        try:
            prompt = self._WB_MAPPING_PROMPT.format(
                country=country,
                sector=sector or "general",
            )
            text = self._generate(prompt)
            parsed = self._parse_json_response(text)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
            return {}
        except Exception as exc:
            logger.warning("%s WB mapping proposal failed: %s", self.provider_name, exc)
            return {}

    # ------------------------------------------------------------------
    # Report generation (Sec 18) -- Full 13-section prompt (Phase E1)
    # ------------------------------------------------------------------

    _REPORT_PROMPT = """\
You are a Bloomberg-style financial analyst specializing in comprehensive equity research.

You have been provided with a complete company profile that includes:
- 2 years of historical financial and market data
- Advanced temporal analysis using 25+ mathematical models (HMM, Kalman, \
GARCH, VAR, LSTM with MC Dropout, Temporal Fusion Transformer, Random Forest, XGBoost, etc.)
- Survival mode analysis (company + country)
- Ethical filter assessments (Purchasing Power, Solvency, Gharar, Cash is King)
- Multi-horizon predictions with Conformal Prediction intervals (distribution-free, \
guaranteed coverage -- not Gaussian assumptions)
- SHAP feature attribution explaining what drove each prediction
- MC Dropout epistemic uncertainty (separates "model does not know" from "inherent randomness")
- Dynamic Time Warping (DTW) historical analogs ("the last time this pattern occurred was...")
- Regime detection and structural break analysis
- Linked variables (sector, industry, competitors, macro indicators)
- Financial health composite scoring (0-100 scale across 5 tiers)
- News sentiment analysis (AI-scored from recent stock news articles)
- Peer percentile ranking (target vs peer group on all key metrics)
- Macro environment quadrant classification (goldilocks/reflation/stagflation/deflation)

Your task is to generate a professional investment report that synthesizes \
this information into actionable insights for sophisticated investors.

---

REPORT STRUCTURE (MUST INCLUDE ALL 13 SECTIONS):

1. EXECUTIVE SUMMARY
   - 3 bullet points summarizing key findings
   - Clear investment recommendation: BUY / HOLD / SELL with confidence level (High/Medium/Low)
   - 12-month target price with rationale

2. COMPANY OVERVIEW
   - Company identity and classification
   - Current market position and capitalization
   - Sector and industry context

3. HISTORICAL PERFORMANCE ANALYSIS (2 Years)
   - Total return vs real return (Purchasing Power filter applied)
   - Risk-adjusted performance: Sharpe ratio, maximum drawdown
   - Regime breakdown: time spent in bull/bear/high-volatility regimes
   - Structural breaks and major market events detected
   - Up days vs down days distribution

4. CURRENT FINANCIAL HEALTH (Tier-by-Tier Breakdown)

   **Explain the Tier Hierarchy:**
   - Why the 5-tier system matters
   - How weights change in different survival regimes
   - Current hierarchy weights and what they mean

   **Tier 1: Liquidity & Cash** -- Cash and equivalents, cash ratio, free cash flow; Cash is King filter results
   **Tier 2: Solvency & Debt** -- Debt-to-equity, net debt to EBITDA, interest coverage; Solvency filter results
   **Tier 3: Market Stability** -- Volatility, drawdown, volume; Gharar filter results
   **Tier 4: Profitability** -- Margins (gross, operating, net), ROE, ROA
   **Tier 5: Growth & Valuation** -- Revenue/earnings growth, P/E, EV/EBITDA

5. SURVIVAL MODE ANALYSIS
   - Current survival status (company, country, protection)
   - Historical survival episodes (count and duration)
   - Capital allocation quality (vanity score): composite 0-100 score measuring \
management discipline. Components: R&D vs Growth Mismatch, SGA Bloat (peer-relative), \
Capital Misallocation (debt/dividend signals), Competitive Decay (losing ground), \
Sentiment-Reality Gap (positive PR vs declining fundamentals). Labels: Disciplined (<20), \
Moderate (20-40), Wasteful (40-70), Reckless (>70). Interpret the trend (rising/falling/stable) \
and explain which components drive the score. Relate to management quality for investors.

6. LINKED VARIABLES & MARKET CONTEXT
   - **Peer Ranking**: percentile rank of target vs peer group on key metrics. \
A composite rank above 60 = above average; below 40 = below average. Show the top \
variable rankings in a table. Explain what relative positioning means for investors.
   - Sector performance: relative strength vs sector median
   - Industry positioning: valuation premium/discount vs industry
   - Competitor health assessment: how does the company compare?
   - Supply chain risk analysis (if applicable)
   - **Macro Environment Quadrant**: classify as Goldilocks (growth + low inflation), \
Reflation (growth + high inflation), Stagflation (low growth + high inflation), or \
Deflation (low growth + low inflation). Explain how the current quadrant affects this stock. \
Show the quadrant distribution over the analysis window.
   - **News Sentiment**: summarize AI-scored sentiment from recent news articles. \
Include mean sentiment, latest reading, and momentum trend. \
Explain what the sentiment trajectory means for near-term price action.
   - Macro indicators from per-region APIs (inflation, GDP, unemployment, FX)

7. TEMPORAL ANALYSIS & MODEL INSIGHTS
   - Current market regime and regime distribution over 2 years
   - Regime transitions and what they signal
   - Structural breaks detected
   - Model performance summary by tier (accuracy percentages)
   - Best performing module (from 23+ model ensemble including Kalman, GARCH, \
VAR, LSTM, Temporal Fusion Transformer, tree ensembles)
   - Conformal prediction calibration quality (empirical coverage vs target)
   - SHAP global feature importance: which variables matter most across all predictions
   - Confidence levels in predictions

8. PREDICTIONS & FORECASTS
   **Next Day:** NOTE: Due to Technical Alpha protection, only Low price \
is shown for next-day OHLC. Include Tier 1-5 variable predictions with \
Conformal Prediction intervals (distribution-free, guaranteed 90% coverage). \
Survival probability for next day.
   **Next Week:** Expected return and volatility, full OHLC candlestick series, predicted technical patterns.
   **Next Month:** Price target range (conformal 5th-95th percentile bands), key events to watch, regime shift predictions.
   **Next Year:** Annual outlook with widening uncertainty, predicted regime changes by quarter, long-term trajectory.
   **Monte Carlo Uncertainty:** Tail risk scenarios (worst 5%), base case (50th percentile), upside scenarios (top 5%).
   **SHAP Feature Drivers:** For key predictions, explain WHAT drove the forecast \
(e.g. "next-day cash_ratio prediction driven primarily by: +0.03 from declining \
short-term debt, -0.01 from rising volatility"). Use the SHAP narratives from the profile data.
   **Historical Analogs (DTW):** If available, describe the closest historical analog \
periods found via Dynamic Time Warping. Example: "The last time this company showed \
a similar pattern of rising debt + falling margins + high macro stress was [date]. \
In the following month, the stock [outcome]." Include the empirical return distribution \
from analog outcomes.
   **Epistemic vs Aleatoric Uncertainty:** Where MC Dropout data is available, \
distinguish between model uncertainty ("the model is unsure about this prediction") \
and inherent randomness ("even a perfect model would see variance here"). This helps \
investors understand the *quality* of each prediction.

9. TECHNICAL PATTERNS & CHART ANALYSIS
   - Describe the 2-year price chart with regime shading
   - Historical candlestick patterns detected (last 6 months)
   - Predicted patterns for next week/month
   - Support and resistance levels

10. ETHICAL FILTER ASSESSMENT
    **Purchasing Power Filter:** Verdict, nominal vs real return, inflation impact.
    **Solvency Filter:** Verdict, debt-to-equity ratio, threshold.
    **Gharar Filter (Uncertainty/Speculation):** Verdict, volatility, stability score.
    **Cash is King Filter:** Verdict, FCF yield.
    **Overall Ethical Score:** Combine all filters; is this investment suitable for ethical/Islamic investors?
    Universal lessons: Why these filters matter for ALL investors.

11. RISK FACTORS & LIMITATIONS
    - Model assumptions and their limitations
    - Key risks: company-specific, industry/sector, macro/country
    - Scenarios that could invalidate predictions
    - Black swan events not captured by models
    11.1 LIMITATIONS (SHORT, REQUIRED):
    Provide 5-10 bullets covering: data window, OHLCV source (FMP) caveats, \
macro frequency reality (macro APIs provide monthly/quarterly data, aligned daily via as-of logic), \
missingness summary, any modules that failed and how the report compensated. \
Must be easy for a non-technical client to understand.

12. INVESTMENT RECOMMENDATION
    **Recommendation:** [BUY / HOLD / SELL]
    **Confidence Level:** [High / Medium / Low]
    **12-Month Target Price:** with rationale
    **Key Catalysts to Watch:** events or metrics that would change the recommendation
    **Entry Strategy:** recommended entry price or conditions
    **Exit Strategy:** price targets for profits and stop-loss levels
    **Position Sizing:** suggested portfolio allocation based on risk profile

13. APPENDIX
    - Methodology summary: all 23+ temporal modules used:
      * Regime Detection: HMM, GMM, PELT, Bayesian Change Point
      * Forecasting: Adaptive Kalman, GARCH, VAR, LSTM, Temporal Fusion Transformer (TFT)
      * Tree Ensembles: Random Forest, XGBoost, Gradient Boosting
      * Causality: Granger Causality, Transfer Entropy, Copula Models
      * Uncertainty: Conformal Prediction (distribution-free intervals), \
MC Dropout (epistemic uncertainty), Regime-Aware Monte Carlo, Importance Sampling
      * Explainability: SHAP (per-prediction feature attribution), Sobol Sensitivity
      * Historical Analogs: Dynamic Time Warping (DTW) for finding similar past periods
      * Optimisation: Genetic Algorithm for ensemble weights
      * Pattern Recognition: Candlestick Detector, Wavelet/Fourier Decomposition
    - Forward pass + burn-out process explanation (day-by-day predict-compare-update)
    - Conformal Prediction explanation: why distribution-free intervals are more \
reliable than Gaussian assumptions for financial data
    - SHAP explanation: how per-prediction drivers are computed
    - Ensemble weighting approach (inverse-RMSE + GA optimisation)
    - Variable tier definitions (Tier 1-5 explained)
    - Glossary of technical terms (including Conformal Prediction, SHAP, TFT, \
regime, structural break, nonconformity score)
    - Data sources: PIT government filings (financials + OHLCV), per-region macro APIs, LLM (relationships)
    - Data timestamps and coverage
    - Disclaimer and limitations

---

FORMATTING REQUIREMENTS:
- Use markdown formatting with clear section headers (##, ###)
- Use tables for financial data where appropriate
- Use bullet points and numbered lists for clarity
- Bold key metrics and verdicts
- Italicize interpretive commentary
- Include placeholders for charts: [CHART: Description]
- Keep language professional but accessible
- Explain technical concepts when first introduced
- Total length: aim for 8,000-12,000 words (comprehensive but readable)

---

COMPLETE COMPANY PROFILE DATA:
{profile_json}

---

Generate the complete Bloomberg-style investment report now.
"""

    def generate_report(
        self,
        company_profile_json: str,
        *,
        max_output_tokens: int = 16000,
        temperature: float = 0.3,
        timeout: int = 120,
    ) -> str:
        """Generate a Bloomberg-style analysis report from profile data.

        Automatically caps max_output_tokens to the model's limit if
        the requested value exceeds it.

        Parameters
        ----------
        company_profile_json:
            JSON string of the full company profile.
        max_output_tokens:
            Maximum tokens for the response (default 16000 for
            comprehensive reports).
        temperature:
            Sampling temperature (lower = more factual).
        timeout:
            Request timeout in seconds (report generation is slow).

        Returns
        -------
        Markdown report string, or a fallback message on failure.
        """
        # Cap to model's actual limit
        effective_tokens = min(max_output_tokens, self.max_output_tokens)
        if effective_tokens < max_output_tokens:
            logger.info(
                "%s model '%s' max output is %d tokens (requested %d); "
                "capping to model limit.",
                self.provider_name, self.model_name,
                self.max_output_tokens, max_output_tokens,
            )

        try:
            prompt = self._REPORT_PROMPT.format(
                profile_json=company_profile_json,
            )
            return self._generate_with_config(
                prompt,
                max_output_tokens=effective_tokens,
                temperature=temperature,
                timeout=timeout,
            )

        except Exception as exc:
            logger.error("%s report generation failed: %s", self.provider_name, exc)
            return (
                "# Report Generation Failed\n\n"
                "The automated report could not be generated. "
                "Please review the raw data in the cache artifacts.\n\n"
                f"Error: {exc}\n"
            )

    # ------------------------------------------------------------------
    # Sentiment scoring (batch)
    # ------------------------------------------------------------------

    def score_sentiment(
        self,
        headlines: list[str],
        *,
        batch_size: int = 500,
    ) -> list[float]:
        """Score sentiment for a batch of news headlines.

        Sends all headlines in a single call (or splits into batches of
        ``batch_size`` if too many). Returns a list of scores from -1.0
        (very bearish) to +1.0 (very bullish).

        Parameters
        ----------
        headlines:
            List of headline strings.
        batch_size:
            Maximum headlines per API call.

        Returns
        -------
        list[float]
            Sentiment scores aligned 1:1 with input headlines.
            Returns empty list on failure.
        """
        if not headlines:
            return []

        all_scores: list[float] = []

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            numbered = "\n".join(
                f"{j + 1}. {h}" for j, h in enumerate(batch)
            )
            prompt = (
                "Score the sentiment of each financial news headline below "
                "from -1.0 (very bearish/negative for the stock) to +1.0 "
                "(very bullish/positive for the stock). 0.0 means neutral.\n\n"
                "Return ONLY a JSON array of numbers in the same order, "
                "nothing else. Example: [-0.3, 0.8, 0.0, -0.5]\n\n"
                f"Headlines:\n{numbered}"
            )

            try:
                text = self._generate(prompt)
                scores = self._parse_json_response(text)

                if isinstance(scores, list) and len(scores) == len(batch):
                    # Validate all are numbers in [-1, 1]
                    validated = []
                    for s in scores:
                        try:
                            val = float(s)
                            validated.append(max(-1.0, min(1.0, val)))
                        except (TypeError, ValueError):
                            validated.append(0.0)
                    all_scores.extend(validated)
                else:
                    logger.warning(
                        "%s sentiment: expected %d scores, got %s",
                        self.provider_name,
                        len(batch),
                        type(scores).__name__,
                    )
                    all_scores.extend([0.0] * len(batch))

            except Exception as exc:
                logger.warning("%s sentiment scoring failed: %s", self.provider_name, exc)
                all_scores.extend([0.0] * len(batch))

        return all_scores
