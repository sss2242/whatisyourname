"""Product segment analysis -- revenue decomposition, lifecycle, pricing power.

Computes 20 product-level cache columns from segment revenue data.
Runs at Step 4d in main.py (after cache build, before derived variables).

Data sources (waterfall):
1. XBRL segment dimensions from edgartools (free, real-time)
2. SimFin API segment data (free, 2K/day)
3. LLM extraction from annual report text (fallback)

Methods implemented:
- M1: Revenue segment decomposition (HHI, concentration)
- M2: Product lifecycle classification (Bass diffusion model)
- M3: Gross margin bridge (mix vs cost decomposition)
- M4: TAM/market share estimation (SOM from macro data)
- M6: Pricing power index (real revenue growth vs PPI)
- M9: Input cost pressure (commodity trends vs margins)
- M11: Cannibalization detection (new vs old product revenue shifts)
- M12: Network effect estimation (revenue acceleration + margin stability)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ProductSegmentResult:
    """Result from product segment analysis."""

    available: bool = False
    source: str = ""           # "xbrl", "simfin", "llm", "none"
    n_segments: int = 0
    dominant_segment: str = ""
    dominant_segment_pct: float = 0.0
    hhi: float = 0.0
    lifecycle_stage: str = "unknown"
    growth_runway_quarters: int = 0
    pricing_power: float = 0.0
    cannibalization_rate: float = 0.0
    network_effect_score: float = 0.0
    error: str = ""

    def to_profile_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "source": self.source,
            "n_segments": self.n_segments,
            "dominant_segment": self.dominant_segment,
            "dominant_segment_pct": round(self.dominant_segment_pct, 4),
            "hhi": round(self.hhi, 4),
            "lifecycle_stage": self.lifecycle_stage,
            "growth_runway_quarters": self.growth_runway_quarters,
            "pricing_power": round(self.pricing_power, 4),
            "cannibalization_rate": round(self.cannibalization_rate, 4),
            "network_effect_score": round(self.network_effect_score, 4),
        }


# ---------------------------------------------------------------------------
# Data fetching (waterfall: XBRL -> SimFin -> LLM)
# ---------------------------------------------------------------------------

def fetch_product_segments(
    ticker: str,
    market_id: str,
    pit_client: Any = None,
    secrets: dict | None = None,
) -> dict[str, Any]:
    """Fetch product segment revenue via 4-source waterfall.

    Returns dict with 'source' and 'segments' keys.
    'segments' is a dict of {segment_name: pd.Series} with quarterly revenue.
    """
    # Source 1: XBRL segment dimensions (US only, via edgartools)
    if market_id == "us_sec_edgar" and pit_client is not None:
        try:
            segments = _try_xbrl_segments(ticker, pit_client)
            if segments and len(segments) >= 2:
                logger.info("Product segments via XBRL: %d segments", len(segments))
                return {"source": "xbrl", "segments": segments}
        except Exception as exc:
            logger.debug("XBRL segment extraction failed: %s", exc)

    # Source 2: SimFin API
    try:
        from operator1.clients.simfin_segments import fetch_simfin_segments
        segments = fetch_simfin_segments(ticker)
        if segments and len(segments) >= 2:
            return {"source": "simfin", "segments": segments}
    except Exception as exc:
        logger.debug("SimFin segment fetch failed: %s", exc)

    # Source 3: LLM extraction from latest filing text
    if secrets and pit_client is not None:
        try:
            segments = _try_llm_segment_extraction(ticker, market_id, pit_client, secrets)
            if segments and len(segments) >= 2:
                return {"source": "llm", "segments": segments}
        except Exception as exc:
            logger.debug("LLM segment extraction failed: %s", exc)

    # Source 4: Fuzzy PDF extraction from filing discovery (no LLM needed)
    # Covers Tier 2 markets (AU, CA, SG, ZA, AE, IN, HK, SA, MX) where
    # the filing discovery framework can download annual report PDFs.
    try:
        segments = _try_pdf_segment_extraction(ticker, market_id)
        if segments and len(segments) >= 2:
            return {"source": "pdf", "segments": segments}
    except Exception as exc:
        logger.debug("PDF segment extraction failed: %s", exc)

    return {"source": "none", "segments": {}}


def _try_xbrl_segments(ticker: str, pit_client: Any) -> dict[str, pd.Series]:
    """Extract segment revenue from XBRL dimension members via edgartools."""
    try:
        from edgar import Company  # type: ignore[import-untyped]

        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        if not filings:
            return {}

        segments: dict[str, list[dict]] = {}
        # Check last 3 annual filings for segment data
        for filing in list(filings)[:3]:
            try:
                xbrl_data = filing.xbrl()
                if xbrl_data is None:
                    continue

                facts = xbrl_data.facts if hasattr(xbrl_data, "facts") else []
                for fact in facts:
                    # Look for revenue facts with ProductOrServiceAxis dimension
                    concept = getattr(fact, "concept", "") or ""
                    if "Revenue" not in concept:
                        continue

                    dimensions = getattr(fact, "dimensions", []) or []
                    for dim in dimensions:
                        dim_name = getattr(dim, "dimension", "") or ""
                        if "ProductOrServiceAxis" in dim_name or "SegmentAxis" in dim_name:
                            member = getattr(dim, "member", "") or ""
                            segment_name = member.replace("Member", "").split(":")[-1]
                            if not segment_name:
                                continue

                            period_end = getattr(fact, "period_end", None)
                            value = getattr(fact, "value", None)
                            if period_end and value is not None:
                                try:
                                    segments.setdefault(segment_name, []).append({
                                        "date": pd.Timestamp(str(period_end)),
                                        "revenue": float(value),
                                    })
                                except (ValueError, TypeError):
                                    continue
            except Exception:
                continue

        return {
            name: pd.Series(
                {r["date"]: r["revenue"] for r in rows}
            ).sort_index()
            for name, rows in segments.items()
            if len(rows) >= 2
        }
    except Exception as exc:
        logger.debug("XBRL segment extraction error: %s", exc)
        return {}


def _try_llm_segment_extraction(
    ticker: str,
    market_id: str,
    pit_client: Any,
    secrets: dict,
) -> dict[str, pd.Series]:
    """Extract segment revenue from filing text via LLM."""
    try:
        from operator1.clients.llm_factory import create_llm_client

        llm = create_llm_client(secrets)
        if llm is None:
            return {}

        # Get the latest filing profile for context
        profile = pit_client.get_profile(ticker)
        company_name = profile.get("name", ticker)

        prompt = (
            f"For {company_name} ({ticker}), list the major product/business segments "
            f"with their most recent annual revenue in USD. "
            f"Return ONLY valid JSON: "
            f'{{"segments": [{{"name": "Segment Name", "revenue": 12345678900}}]}}'
        )

        response = llm.generate(prompt)

        import json
        # Try to extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            raw_segments = data.get("segments", [])

            if len(raw_segments) >= 2:
                now = pd.Timestamp.now().normalize()
                return {
                    seg["name"]: pd.Series({now: float(seg["revenue"])})
                    for seg in raw_segments
                    if seg.get("name") and seg.get("revenue")
                }
    except Exception as exc:
        logger.debug("LLM segment extraction error: %s", exc)
    return {}


def _try_pdf_segment_extraction(
    ticker: str,
    market_id: str,
) -> dict[str, pd.Series]:
    """Extract segment revenue from annual report PDFs via fuzzy PDF parser.

    Uses the filing discovery framework to download annual report PDFs,
    then applies the fuzzy_pdf_parser's segment extraction (no LLM needed).
    This covers Tier 2 markets (AU, CA, SG, ZA, AE, IN, HK, SA, MX)
    where structured XBRL segment data is unavailable.
    """
    try:
        from operator1.clients.filing_discoverer import get_discoverer
        from operator1.clients.fuzzy_pdf_parser import extract_segments_from_pdf
    except ImportError:
        logger.debug("Filing discoverer or fuzzy_pdf_parser not available")
        return {}

    discoverer = get_discoverer(market_id)
    if discoverer is None:
        return {}

    try:
        # Discover filings (annual reports preferred for segment data)
        discovery = discoverer.discover_filings(ticker, years=2)
        if not discovery or not discovery.filings:
            return {}

        # Filter for annual reports first, then any filing
        annual_filings = [
            f for f in discovery.filings
            if f.filing_type in ("annual", "annual_report", "10-K", "20-F")
        ]
        target_filings = annual_filings if annual_filings else discovery.filings[:3]

        for filing in target_filings[:3]:
            try:
                pdf_bytes = discoverer.download_filing(filing)
                if not pdf_bytes or len(pdf_bytes) < 1000:
                    continue

                # Validate PDF magic bytes
                if not pdf_bytes[:4] == b"%PDF":
                    continue

                raw_segments = extract_segments_from_pdf(
                    pdf_bytes,
                    filing_date=getattr(filing, "filing_date", "") or "",
                    report_date=getattr(filing, "report_date", "") or "",
                    market_id=market_id,
                )

                if raw_segments and len(raw_segments) >= 2:
                    # Convert to pd.Series format expected by the pipeline
                    report_date = getattr(filing, "report_date", None)
                    ts = pd.Timestamp(str(report_date)) if report_date else pd.Timestamp.now().normalize()

                    return {
                        name: pd.Series({ts: revenue})
                        for name, revenue in raw_segments.items()
                    }
            except Exception as exc:
                logger.debug("PDF segment extraction for filing failed: %s", exc)
                continue

    except Exception as exc:
        logger.debug("PDF segment discovery failed: %s", exc)

    return {}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def compute_product_segment_features(
    cache: pd.DataFrame,
    segment_data: dict[str, pd.Series],
    target_profile: dict,
    macro_data: dict | None = None,
) -> tuple[pd.DataFrame, ProductSegmentResult]:
    """Compute all product-level features and merge into cache.

    Parameters
    ----------
    cache:
        Daily feature cache from pipeline.
    segment_data:
        Dict of {segment_name: pd.Series} with quarterly revenue.
    target_profile:
        Company profile dict with sector, industry.
    macro_data:
        Dict of macro indicators (gdp, inflation, etc.)

    Returns
    -------
    (cache, ProductSegmentResult)
    """
    result = ProductSegmentResult()

    if not segment_data or len(segment_data) < 2:
        return cache, result

    # ── M1: Revenue Segment Decomposition ──
    latest_revs = {}
    for name, series in segment_data.items():
        if len(series) > 0:
            latest_revs[name] = float(series.iloc[-1])

    total_rev = sum(latest_revs.values())
    if total_rev <= 0:
        return cache, result

    shares = {name: rev / total_rev for name, rev in latest_revs.items()}

    # HHI (0 = perfectly diversified, 1 = single product)
    hhi = sum(s ** 2 for s in shares.values())
    cache["segment_hhi"] = hhi
    cache["segment_count"] = float(len(segment_data))

    # Dominant segment
    dominant = max(shares, key=shares.get)
    cache["dominant_segment_pct"] = shares[dominant]

    # Per-segment growth rates
    growth_rates = {}
    for name, series in segment_data.items():
        if len(series) >= 2:
            prev = float(series.iloc[-2])
            curr = float(series.iloc[-1])
            growth_rates[name] = (curr - prev) / abs(prev) if prev != 0 else 0.0

    cache["dominant_segment_growth"] = growth_rates.get(dominant, 0.0)

    # Diversification change (4Q HHI delta)
    if all(len(s) >= 5 for s in segment_data.values()):
        old_revs = {name: float(s.iloc[-5]) for name, s in segment_data.items()}
        old_total = sum(old_revs.values())
        if old_total > 0:
            old_shares = {name: rev / old_total for name, rev in old_revs.items()}
            old_hhi = sum(s ** 2 for s in old_shares.values())
            cache["segment_diversification_delta"] = hhi - old_hhi

    # ── M2: Product Lifecycle Classification (Bass) ──
    lifecycle_stage = "unknown"
    runway = 0
    for name, series in segment_data.items():
        if len(series) >= 6 and name == dominant:
            lifecycle_stage, runway = _classify_lifecycle(series)
            cache["product_lifecycle_stage"] = _stage_to_numeric(lifecycle_stage)
            cache["growth_runway_quarters"] = float(runway)

    # Maturity concentration
    maturity_rev = 0.0
    for name, series in segment_data.items():
        if len(series) >= 6:
            stage, _ = _classify_lifecycle(series)
            if stage in ("maturity", "decline"):
                maturity_rev += latest_revs.get(name, 0)
    cache["maturity_concentration"] = maturity_rev / total_rev

    # ── M3: Gross Margin Bridge ──
    if "gross_margin" in cache.columns:
        mix_effect = _compute_mix_effect(segment_data)
        cache["margin_mix_effect"] = mix_effect
        cache["margin_sustainability_score"] = _margin_sustainability(mix_effect, cache)

    # ── M4: Market Share Estimation ──
    if macro_data and macro_data.get("gdp") is not None:
        sector = target_profile.get("sector", "")
        som = _estimate_market_share(total_rev, macro_data, sector)
        cache["estimated_market_share"] = som
        if all(len(s) >= 5 for s in segment_data.values()):
            old_total_rev = sum(float(s.iloc[-5]) for s in segment_data.values())
            old_som = _estimate_market_share(old_total_rev, macro_data, sector)
            cache["som_trend_4q"] = som - old_som

    # ── M5: Customer Concentration Proxy ──
    cache["customer_concentration_proxy"] = hhi * shares.get(dominant, 0)

    # ── M6: Pricing Power Index ──
    if "revenue" in cache.columns and macro_data:
        pp = _compute_pricing_power(cache, macro_data, target_profile.get("sector", ""))
        cache["pricing_power_index"] = pp
    else:
        pp = 0.0

    # ── M9: Input Cost Pressure ──
    if macro_data:
        icp = _compute_input_cost_pressure(cache, macro_data, target_profile.get("sector", ""))
        cache["input_cost_pressure"] = icp

    # ── M11: Cannibalization Detection ──
    cannibal_rate = 0.0
    net_new_pct = 1.0
    if len(segment_data) >= 2 and all(len(s) >= 4 for s in segment_data.values()):
        cannibal = _detect_cannibalization(segment_data)
        cannibal_rate = cannibal.get("rate", 0)
        net_new_pct = cannibal.get("net_new_pct", 1.0)
        cache["cannibalization_rate"] = cannibal_rate
        cache["net_new_revenue_pct"] = net_new_pct

    # ── M12: Network Effect (tech/platform sectors) ──
    nfx = 0.0
    sector = target_profile.get("sector", "").lower()
    if any(w in sector for w in ("technology", "communication", "software", "internet", "information")):
        nfx = _estimate_network_effect(cache)
        cache["network_effect_score"] = nfx

    # Build result
    result.available = True
    result.n_segments = len(segment_data)
    result.dominant_segment = dominant
    result.dominant_segment_pct = shares[dominant]
    result.hhi = hhi
    result.lifecycle_stage = lifecycle_stage
    result.growth_runway_quarters = runway
    result.pricing_power = pp
    result.cannibalization_rate = cannibal_rate
    result.network_effect_score = nfx

    logger.info(
        "Product segments: %d segments, HHI=%.3f, dominant=%s (%.0f%%), "
        "lifecycle=%s, pricing_power=%.3f",
        len(segment_data), hhi, dominant, shares[dominant] * 100,
        lifecycle_stage, pp,
    )

    return cache, result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _classify_lifecycle(series: pd.Series) -> tuple[str, int]:
    """Bass model lifecycle classification."""
    try:
        from scipy.optimize import curve_fit

        cumulative = series.cumsum()
        t = np.arange(len(cumulative))

        def bass_cdf(t, p, q, m):
            exp_term = np.exp(-(p + q) * t)
            return m * (1 - exp_term) / (1 + (q / max(p, 1e-6)) * exp_term)

        popt, _ = curve_fit(
            bass_cdf, t, cumulative.values,
            p0=[0.01, 0.3, float(cumulative.iloc[-1]) * 1.5],
            maxfev=5000,
        )
        p, q, m = popt

        F_current = float(cumulative.iloc[-1]) / m if m > 0 else 0

        if F_current < 0.1:
            runway = int((0.5 * m - cumulative.iloc[-1]) / series.mean()) if series.mean() > 0 else 20
            return "introduction", max(0, runway)
        elif F_current < 0.5:
            peak_t = math.log(q / max(p, 1e-6)) / (p + q) if p > 0 and q > 0 else len(t)
            return "growth", max(0, int(peak_t - len(t)))
        elif F_current < 0.9:
            return "maturity", 0
        else:
            return "decline", 0

    except Exception:
        # Fallback: simple growth rate classification
        if len(series) >= 4:
            prev = float(series.iloc[-4])
            curr = float(series.iloc[-1])
            growth = (curr - prev) / abs(prev) if prev != 0 else 0
            if growth > 0.20:
                return "growth", 8
            elif growth > 0.0:
                return "maturity", 0
            else:
                return "decline", 0
        return "unknown", 0


def _stage_to_numeric(stage: str) -> float:
    return {"introduction": 0.0, "growth": 0.25, "maturity": 0.5, "decline": 0.75, "unknown": 0.5}.get(stage, 0.5)


def _compute_mix_effect(segment_data: dict[str, pd.Series]) -> float:
    """M3: Gross margin bridge mix effect proxy."""
    if len(segment_data) < 2:
        return 0.0

    current_revs = {name: float(s.iloc[-1]) for name, s in segment_data.items() if len(s) > 0}
    current_total = sum(current_revs.values())
    if current_total <= 0:
        return 0.0

    growth_weighted = 0.0
    for name, series in segment_data.items():
        if len(series) >= 2:
            share = float(series.iloc[-1]) / current_total
            prev = float(series.iloc[-2])
            growth = (float(series.iloc[-1]) - prev) / abs(prev) if prev != 0 else 0
            growth_weighted += share * growth

    return growth_weighted


def _margin_sustainability(mix_effect: float, cache: pd.DataFrame) -> float:
    """Score how sustainable margin changes are."""
    if "gross_margin" not in cache.columns:
        return 0.5

    gm = cache["gross_margin"].dropna()
    if len(gm) < 63:
        return 0.5

    gm_change = float(gm.iloc[-1]) - float(gm.iloc[-63])

    if abs(gm_change) < 0.01:
        return 0.5

    if gm_change > 0:
        return min(1.0, 0.5 + mix_effect * 5) if mix_effect > 0 else max(0.0, 0.5 - abs(mix_effect) * 5)
    return max(0.0, 0.3 - abs(gm_change) * 2)


def _compute_pricing_power(cache: pd.DataFrame, macro_data: dict, sector: str) -> float:
    """M6: Pricing power = real revenue growth vs industry PPI."""
    if "revenue" not in cache.columns:
        return 0.0

    rev = cache["revenue"].dropna()
    if len(rev) < 252:
        return 0.0

    prev_rev = float(rev.iloc[-252])
    if prev_rev == 0:
        return 0.0
    rev_growth = (float(rev.iloc[-1]) - prev_rev) / abs(prev_rev)

    ppi_growth = 0.03  # default 3%
    inf_series = macro_data.get("inflation")
    if inf_series is not None and not inf_series.empty:
        ppi_growth = float(inf_series.iloc[-1]) / 100.0

    if rev_growth == 0:
        return 0.0
    return (rev_growth - ppi_growth) / abs(rev_growth)


def _compute_input_cost_pressure(cache: pd.DataFrame, macro_data: dict, sector: str) -> float:
    """M9: Input cost pressure from relevant commodity indices."""
    inf_series = macro_data.get("inflation")
    if inf_series is None or inf_series.empty or len(inf_series) < 2:
        return 0.0

    first = float(inf_series.iloc[0])
    last = float(inf_series.iloc[-1])
    if first == 0:
        return 0.0
    trend = (last - first) / abs(first)
    return max(0.0, min(1.0, trend))


def _detect_cannibalization(segment_data: dict[str, pd.Series]) -> dict:
    """M11: Detect if newer segments eat older segment revenue."""
    lengths = {name: len(s) for name, s in segment_data.items()}
    newest = min(lengths, key=lengths.get)
    oldest = [n for n in segment_data if n != newest]

    if not oldest or len(segment_data[newest]) < 4:
        return {"rate": 0.0, "net_new_pct": 1.0}

    new_rev = float(segment_data[newest].iloc[-4:].sum())

    old_decline = 0.0
    for name in oldest:
        s = segment_data[name]
        if len(s) >= 8:
            pre = float(s.iloc[-8:-4].mean())
            post = float(s.iloc[-4:].mean())
            if pre > post:
                old_decline += (pre - post) * 4

    rate = old_decline / new_rev if new_rev > 0 else 0.0
    return {"rate": min(1.0, rate), "net_new_pct": max(0.0, 1.0 - rate)}


def _estimate_network_effect(cache: pd.DataFrame) -> float:
    """M12: Network effect from revenue acceleration + margin stability."""
    if "revenue" not in cache.columns or "gross_margin" not in cache.columns:
        return 0.0

    rev = cache["revenue"].dropna()
    if len(rev) < 126:
        return 0.0

    rev_growth = rev.pct_change(63).dropna()
    if len(rev_growth) < 2:
        return 0.0
    acceleration = float(rev_growth.iloc[-1]) - float(rev_growth.iloc[0])

    margin = cache["gross_margin"].dropna()
    margin_stable = float(margin.std()) < 0.05 if len(margin) > 20 else True

    if acceleration > 0 and margin_stable:
        return min(1.0, acceleration * 10)
    return 0.0


def _estimate_market_share(company_revenue: float, macro_data: dict, sector: str) -> float:
    """M4: Estimate SOM from company revenue / sector GDP proxy."""
    gdp_series = macro_data.get("gdp")
    if gdp_series is None or gdp_series.empty:
        return 0.0

    gdp = float(gdp_series.iloc[-1])
    if gdp <= 0:
        return 0.0

    SECTOR_GDP_SHARE = {
        "technology": 0.08, "healthcare": 0.07, "financials": 0.08,
        "energy": 0.06, "industrials": 0.06, "consumer discretionary": 0.05,
        "consumer staples": 0.04, "materials": 0.03, "utilities": 0.02,
        "real estate": 0.03, "communication services": 0.04,
        "information technology": 0.08, "consumer": 0.05,
    }

    sector_lower = sector.lower() if sector else ""
    sector_share = 0.05
    for key, share in SECTOR_GDP_SHARE.items():
        if key in sector_lower:
            sector_share = share
            break

    sector_tam = gdp * sector_share * 1e9
    return min(1.0, company_revenue / sector_tam) if sector_tam > 0 else 0.0
