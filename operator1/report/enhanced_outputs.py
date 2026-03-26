"""Enhanced report outputs using community packages.

Generates three additional output formats alongside the core Markdown report:

1. **fpdf2 PDF** -- Pure-Python PDF with branded styling (no pandoc needed)
2. **mplfinance charts** -- Professional candlestick charts with volume/MA/regime
3. **quantstats tearsheet** -- Performance analytics HTML with 40+ metrics
4. **plotly interactive HTML** -- Dashboard with zoom/pan/hover on all charts

All functions are optional: if a package is missing, the function logs a
warning and returns None. The main report_generator.py calls these as
supplementary outputs after the core Markdown is built.

Top-level entry points:
    ``generate_fpdf2_pdf()`` -- branded PDF from Markdown + chart images
    ``generate_mplfinance_charts()`` -- candlestick/price charts
    ``generate_quantstats_tearsheet()`` -- performance tearsheet HTML
    ``generate_plotly_dashboard()`` -- interactive HTML dashboard
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brand palette (shared with report_generator.py)
# ---------------------------------------------------------------------------

_BG = "#1c1b22"
_FG = "#eae7e1"
_GRID = "#2d2b33"
_ACCENT = "#6d4aff"
_RED = "#dc3545"
_GREEN = "#1ea885"
_GOLD = "#e8950a"


# ---------------------------------------------------------------------------
# 1. fpdf2 Pure-Python PDF
# ---------------------------------------------------------------------------


def generate_fpdf2_pdf(
    markdown: str,
    chart_paths: list[str],
    output_path: str | Path,
    profile: dict[str, Any],
) -> str | None:
    """Generate a branded PDF using fpdf2 (no pandoc/weasyprint needed).

    Parameters
    ----------
    markdown:
        Full Markdown report text.
    chart_paths:
        List of chart PNG file paths.
    output_path:
        Output PDF file path.
    profile:
        Company profile dict (for metadata).

    Returns
    -------
    PDF file path on success, None on failure.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.info("fpdf2 not installed; skipping pure-Python PDF generation.")
        return None

    try:
        identity = profile.get("identity", {})
        company = identity.get("name", "Company Analysis")
        ticker = identity.get("ticker", "")
        generated = profile.get("meta", {}).get(
            "generated_at", datetime.now(timezone.utc).isoformat()
        )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.set_margins(15, 15, 15)

        # -- Cover page --
        pdf.add_page()
        pdf.set_fill_color(28, 27, 34)  # _BG
        pdf.rect(0, 0, 210, 297, "F")
        pdf.set_text_color(234, 231, 225)  # _FG
        pdf.set_font("Helvetica", "B", 28)
        pdf.ln(80)
        pdf.cell(0, 15, "OPERATOR 1", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 8, "Point-in-Time Financial Analysis", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(20)
        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 12, f"{company} ({ticker})", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {generated}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, "25 markets | $91T+ coverage | 45+ models", align="C", new_x="LMARGIN", new_y="NEXT")

        # -- Content pages --
        pdf.add_page()
        pdf.set_fill_color(255, 255, 255)
        pdf.rect(0, 0, 210, 297, "F")
        pdf.set_text_color(30, 30, 30)

        # Parse Markdown into sections
        sections = markdown.split("\n## ")
        for i, section in enumerate(sections):
            if i == 0:
                # First section is everything before the first ##
                lines = section.strip().split("\n")
                # Skip the title line (already on cover)
                content_lines = [l for l in lines if not l.startswith("# ")]
                if content_lines:
                    pdf.set_font("Helvetica", "", 9)
                    text = "\n".join(content_lines).strip()
                    if text:
                        pdf.multi_cell(0, 4, text)
                        pdf.ln(3)
                continue

            lines = section.strip().split("\n", 1)
            heading = lines[0].strip().rstrip("#").strip()
            body = lines[1].strip() if len(lines) > 1 else ""

            # Section heading
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(109, 74, 255)  # _ACCENT
            pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(109, 74, 255)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(3)

            # Section body
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(30, 30, 30)

            # Process body: handle tables, bold, etc.
            for line in body.split("\n"):
                stripped = line.strip()
                if not stripped:
                    pdf.ln(2)
                    continue

                # Skip image references and table separators
                if stripped.startswith("![") or stripped.startswith("|--"):
                    continue

                # Check if near page bottom before rendering
                if pdf.get_y() > 270:
                    pdf.add_page()

                try:
                    # Bold lines
                    if stripped.startswith("**") and stripped.endswith("**"):
                        pdf.set_font("Helvetica", "B", 9)
                        pdf.multi_cell(0, 4, stripped.strip("*"))
                        pdf.set_font("Helvetica", "", 9)
                    # Table rows
                    elif stripped.startswith("|"):
                        cells = [c.strip() for c in stripped.split("|")[1:-1]]
                        if cells:
                            n_cells = max(len(cells), 1)
                            col_w = max(180 / n_cells, 20)
                            col_w = min(col_w, 60)
                            max_chars = max(int(col_w / 2.5), 8)
                            pdf.set_font("Helvetica", "", 7)
                            for cell in cells:
                                text = cell[:max_chars] if len(cell) > max_chars else cell
                                pdf.cell(col_w, 4, text, border=1)
                            pdf.ln()
                            pdf.set_font("Helvetica", "", 9)
                    # Bullet points
                    elif stripped.startswith("- "):
                        pdf.multi_cell(0, 4, f"    {stripped}")
                    # Horizontal rules
                    elif stripped == "---":
                        pdf.ln(2)
                        pdf.set_draw_color(200, 200, 200)
                        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
                        pdf.ln(3)
                    else:
                        pdf.multi_cell(0, 4, stripped)
                except Exception:
                    # Skip any line that causes rendering issues
                    pass

        # -- Chart pages --
        for chart_path in chart_paths:
            p = Path(chart_path)
            if p.exists() and p.suffix.lower() == ".png":
                pdf.add_page()
                pdf.set_fill_color(28, 27, 34)
                pdf.rect(0, 0, 210, 297, "F")
                title = p.stem.replace("_", " ").title()
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(234, 231, 225)
                pdf.cell(0, 10, title, align="C", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)
                # Embed chart image (full width)
                pdf.image(str(p), x=10, w=190)

        # -- Save --
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(out))
        logger.info("fpdf2 PDF generated: %s (%d pages)", out, pdf.pages_count)
        return str(out)

    except Exception as exc:
        logger.warning("fpdf2 PDF generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 2. mplfinance Professional Charts
# ---------------------------------------------------------------------------


def generate_mplfinance_charts(
    cache: pd.DataFrame,
    profile: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Generate professional candlestick charts using mplfinance.

    Replaces manual matplotlib candlestick drawing with mplfinance's
    native OHLCV charting. Produces branded charts with volume overlay,
    moving averages, and regime shading.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with DatetimeIndex.
    profile:
        Company profile dict.
    output_dir:
        Directory for chart PNG files.

    Returns
    -------
    List of generated chart file paths.
    """
    try:
        import mplfinance as mpf
    except ImportError:
        logger.info("mplfinance not installed; skipping enhanced chart generation.")
        return []

    if cache is None or cache.empty:
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    chart_paths: list[str] = []

    identity = profile.get("identity", {})
    company = identity.get("name", identity.get("ticker", ""))

    # Check for OHLCV data
    has_ohlcv = all(c in cache.columns for c in ("open", "high", "low", "close"))
    if not has_ohlcv:
        logger.info("mplfinance: no OHLCV columns in cache, skipping candlestick charts.")
        return []

    # Ensure DatetimeIndex
    if not isinstance(cache.index, pd.DatetimeIndex):
        try:
            cache.index = pd.to_datetime(cache.index)
        except Exception:
            logger.warning("mplfinance: cannot convert index to DatetimeIndex.")
            return []

    # Custom Operator 1 brand style
    try:
        mc = mpf.make_marketcolors(
            up=_GREEN, down=_RED,
            edge={"up": _GREEN, "down": _RED},
            wick={"up": _GREEN, "down": _RED},
            volume=_ACCENT,
            ohlc=_FG,
        )
        op1_style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            marketcolors=mc,
            facecolor=_BG,
            edgecolor=_GRID,
            figcolor=_BG,
            gridcolor=_GRID,
            gridstyle="--",
            y_on_right=True,
            rc={
                "axes.labelcolor": _FG,
                "xtick.color": _FG,
                "ytick.color": _FG,
            },
        )
    except Exception:
        op1_style = "nightclouds"

    # -- Chart A: Full price history as candlestick --
    try:
        ohlcv = cache[["open", "high", "low", "close"]].copy()
        if "volume" in cache.columns:
            ohlcv["volume"] = cache["volume"]
            has_vol = True
        else:
            has_vol = False

        ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"])

        if len(ohlcv) > 10:
            save_cfg = dict(fname=str(out / "price_candlestick.png"), dpi=180, pad_inches=0.3)

            # Add moving averages and Bollinger bands if enough data
            addplots = []
            if len(ohlcv) > 50:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    bb_mid = ohlcv["close"].rolling(20).mean()
                    bb_std = ohlcv["close"].rolling(20).std()
                    bb_upper = bb_mid + 2 * bb_std
                    bb_lower = bb_mid - 2 * bb_std
                    addplots.append(mpf.make_addplot(bb_upper, color=_GOLD, width=0.7, linestyle="--"))
                    addplots.append(mpf.make_addplot(bb_lower, color=_GOLD, width=0.7, linestyle="--"))
                except Exception:
                    pass

            mpf.plot(
                ohlcv,
                type="candle",
                style=op1_style,
                volume=has_vol,
                mav=(10, 21, 50) if len(ohlcv) > 50 else (),
                addplot=addplots if addplots else None,
                title=f"\n{company} -- Price History (2Y)",
                savefig=save_cfg,
                figsize=(16, 8),
                tight_layout=True,
            )
            chart_paths.append(str(out / "price_candlestick.png"))
            logger.info("mplfinance: generated price_candlestick.png")

    except Exception as exc:
        logger.warning("mplfinance price chart failed: %s", exc)

    # -- Chart B: Predicted OHLC (next month) --
    try:
        ohlc_data = profile.get("ohlc_predictions", {})
        next_month = ohlc_data.get("next_month", {})
        series = next_month.get("series", [])

        if series and len(series) >= 5:
            pred_df = pd.DataFrame(series)
            required = {"open", "high", "low", "close"}
            if required.issubset(pred_df.columns):
                # Create a date index starting from tomorrow
                last_date = cache.index[-1] if len(cache) > 0 else pd.Timestamp.now()
                pred_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(pred_df))
                pred_df.index = pred_dates

                save_cfg = dict(fname=str(out / "predicted_candles_month.png"), dpi=180, pad_inches=0.3)
                mpf.plot(
                    pred_df[["open", "high", "low", "close"]],
                    type="candle",
                    style=op1_style,
                    title=f"\n{company} -- Predicted Price (Next Month)",
                    savefig=save_cfg,
                    figsize=(14, 7),
                    tight_layout=True,
                )
                chart_paths.append(str(out / "predicted_candles_month.png"))
                logger.info("mplfinance: generated predicted_candles_month.png")

    except Exception as exc:
        logger.warning("mplfinance predicted chart failed: %s", exc)

    return chart_paths


# ---------------------------------------------------------------------------
# 3. quantstats Performance Tearsheet
# ---------------------------------------------------------------------------


def generate_quantstats_tearsheet(
    cache: pd.DataFrame,
    profile: dict[str, Any],
    output_path: str | Path,
) -> str | None:
    """Generate a quantstats performance tearsheet HTML.

    Parameters
    ----------
    cache:
        Daily cache DataFrame with ``return_1d`` column.
    profile:
        Company profile dict.
    output_path:
        Output HTML file path.

    Returns
    -------
    HTML file path on success, None on failure.
    """
    try:
        import quantstats as qs
    except ImportError:
        logger.info("quantstats not installed; skipping tearsheet generation.")
        return None

    if cache is None or cache.empty:
        return None

    # Get daily returns
    if "return_1d" in cache.columns:
        returns = cache["return_1d"].dropna()
    elif "close" in cache.columns:
        returns = cache["close"].pct_change().dropna()
    else:
        logger.info("quantstats: no return or close data available.")
        return None

    if len(returns) < 20:
        logger.info("quantstats: insufficient data (%d observations).", len(returns))
        return None

    try:
        identity = profile.get("identity", {})
        company = identity.get("name", "Company")
        ticker = identity.get("ticker", "")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Extend pandas with quantstats
        qs.extend_pandas()

        # Generate full tearsheet HTML
        qs.reports.html(
            returns,
            benchmark=None,
            output=str(out),
            title=f"{company} ({ticker}) -- Performance Tearsheet",
            download_filename=f"{ticker}_tearsheet.html",
        )

        logger.info("quantstats tearsheet generated: %s (%d days)", out, len(returns))
        return str(out)

    except Exception as exc:
        logger.warning("quantstats tearsheet generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 4. Plotly Interactive Dashboard
# ---------------------------------------------------------------------------


def generate_plotly_dashboard(
    cache: pd.DataFrame,
    profile: dict[str, Any],
    output_path: str | Path,
) -> str | None:
    """Generate an interactive Plotly HTML dashboard.

    Creates a self-contained HTML file with interactive charts:
    - Candlestick with volume (zoom/pan/hover)
    - Financial health radar (5 tier scores)
    - Survival timeline
    - Conflict risk gauge

    Parameters
    ----------
    cache:
        Daily cache DataFrame.
    profile:
        Company profile dict.
    output_path:
        Output HTML file path.

    Returns
    -------
    HTML file path on success, None on failure.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.info("plotly not installed; skipping interactive dashboard.")
        return None

    if cache is None or cache.empty:
        return None

    try:
        identity = profile.get("identity", {})
        company = identity.get("name", "Company")
        ticker = identity.get("ticker", "")

        figs: list[go.Figure] = []

        # Ensure DatetimeIndex
        if not isinstance(cache.index, pd.DatetimeIndex):
            try:
                cache.index = pd.to_datetime(cache.index)
            except Exception:
                pass

        # -- Panel 1: Interactive Candlestick --
        has_ohlcv = all(c in cache.columns for c in ("open", "high", "low", "close"))
        if has_ohlcv:
            ohlcv = cache[["open", "high", "low", "close"]].dropna()
            if len(ohlcv) > 5:
                fig_candle = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                )
                fig_candle.add_trace(
                    go.Candlestick(
                        x=ohlcv.index,
                        open=ohlcv["open"], high=ohlcv["high"],
                        low=ohlcv["low"], close=ohlcv["close"],
                        name="Price",
                        increasing_line_color=_GREEN,
                        decreasing_line_color=_RED,
                    ),
                    row=1, col=1,
                )
                if "volume" in cache.columns:
                    vol = cache["volume"].reindex(ohlcv.index)
                    colors = [_GREEN if c >= o else _RED
                              for o, c in zip(ohlcv["open"], ohlcv["close"])]
                    fig_candle.add_trace(
                        go.Bar(x=ohlcv.index, y=vol, name="Volume",
                               marker_color=colors, opacity=0.5),
                        row=2, col=1,
                    )
                fig_candle.update_layout(
                    title=f"{company} ({ticker}) -- Interactive Price Chart",
                    template="plotly_dark",
                    paper_bgcolor=_BG, plot_bgcolor=_BG,
                    xaxis_rangeslider_visible=False,
                    height=600,
                )
                figs.append(fig_candle)

        # -- Panel 2: Financial Health Radar --
        fh = profile.get("financial_health", {})
        tier_means = fh.get("tier_means", {})
        if tier_means:
            categories = ["Liquidity", "Solvency", "Stability", "Profitability", "Growth"]
            values = [
                tier_means.get("fh_liquidity_score", 0) or 0,
                tier_means.get("fh_solvency_score", 0) or 0,
                tier_means.get("fh_stability_score", 0) or 0,
                tier_means.get("fh_profitability_score", 0) or 0,
                tier_means.get("fh_growth_score", 0) or 0,
            ]
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor=f"rgba(109, 74, 255, 0.2)",
                line=dict(color=_ACCENT, width=2),
                name="Health Score",
            ))
            fig_radar.update_layout(
                title="Financial Health Radar (0-100 per tier)",
                template="plotly_dark",
                paper_bgcolor=_BG, plot_bgcolor=_BG,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color=_FG),
                    angularaxis=dict(color=_FG),
                    bgcolor=_BG,
                ),
                height=500,
            )
            figs.append(fig_radar)

        # -- Panel 3: Survival Timeline --
        if "company_survival_mode_flag" in cache.columns:
            fig_surv = go.Figure()
            flag = cache["company_survival_mode_flag"].fillna(0)
            fig_surv.add_trace(go.Scatter(
                x=cache.index, y=flag,
                fill="tozeroy", fillcolor=f"rgba(220, 53, 69, 0.3)",
                line=dict(color=_RED, width=1),
                name="Company Distress",
            ))
            if "country_survival_mode_flag" in cache.columns:
                cflag = cache["country_survival_mode_flag"].fillna(0)
                fig_surv.add_trace(go.Scatter(
                    x=cache.index, y=cflag + 1.1,
                    fill="tozeroy", fillcolor=f"rgba(232, 149, 10, 0.3)",
                    line=dict(color=_GOLD, width=1),
                    name="Country Crisis",
                ))
            fig_surv.update_layout(
                title="Survival Mode Timeline",
                template="plotly_dark",
                paper_bgcolor=_BG, plot_bgcolor=_BG,
                yaxis=dict(visible=False),
                height=300,
            )
            figs.append(fig_surv)

        # -- Panel 4: Conflict Risk Gauge --
        conflict = profile.get("conflict_risk", {})
        intensity = conflict.get("conflict_intensity_score", 0) or 0
        if isinstance(intensity, (int, float)):
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(intensity),
                title={"text": "Conflict Risk Intensity"},
                gauge=dict(
                    axis=dict(range=[0, 1], tickcolor=_FG),
                    bar=dict(color=_ACCENT),
                    steps=[
                        dict(range=[0, 0.3], color=_GREEN),
                        dict(range=[0.3, 0.6], color=_GOLD),
                        dict(range=[0.6, 1.0], color=_RED),
                    ],
                    threshold=dict(
                        line=dict(color=_FG, width=2),
                        thickness=0.8,
                        value=float(intensity),
                    ),
                ),
                number=dict(font=dict(color=_FG)),
            ))
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor=_BG, plot_bgcolor=_BG,
                height=300,
            )
            figs.append(fig_gauge)

        if not figs:
            logger.info("plotly: no charts to generate (insufficient data).")
            return None

        # -- Combine into single HTML --
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Build HTML with all figures
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>{company} ({ticker}) -- Interactive Dashboard</title>",
            '<meta charset="utf-8">',
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "<style>",
            f"body {{ background: {_BG}; color: {_FG}; font-family: Helvetica, sans-serif; margin: 20px; }}",
            f"h1 {{ color: {_ACCENT}; text-align: center; }}",
            f"h2 {{ color: {_FG}; border-bottom: 1px solid {_GRID}; padding-bottom: 8px; }}",
            ".chart-container { margin: 20px 0; }",
            "</style>",
            "</head><body>",
            f"<h1>OPERATOR 1 -- {company} ({ticker})</h1>",
            f"<p style='text-align:center; color:{_GRID}'>Interactive Financial Analysis Dashboard</p>",
        ]

        for i, fig in enumerate(figs):
            div_id = f"chart_{i}"
            html_parts.append(f'<div class="chart-container" id="{div_id}"></div>')
            # Get plotly JSON
            fig_json = fig.to_json()
            html_parts.append(f"<script>Plotly.newPlot('{div_id}', {fig_json});</script>")

        html_parts.extend([
            f"<p style='text-align:center; color:{_GRID}; margin-top:40px; font-size:11px;'>",
            "Generated by Operator 1 Pipeline | Point-in-Time Financial Analysis",
            "</p>",
            "</body></html>",
        ])

        out.write_text("\n".join(html_parts), encoding="utf-8")
        logger.info("plotly dashboard generated: %s (%d charts)", out, len(figs))
        return str(out)

    except Exception as exc:
        logger.warning("plotly dashboard generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Unified enhanced output generator
# ---------------------------------------------------------------------------


def generate_enhanced_outputs(
    markdown: str,
    cache: pd.DataFrame | None,
    profile: dict[str, Any],
    chart_paths: list[str],
    output_dir: str | Path,
    *,
    generate_pdf: bool = True,
    generate_tearsheet: bool = True,
    generate_interactive: bool = True,
    generate_enhanced_charts: bool = True,
) -> dict[str, Any]:
    """Generate all enhanced report outputs.

    Called from report_generator.py after the core Markdown report is built.

    Returns
    -------
    Dict with paths to generated files (None if not generated).
    """
    out = Path(output_dir)
    result: dict[str, Any] = {
        "fpdf2_pdf_path": None,
        "tearsheet_path": None,
        "interactive_path": None,
        "enhanced_chart_paths": [],
    }

    # Enhanced charts (mplfinance)
    if generate_enhanced_charts and cache is not None:
        enhanced_charts = generate_mplfinance_charts(
            cache, profile, out / "charts",
        )
        result["enhanced_chart_paths"] = enhanced_charts

    # Pure-Python PDF (fpdf2)
    if generate_pdf:
        all_charts = chart_paths + result.get("enhanced_chart_paths", [])
        result["fpdf2_pdf_path"] = generate_fpdf2_pdf(
            markdown, all_charts, out / "report.pdf", profile,
        )

    # Performance tearsheet (quantstats)
    if generate_tearsheet and cache is not None:
        result["tearsheet_path"] = generate_quantstats_tearsheet(
            cache, profile, out / "tearsheet.html",
        )

    # Interactive dashboard (plotly)
    if generate_interactive and cache is not None:
        result["interactive_path"] = generate_plotly_dashboard(
            cache, profile, out / "interactive_dashboard.html",
        )

    return result
