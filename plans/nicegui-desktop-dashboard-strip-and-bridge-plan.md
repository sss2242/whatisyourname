# NiceGUI Desktop Dashboard -- Strip and Bridge Plan (v2)

Replace `run.py` terminal interface with a native desktop dashboard using NiceGUI.
Fixes the rigid flow of run.py with a professional, non-linear UX.

## Problems with run.py's flow

1. **Auto-selects LLM** if any key is in .env -- no way to override or switch
2. **Linear 10-step flow** -- can't go back, can't skip, can't change mind
3. **No state persistence** -- restart = re-enter everything
4. **No results display** -- just runs subprocess and shows stdout
5. **No comparison** -- can't compare two companies side by side
6. **No history** -- previous analyses are lost in cache/ with no way to browse

## Dashboard UX Philosophy

- **Non-linear navigation**: Jump to any page anytime via sidebar
- **Always editable**: LLM provider, market, company can be changed at any point
- **Persistent state**: All settings saved to `cache/dashboard_state.json`
- **Results-first**: Start from the last analysis, not from setup
- **Multi-analysis**: Keep history of past analyses, compare them

## Dashboard Layout

```
+------------------------------------------------------------------+
| OPERATOR 1                           [theme] [health: 23/25 OK]  |
+----------+-------------------------------------------------------+
|          |                                                       |
|  [home]  |                                                       |
|  [new]   |  MAIN CONTENT                                        |
|  [report]|                                                       |
|  [health]|  (changes based on sidebar selection)                 |
|  [config]|                                                       |
|          |                                                       |
+----------+-------------------------------------------------------+
```

## Page Designs

### Startup: Splash Screen with Dependency Loading Bar

When the app launches, show a splash/loading screen that checks all
dependencies with an animated progress bar (not a boring checklist):

```
+-------------------------------------------------------+
|                                                        |
|           OPERATOR 1                                   |
|           Point-in-Time Financial Analysis             |
|           25 markets | $95T+ coverage                  |
|                                                        |
|  Initializing...                                       |
|                                                        |
|  [=================>                    ]  47%          |
|  Checking Stage 2: ML libraries...                     |
|                                                        |
|  Stage 1 (Core)     numpy, pandas, scipy       done    |
|  Stage 2 (ML)       scikit-learn, xgboost...   now     |
|  Stage 3 (DL)       torch, pymc                        |
|  Stage 4 (Wrappers) edgartools, yfinance               |
|                                                        |
+-------------------------------------------------------+
```

After loading completes (100%), auto-transition to Home page.
If any stage fails, show which packages are missing with an
[Install Missing] button that runs pip install in-app.

The progress bar increments in real steps:
- 0-10%: Python version + internet check
- 10-35%: Stage 1 imports (numpy, pandas, scipy, etc.)
- 35-60%: Stage 2 imports (sklearn, statsmodels, xgboost, etc.)
- 60-80%: Stage 3 imports (torch, pymc)
- 80-95%: Stage 4 imports (edgartools, yfinance, etc.)
- 95-100%: Load .env, health status, dashboard state

Each package import is tested individually. The bar moves smoothly
with small increments per package. Failed packages show in red
next to their stage but don't block the app from loading.

### Page 1: Home (default landing page)

Shows the LAST analysis result or a welcome screen if first run.

**If previous analysis exists:**
```
+-------------------------------------------------------+
|  APPLE INC (AAPL)                    US / NYSE         |
|  Technology / Consumer Electronics                     |
|                                                        |
|  +----------+  +----------+  +----------+  +--------+ |
|  | Health   |  | Survival |  | Regime   |  | MC     | |
|  | Score    |  | Prob     |  | Current  |  | Surv%  | |
|  |   72/100 |  |   0.85   |  | low_vol  |  |  91%   | |
|  | Strong   |  | Stable   |  |          |  |        | |
|  +----------+  +----------+  +----------+  +--------+ |
|                                                        |
|  [Chart: 2yr price + regime bands]                     |
|                                                        |
|  Quick Actions:                                        |
|  [Re-run Analysis] [View Full Report] [New Company]    |
|                                                        |
|  Recent Analyses:                                      |
|  | Company    | Market | Date       | Health | Status | |
|  | AAPL       | US     | 2026-03-23 | 72     | Done   | |
|  | Samsung    | KR     | 2026-03-22 | 65     | Done   | |
|  | Petrobras  | BR     | 2026-03-20 | 48     | Done   | |
+-------------------------------------------------------+
```

**If first run (no history):**
```
+-------------------------------------------------------+
|  Welcome to Operator 1                                 |
|  Point-in-Time Financial Analysis                      |
|                                                        |
|  25 markets | $95T+ coverage | 25+ math models        |
|                                                        |
|  [Start New Analysis]  [Configure API Keys]            |
|                                                        |
|  Quick health: 23/25 markets healthy                   |
+-------------------------------------------------------+
```

### Page 2: New Analysis (replaces run.py Steps 3-10)

**Non-linear form -- all fields visible, all editable, no forced order:**

```
+-------------------------------------------------------+
|  NEW ANALYSIS                                          |
|                                                        |
|  LLM Provider    [v Gemini     ]  Model [v gemini-2.0-flash]  |
|  (All detected keys shown. User ALWAYS picks.)         |
|  Available: [x] Gemini (2 keys)  [ ] Claude  [ ] OpenRouter   |
|                                                        |
|  --- Market Selection ---                              |
|  Region  [v Asia          ]                            |
|  Market  [v Japan - JPX   ]                            |
|                                                        |
|  --- Company ---                                       |
|  Company [Toyota Motor     ]  [Search]                |
|  Status: Found via J-Quants (7203)                    |
|                                                        |
|  --- Options ---                                       |
|  [x] Discover linked entities (competitors, suppliers) |
|  [x] Run temporal models (forecasting, burn-out)       |
|  [ ] Generate PDF report                               |
|  Data mode: (o) Standard  ( ) Enhanced (API keys)      |
|                                                        |
|  Estimated time: ~30-60 minutes                        |
|                                                        |
|  [Run Analysis]                                        |
|                                                        |
|  --- Progress (visible during run) ---                 |
|  Step 3/10: Extracting financial data...  [=====>  ]   |
|  Log: Fetching income statement for 7203...            |
+-------------------------------------------------------+
```

**Key UX improvements over run.py:**
- LLM provider is ALWAYS a dropdown -- never auto-selected
- All available keys shown with checkboxes (can have multiple active)
- Model selection dropdown populated from provider's model list
- Region/Market/Company are independent dropdowns (no forced order)
- All options visible at once (no "press Y/N" prompts)
- Progress section appears inline when analysis starts
- Can navigate away during analysis (it runs in background)
- Can cancel a running analysis

### Page 3: Report Viewer

**Interactive report with embedded charts (not just markdown text):**

```
+-------------------------------------------------------+
|  ANALYSIS REPORT: Toyota Motor (7203)                  |
|  Generated: 2026-03-23 | Tier: [Basic|Pro|Premium]    |
|                                                        |
|  [Download MD] [Download PDF] [Download JSON]          |
|                                                        |
|  Tabs: [Summary] [Financials] [Charts] [Full Report]  |
|                                                        |
|  --- Summary Tab ---                                   |
|  Executive summary text from LLM...                    |
|                                                        |
|  --- Financials Tab ---                                |
|  Interactive table: balance sheet / income / cashflow  |
|  Sort by column, filter by period                      |
|                                                        |
|  --- Charts Tab ---                                    |
|  [Candlestick]  2yr OHLCV with regime color bands     |
|  [Health Gauge]  Composite score: 72/100 Strong        |
|  [Radar Chart]   5-tier breakdown (liquidity, debt...) |
|  [Survival]      MC probability distribution           |
|  [Peer Compare]  Bar chart vs competitors              |
|                                                        |
|  --- Full Report Tab ---                               |
|  Complete markdown rendered as HTML                    |
+-------------------------------------------------------+
```

### Page 4: Health Monitor (Professional Dashboard)

Full-featured monitoring dashboard with real-time probing, history,
and drill-down per market. Three sections: Overview, Market Grid, Detail.

```
+-------------------------------------------------------+
|  SYSTEM HEALTH                                         |
|                                                        |
|  +---------------------------------------------------+|
|  | OVERVIEW BAR                                       ||
|  |                                                    ||
|  | [=====23 Healthy=====][~~2 Degraded~~][           ]||
|  | 92% operational                    Last: 2h ago    ||
|  |                                    [Run Full Check]||
|  +---------------------------------------------------+|
|                                                        |
|  Tabs: [Markets] [History] [Alerts] [Dependencies]     |
|                                                        |
|  --- Markets Tab: Interactive Grid ---                 |
|                                                        |
|  TIER 1 MARKETS                                        |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|  | US    | | UK    | | EU    | | FR    | | DE    |    |
|  | EDGAR | | CH    | | ESEF  | | ESEF  | | BA    |    |
|  | [===] | | [===] | | [===] | | [===] | | [== ] |    |
|  | L3 OK | | L3 OK | | L3 OK | | L3 OK | | L3 fb |    |
|  | 1.2s  | | 2.1s  | | 3.4s  | | 2.8s  | | 4.1s  |    |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|  | JP    | | KR    | | TW    | | BR    | | CL    |    |
|  | JQnts | | DART  | | MOPS  | | CVM   | | ADR   |    |
|  | [===] | | [===] | | [===] | | [===] | | [== ] |    |
|  | L3 OK | | L3 OK | | L3 OK | | L3 OK | | L3 fb |    |
|  | 1.8s  | | 2.5s  | | 3.1s  | | 2.2s  | | 1.5s  |    |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|                                                        |
|  TIER 2 MARKETS                                        |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|  | IN    | | CN    | | HK    | | CA    | | AU    |    |
|  | BSE   | | aksh  | | EM    | | TMX   | | ASX   |    |
|  | [===] | | [===] | | [===] | | [===] | | [===] |    |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|  | SG    | | SA    | | CH    | | ZA    | | MX    |    |
|  | SGX   | | Tadwl | | SIX   | | JSE   | | BMV   |    |
|  | [===] | | [===] | | [===] | | [===] | | [===] |    |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|  | AE    | | NL    | | ES    | | IT    | | SE    |    |
|  | DFM   | | ESEF  | | ESEF  | | ESEF  | | ESEF  |    |
|  | [===] | | [===] | | [===] | | [===] | | [===] |    |
|  +-------+ +-------+ +-------+ +-------+ +-------+    |
|                                                        |
|  Click any market card to drill down:                  |
|                                                        |
|  --- Drill-down: DE (Germany) ---                      |
|  Status:      DEGRADED (using fallback)                |
|  Primary:     filings.xbrl.org -> 0 filings            |
|  Fallback:    Bundesanzeiger ONNX captcha -> WORKING   |
|  Active path: bundesanzeiger_onnx                      |
|  Latency:     4.1s (captcha solve + HTML parse)        |
|  Last OK:     2026-03-23 18:45                         |
|  Failures:    0 consecutive                            |
|  Probe log:   L0 OK -> L1 OK -> L2 OK -> L3 OK (fb)   |
|                                                        |
|  --- History Tab ---                                   |
|  ECharts line chart: healthy/degraded/critical count   |
|  over past 30 days (from health_history.jsonl)         |
|                                                        |
|  --- Alerts Tab ---                                    |
|  Timeline of status changes:                           |
|  [!] 2026-03-23: DE healthy -> degraded                |
|  [!] 2026-03-21: CL healthy -> degraded                |
|  [+] 2026-03-20: SA degraded -> healthy                |
|                                                        |
|  --- Dependencies Tab ---                              |
|  Stage 1 (Core):     [====================] 100%       |
|  Stage 2 (ML):       [====================] 100%       |
|  Stage 3 (DL):       [====================] 100%       |
|  Stage 4 (Wrappers): [====================] 100%       |
|                                                        |
|  Package          Version   Status                     |
|  numpy            2.4.2     OK                         |
|  pandas           2.3.3     OK                         |
|  torch            2.10.0    OK                         |
|  scikit-learn     1.8.0     OK                         |
|  edgartools       5.19.1    OK                         |
|  ...                                                   |
+-------------------------------------------------------+
```

**Card color coding:**
- Green background + full bar `[===]` = healthy (L3 primary)
- Yellow background + partial bar `[== ]` = degraded (L3 via fallback)
- Red background + empty bar `[   ]` = critical (all paths failing)
- Gray = unknown / not checked

**Real-time probing:**
- Click [Run Full Check] -> progress bar sweeps across all 25 markets
- Each card updates live as its probe completes (green flash = OK, red flash = fail)
- Total check takes ~30-60s (L0-L3 on all 25 markets)

### Page 5: Configuration

```
+-------------------------------------------------------+
|  CONFIGURATION                                         |
|                                                        |
|  --- API Keys ---                                      |
|  GEMINI_API_KEY     [AIza...****]  [x] Active  [Edit]  |
|  ANTHROPIC_API_KEY  [sk-a...****]  [ ] Active  [Edit]  |
|  OPENROUTER_API_KEY [not set    ]  [ ] Active  [Set]   |
|  EDGAR_IDENTITY     [user@email]   [x] Active  [Edit]  |
|  ...                                                   |
|                                                        |
|  --- Pipeline Settings ---                             |
|  HTTP timeout:      [30] seconds                       |
|  Max retries:       [5]                                |
|  Cache TTL:         [24] hours                         |
|  Estimation method: [v split (recommended)]            |
|                                                        |
|  --- Cache ---                                         |
|  Cache size: 234 MB  [Clear Cache]  [Clear Reports]    |
|                                                        |
|  --- About ---                                         |
|  Operator 1 v1.0                                       |
|  25 markets | 10 Tier 1 + 15 Tier 2                    |
|  Python 3.12.3 | All dependencies OK                   |
+-------------------------------------------------------+
```

## Implementation Steps

### Step 1: Install + skeleton + splash screen
- [ ] `pip install nicegui`
- [ ] Create `dashboard.py` with app shell
- [ ] Splash screen with animated progress bar (dependency loading)
- [ ] Per-package import testing with real-time bar advancement (0-100%)
- [ ] Failed packages shown in red, [Install Missing] button
- [ ] Auto-transition to Home after 100%
- [ ] Sidebar navigation with 5 pages
- [ ] Dark theme by default (financial apps are dark)
- [ ] `ui.run(native=True, title='Operator 1', window_size=(1400, 900))`

### Step 2: State management
- [ ] Create `DashboardState` class in `operator1/dashboard/state.py`
- [ ] Auto-load .env keys on startup (but NEVER auto-select LLM)
- [ ] Persist user choices to `cache/dashboard_state.json`
- [ ] Track analysis history in `cache/analysis_history.json`

### Step 3: Home page
- [ ] Load last analysis from cache
- [ ] Summary cards (health score, survival, regime, MC)
- [ ] Recent analyses table with clickable rows
- [ ] Welcome screen for first-time users
- [ ] Quick actions: Re-run, View Report, New Company

### Step 4: New Analysis page
- [ ] LLM provider dropdown (NEVER auto-select -- always show all available)
- [ ] All detected keys shown as checkboxes (user picks active provider)
- [ ] Model selector dropdown (populated from provider's registry)
- [ ] Region -> Market cascading dropdowns (from pit_registry)
- [ ] Company search input with live validation indicator
- [ ] Pipeline options as toggle switches (not Y/N prompts)
- [ ] Data mode radio: Standard / Enhanced
- [ ] Runtime estimate display
- [ ] Run button -> background thread with subprocess
- [ ] Inline progress bar + live log scroll area
- [ ] Cancel button to abort running pipeline

### Step 5: Report page
- [ ] Tab layout: Summary / Financials / Charts / Full Report
- [ ] ECharts candlestick from OHLCV cache data
- [ ] ECharts gauge for financial health composite score
- [ ] ECharts radar chart for 5-tier breakdown
- [ ] ECharts bar chart for peer comparison
- [ ] AG Grid table for financial statements (sortable, filterable)
- [ ] Markdown renderer for full report text
- [ ] Report tier selector: Basic / Pro / Premium
- [ ] Download buttons: Markdown, PDF, JSON profile

### Step 6: Health Monitor page (professional dashboard)
- [ ] Overview bar: colored segments (green/yellow/red) with percentage
- [ ] 4-tab layout: Markets / History / Alerts / Dependencies
- [ ] Markets tab: 5x5 card grid, Tier 1 + Tier 2 sections
- [ ] Card color coding: green=healthy, yellow=degraded, red=critical
- [ ] Mini progress bar per card showing probe level passed
- [ ] Click card -> drill-down panel (status, primary/fallback paths, latency, probe log)
- [ ] [Run Full Check] button: live sweep across all 25 cards
- [ ] Each card flashes green/red as its probe completes in real-time
- [ ] History tab: ECharts line chart of healthy/degraded/critical over 30 days
- [ ] Alerts tab: timeline of status changes with icons
- [ ] Dependencies tab: 4 stage progress bars (all at 100% if OK)
- [ ] Per-package table: name, version, status (from splash screen data)

### Step 7: Config page
- [ ] API key management (show masked values, edit inline, save to .env)
- [ ] Pipeline settings from `global_config.yml` (editable)
- [ ] Cache management: size display + [Clear Cache] + [Clear Reports]
- [ ] About section: version, market count, Python version

### Step 8: Packaging
- [ ] PyInstaller spec file for single binary
- [ ] Linux .desktop file for app menu integration
- [ ] `build_desktop.py` build script
- [ ] Test on Linux Mint
