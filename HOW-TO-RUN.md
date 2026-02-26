# How to Run Operator 1

Step-by-step instructions for running Operator 1 on Linux Mint and Windows.

---

## Section 1: Linux Mint (and Ubuntu/Debian-based distros)

### Prerequisites

You need Python 3.10 or newer and git. Linux Mint ships with Python, but you may need to install `pip` and `venv`.

Open a terminal (Ctrl+Alt+T) and run:

```bash
# Check your Python version (need 3.10+)
python3 --version

# Install pip and venv if not already present
sudo apt update
sudo apt install python3-pip python3-venv git -y
```

### Step 1: Clone the repository

```bash
cd ~
git clone https://github.com/Abdu2424/Op-1.git
cd Op-1
```

### Step 2: Create a virtual environment

A virtual environment keeps Operator 1's dependencies isolated from your system Python.

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt. Every time you open a new terminal to run Operator 1, you need to activate it again with `source venv/bin/activate`.

### Step 3: Install dependencies

```bash
pip install --timeout 300 -r requirements.txt
```

This will download and install all required packages. The first run takes **5-10 minutes** because PyTorch is a large download (~2 GB) and some packages compile C extensions. The `--timeout 300` flag prevents pip from timing out on slow connections. If your connection is particularly slow, you can increase it further (e.g. `--timeout 600`).

If the install stalls or times out, try again with retries:

```bash
pip install --timeout 300 --retries 5 -r requirements.txt
```

Some packages (jquants-api-client, pykrx, sdmx1, python-bcb) require Python 3.10+. If you are on Python 3.9, pip will skip those and the pipeline will gracefully fall back to alternative data sources.

**Lighter install** -- skip deep learning models (LSTM, Transformer) for faster setup:

```bash
pip install --timeout 300 requests pandas numpy pyarrow pyyaml python-dotenv statsmodels scikit-learn ruptures hmmlearn arch xgboost matplotlib yfinance fredapi wbgapi edgartools ixbrl-parse pytest
```

The pipeline will still work -- it gracefully skips models whose dependencies are missing.

### Step 4: Configure API keys (optional)

All government filing APIs are free and need no keys. The only optional key is for Gemini AI report generation:

```bash
cp .env.example .env
nano .env
```

Edit the file and replace `your_gemini_api_key_here` with your actual key from [ai.google.dev](https://ai.google.dev/). Save with Ctrl+O, then exit with Ctrl+X.

If you skip this step, reports will still be generated using a built-in template -- they just won't have the AI-generated narrative.

### Step 5: Run the analysis

**Interactive mode (recommended for first-time users):**

```bash
python3 run.py
```

This will guide you through selecting a region, market, and company with numbered menus.

**Direct command (non-interactive):**

```bash
# Analyze Apple (US market)
python3 main.py --market us_sec_edgar --company AAPL

# Analyze Toyota (Japanese market)
python3 main.py --market jp_edinet --company 7203

# Analyze Samsung (Korean market)
python3 main.py --market kr_dart --company 005930

# Analyze Siemens (European market)
python3 main.py --market eu_esef --company "Siemens"

# Analyze Petrobras (Brazilian market)
python3 main.py --market br_cvm --company "Petrobras"
```

**Quick run (skip heavy models for faster results):**

```bash
python3 main.py --market us_sec_edgar --company AAPL --skip-models
```

**Generate PDF report (requires pandoc):**

```bash
sudo apt install pandoc -y
python3 main.py --market us_sec_edgar --company AAPL --pdf
```

### Step 6: View the results

After the pipeline finishes, your results are in the `cache/` folder:

```bash
# Open the report in your browser
xdg-open cache/report/premium_report.md

# Or read it in the terminal
cat cache/report/premium_report.md

# View the full analysis data
cat cache/company_profile.json | python3 -m json.tool | less
```

Charts are saved as PNG files in `cache/report/charts/`.

### Useful commands

```bash
# See all available markets
python3 main.py --list-markets

# See available regions
python3 main.py --list-regions

# See macro data sources
python3 main.py --list-macro

# Re-generate a report from cached data (without re-fetching)
python3 main.py --report-only

# Verbose mode (debug logging)
python3 main.py --market us_sec_edgar --company AAPL --verbose

# See all options
python3 main.py --help
```

### Deactivating the virtual environment

When you are done:

```bash
deactivate
```

---

## Section 2: Windows

### Prerequisites

You need Python 3.10 or newer and git.

1. **Install Python**: Download from [python.org/downloads](https://www.python.org/downloads/). During installation, check the box that says **"Add Python to PATH"** -- this is important.

2. **Install Git**: Download from [git-scm.com](https://git-scm.com/download/win). Use the default settings during installation.

3. **Open a terminal**: Press `Win+R`, type `cmd`, and press Enter. Or search for "Command Prompt" in the Start menu. You can also use PowerShell or Windows Terminal.

### Step 1: Clone the repository

```cmd
cd %USERPROFILE%
git clone https://github.com/Abdu2424/Op-1.git
cd Op-1
```

### Step 2: Create a virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your prompt. Every time you open a new terminal, activate it again with `venv\Scripts\activate`.

If you get an error about execution policy in PowerShell, run this first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install dependencies

```cmd
pip install --timeout 300 -r requirements.txt
```

This takes **5-10 minutes** on first run (PyTorch is ~2 GB). The `--timeout 300` flag prevents pip from timing out on slow connections. If it stalls, try again with:

```cmd
pip install --timeout 300 --retries 5 -r requirements.txt
```

**Lighter install** -- skip deep learning for faster setup:

```cmd
pip install --timeout 300 requests pandas numpy pyarrow pyyaml python-dotenv statsmodels scikit-learn ruptures hmmlearn arch xgboost matplotlib yfinance fredapi wbgapi edgartools ixbrl-parse pytest
```

### Step 4: Configure API keys (optional)

```cmd
copy .env.example .env
notepad .env
```

Notepad will open. Replace `your_gemini_api_key_here` with your actual key from [ai.google.dev](https://ai.google.dev/). Save and close.

### Step 5: Run the analysis

**Interactive mode:**

```cmd
python run.py
```

**Direct command:**

```cmd
:: Analyze Apple (US market)
python main.py --market us_sec_edgar --company AAPL

:: Analyze Toyota (Japanese market)
python main.py --market jp_edinet --company 7203

:: Analyze Samsung (Korean market)
python main.py --market kr_dart --company 005930

:: Quick run (skip heavy models)
python main.py --market us_sec_edgar --company AAPL --skip-models
```

**Generate PDF report (requires pandoc):**

Download pandoc from [pandoc.org/installing.html](https://pandoc.org/installing.html) and install it. Then:

```cmd
python main.py --market us_sec_edgar --company AAPL --pdf
```

### Step 6: View the results

```cmd
:: Open the report in your default browser
start cache\report\premium_report.md

:: Or open in Notepad
notepad cache\report\premium_report.md

:: View analysis data
type cache\company_profile.json
```

Charts are saved as PNG files in `cache\report\charts\`. Double-click any PNG to open it.

### Useful commands

```cmd
:: See all available markets
python main.py --list-markets

:: See available regions
python main.py --list-regions

:: See macro data sources
python main.py --list-macro

:: Re-generate report from cached data
python main.py --report-only

:: Verbose mode
python main.py --market us_sec_edgar --company AAPL --verbose

:: See all options
python main.py --help
```

### Deactivating the virtual environment

```cmd
deactivate
```

---

## Troubleshooting

### "No module named operator1"
Make sure you are running the command from inside the `Op-1` directory and that your virtual environment is activated.

### "pip: command not found" (Linux)
Run `sudo apt install python3-pip -y`.

### "'python' is not recognized" (Windows)
Python was not added to PATH during installation. Reinstall Python and check the "Add Python to PATH" box, or use the full path: `C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe`.

### pip install times out or hangs
The full requirements include large packages (PyTorch ~2 GB). Use the timeout flag:
```bash
pip install --timeout 300 --retries 5 -r requirements.txt
```
If it still fails, use the lighter install command shown in Step 3 -- it skips PyTorch and installs only the core packages needed for financial analysis.

### PyTorch install fails or takes too long
PyTorch is optional. Skip it and use the lighter install command shown above. The pipeline will fall back to statistical models (Kalman, GARCH, tree ensembles).

### "GEMINI_API_KEY not set"
This is a warning, not an error. Reports will still be generated using the built-in template. If you want AI-generated narratives, get a free key from [ai.google.dev](https://ai.google.dev/).

### Rate limiting / 429 errors
Some APIs have rate limits. The app handles this with automatic exponential backoff (2s, 4s, 8s, 16s, 32s). If it keeps failing, wait a few minutes and try again.

### Analysis takes a long time
Use `--skip-models` to skip the temporal modeling phase (regime detection, forecasting, Monte Carlo). This produces a report based on financial health and fundamental analysis only, which is still useful for screening.

### "pandoc not found" when using --pdf
Install pandoc:
- **Linux Mint**: `sudo apt install pandoc -y`
- **Windows**: Download from [pandoc.org/installing.html](https://pandoc.org/installing.html)
