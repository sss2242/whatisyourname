# How to Run Operator 1

Step-by-step instructions for running Operator 1 on Linux Mint and Windows.

---

## Section 1: Linux Mint (and Ubuntu/Debian-based distros)

### Prerequisites

You need Python 3.12 or newer and git. Linux Mint ships with Python, but the version may be older than 3.12.

Open a terminal (Ctrl+Alt+T) and run:

```bash
# Check your Python version (need 3.12+)
python3 --version

# Install pip, venv, and build tools
sudo apt update
sudo apt install python3-pip python3-venv python3-dev build-essential git -y

# If your Python is older than 3.12, add the deadsnakes PPA:
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y
```

### Step 1: Clone the repository

```bash
cd ~
git clone https://github.com/oos24/whatisyourname.git
cd whatisyourname
```

### Step 2: Create a virtual environment

A virtual environment keeps Operator 1's dependencies isolated from your system Python.

```bash
# Use python3.12 if you installed it via deadsnakes PPA, otherwise python3
python3.12 -m venv venv   # or: python3 -m venv venv (if your python3 is 3.12+)
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt. Every time you open a new terminal to run Operator 1, you need to activate it again with `source venv/bin/activate`.

> **Note**: The `install.sh` script handles virtual environment creation automatically. If you use `./install.sh`, you can skip this step.

### Step 3: Install dependencies

All versions are pinned and verified on Python 3.12.3 (`.python-version`).

**Option A: Staged install (recommended -- avoids timeouts)**

```bash
chmod +x install.sh
./install.sh
```

This installs in 4 stages. If any stage fails (e.g. slow connection), re-run just that stage:

```bash
./install.sh 1   # Stage 1: Core libraries (~30s)
./install.sh 2   # Stage 2: ML and statistics (~1-2min)
./install.sh 3   # Stage 3: Deep learning + Bayesian (~5-8min, PyTorch is ~2 GB)
./install.sh 4   # Stage 4: Data source wrappers (~1-2min)
```

Or run each stage manually with pip:

```bash
pip install --timeout 300 -r requirements/stage1-core.txt
pip install --timeout 300 -r requirements/stage2-ml.txt
pip install --timeout 300 -r requirements/stage3-deeplearning.txt
pip install --timeout 300 -r requirements/stage4-wrappers.txt
```

**Option B: All at once** (may timeout on slow connections)

```bash
pip install --timeout 300 --retries 5 -r requirements.txt
```

**Option C: Lighter install** -- skip deep learning (LSTM, Transformer, Bayesian) for faster setup:

```bash
pip install --timeout 300 -r requirements/stage1-core.txt
pip install --timeout 300 -r requirements/stage2-ml.txt
pip install --timeout 300 -r requirements/stage4-wrappers.txt
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

You need Python 3.12 or newer and git.

1. **Install Python**: Download from [python.org/downloads](https://www.python.org/downloads/). During installation, check the box that says **"Add Python to PATH"** -- this is important.

2. **Install Git**: Download from [git-scm.com](https://git-scm.com/download/win). Use the default settings during installation.

3. **Open a terminal**: Press `Win+R`, type `cmd`, and press Enter. Or search for "Command Prompt" in the Start menu. You can also use PowerShell or Windows Terminal.

### Step 1: Clone the repository

```cmd
cd %USERPROFILE%
git clone https://github.com/oos24/whatisyourname.git
cd whatisyourname
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

**Option A: Use the install script (recommended)**

```cmd
install.bat
```

This creates a virtual environment and installs all 4 stages. If any stage fails (e.g. slow connection), re-run just that stage:

```cmd
install.bat 1   &REM Stage 1: Core libraries (~30s)
install.bat 2   &REM Stage 2: ML and statistics (~1-2min)
install.bat 3   &REM Stage 3: Deep learning + Bayesian (~5-8min, PyTorch is ~2 GB)
install.bat 4   &REM Stage 4: Data source wrappers (~1-2min)
```

After install.bat finishes, activate the virtual environment:

```cmd
venv\Scripts\activate
```

**Option B: Manual staged install**

```cmd
pip install --timeout 300 -r requirements\stage1-core.txt
pip install --timeout 300 -r requirements\stage2-ml.txt
pip install --timeout 300 -r requirements\stage3-deeplearning.txt
pip install --timeout 300 -r requirements\stage4-wrappers.txt
```

If any stage fails, just re-run that one command. Stage 3 is the largest (~2 GB for PyTorch).

**Option C: All at once** (may timeout on slow connections):

```cmd
pip install --timeout 300 --retries 5 -r requirements.txt
```

**Option D: Lighter install** -- skip deep learning for faster setup:

```cmd
pip install --timeout 300 -r requirements\stage1-core.txt
pip install --timeout 300 -r requirements\stage2-ml.txt
pip install --timeout 300 -r requirements\stage4-wrappers.txt
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
Make sure you are running the command from inside the `whatisyourname` directory and that your virtual environment is activated.

### install.sh fails with "dnf: command not found" (Linux Mint / Ubuntu)
The old install.sh was written for Amazon Linux. The updated version auto-detects your OS and uses the correct package manager (`apt` for Debian/Ubuntu/Mint, `dnf` for Fedora/RHEL). Make sure you have the latest `install.sh`.

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
