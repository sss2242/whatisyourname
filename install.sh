#!/usr/bin/env bash
# install.sh -- Install all Operator 1 dependencies in 4 stages using Python 3.12
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# On Amazon Linux 2023:
#   sudo dnf install -y python3.12 python3.12-pip python3.12-setuptools
#
# On Ubuntu/Debian (deadsnakes PPA):
#   sudo add-apt-repository ppa:deadsnakes/ppa
#   sudo apt install -y python3.12 python3.12-venv python3.12-dev
#
set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
PIP_TIMEOUT="${PIP_TIMEOUT:-300}"

echo "==> Using $($PYTHON --version 2>&1)"
echo "==> Upgrading pip..."
$PYTHON -m pip install --upgrade pip

echo ""
echo "==> Stage 1/4: Core libraries (numpy, pandas, scipy, etc.)"
$PYTHON -m pip install --timeout "$PIP_TIMEOUT" -r requirements/stage1-core.txt

echo ""
echo "==> Stage 2/4: ML / Statistics (scikit-learn, xgboost, statsmodels, etc.)"
$PYTHON -m pip install --timeout "$PIP_TIMEOUT" -r requirements/stage2-ml.txt

echo ""
echo "==> Stage 3/4: Deep Learning + Bayesian (torch ~2 GB, pymc)"
echo "    If this stage times out, re-run -- pip will resume from cache."
$PYTHON -m pip install --timeout "$PIP_TIMEOUT" -r requirements/stage3-deeplearning.txt

echo ""
echo "==> Stage 4/4: Data Source Wrappers (edgartools, yfinance, wbgapi, etc.)"
$PYTHON -m pip install --timeout "$PIP_TIMEOUT" -r requirements/stage4-wrappers.txt

echo ""
echo "==> All 4 stages installed successfully."
$PYTHON -c "
import numpy, pandas, scipy, sklearn, torch
print(f'  Python:       {__import__(\"sys\").version.split()[0]}')
print(f'  NumPy:        {numpy.__version__}')
print(f'  Pandas:       {pandas.__version__}')
print(f'  SciPy:        {scipy.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  PyTorch:      {torch.__version__}')
"
echo "Done."
