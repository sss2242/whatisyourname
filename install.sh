#!/usr/bin/env bash
# install.sh -- Install Python 3.12.3 from source and all dependency stages
# Tested on Amazon Linux 2023 (x86_64)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# The script installs Python 3.12.3 to /usr/local and then installs
# pip packages in four stages (core, ML, deep-learning, wrappers).

set -euo pipefail

PYTHON_VERSION="3.12.3"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
PREFIX="/usr/local"
PIP="${PREFIX}/bin/pip3.12"
PYTHON="${PREFIX}/bin/python3.12"

# ---------- helpers ----------
info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# ---------- 0. Build dependencies ----------
info "Installing build dependencies ..."
sudo dnf install -y \
  gcc make \
  openssl-devel bzip2-devel libffi-devel zlib-devel \
  readline-devel sqlite-devel xz-devel tk-devel

# ---------- 1. Download & compile Python ----------
if "${PYTHON}" --version 2>/dev/null | grep -q "${PYTHON_VERSION}"; then
  info "Python ${PYTHON_VERSION} already installed -- skipping build."
else
  info "Downloading Python ${PYTHON_VERSION} ..."
  cd /tmp
  curl -sO "${PYTHON_URL}"
  tar xzf "Python-${PYTHON_VERSION}.tgz"
  cd "Python-${PYTHON_VERSION}"

  info "Configuring ..."
  ./configure --enable-optimizations --with-ensurepip=install --prefix="${PREFIX}" >/dev/null

  info "Building ($(nproc) cores) ..."
  make -j"$(nproc)" >/dev/null 2>&1

  info "Installing (altinstall to avoid overwriting system python) ..."
  sudo make altinstall >/dev/null 2>&1
fi

"${PYTHON}" --version
"${PIP}" --version

# ---------- 2. Install pip packages in stages ----------
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

info "Stage 1/4: Core libraries ..."
"${PIP}" install --timeout 300 -r "${REPO_DIR}/requirements/stage1-core.txt"

info "Stage 2/4: ML & statistics ..."
"${PIP}" install --timeout 300 -r "${REPO_DIR}/requirements/stage2-ml.txt"

info "Stage 3/4: Deep learning & Bayesian ..."
"${PIP}" install --timeout 300 -r "${REPO_DIR}/requirements/stage3-deeplearning.txt"

info "Stage 4/4: Data source wrappers ..."
"${PIP}" install --timeout 300 -r "${REPO_DIR}/requirements/stage4-wrappers.txt"

# ---------- 3. Quick smoke test ----------
info "Running import smoke test ..."
"${PYTHON}" -c "
import numpy, pandas, scipy, requests, yaml, matplotlib
import sklearn, statsmodels, xgboost, ruptures, hmmlearn, arch, shap, mapie, dtaidistance
import torch, arviz, pymc
import edgar, yfinance, wbgapi, fredapi
print('All stage imports OK')
"

info "Done. Python ${PYTHON_VERSION} + all dependencies installed."
