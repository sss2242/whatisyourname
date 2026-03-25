#!/usr/bin/env bash
# install.sh -- Install Python 3.12 and all dependency stages
#
# Supports: Linux Mint, Ubuntu, Debian, Fedora, RHEL, Amazon Linux, macOS
#
# Usage:
#   chmod +x install.sh
#   ./install.sh          # Install everything (all 4 stages)
#   ./install.sh 1        # Stage 1 only: Core libraries (~30s)
#   ./install.sh 2        # Stage 2 only: ML and statistics (~1-2min)
#   ./install.sh 3        # Stage 3 only: Deep learning + Bayesian (~5-8min)
#   ./install.sh 4        # Stage 4 only: Data source wrappers (~1-2min)
#   ./install.sh venv     # Just create the virtual environment
#   ./install.sh check    # Run import smoke test only

set -euo pipefail

PYTHON_VERSION="3.12"
PYTHON_VERSION_FULL="3.12.3"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${REPO_DIR}/venv"

# ---------- helpers ----------
info()  { echo -e "\033[32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[31m[ERROR]\033[0m $*" >&2; exit 1; }

# ---------- OS detection ----------
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_ID_LIKE="${ID_LIKE:-}"
    elif [[ "$(uname)" == "Darwin" ]]; then
        OS_ID="macos"
        OS_ID_LIKE=""
    else
        OS_ID="unknown"
        OS_ID_LIKE=""
    fi
}

is_debian_based() {
    [[ "${OS_ID}" == "debian" || "${OS_ID}" == "ubuntu" || "${OS_ID}" == "linuxmint" || "${OS_ID}" == "pop" || "${OS_ID}" == "elementary" || "${OS_ID}" == "zorin" || "${OS_ID_LIKE}" == *"debian"* || "${OS_ID_LIKE}" == *"ubuntu"* ]]
}

is_fedora_based() {
    [[ "${OS_ID}" == "fedora" || "${OS_ID}" == "rhel" || "${OS_ID}" == "centos" || "${OS_ID}" == "amzn" || "${OS_ID}" == "rocky" || "${OS_ID}" == "alma" || "${OS_ID_LIKE}" == *"fedora"* || "${OS_ID_LIKE}" == *"rhel"* ]]
}

# ---------- 0. Install system build dependencies ----------
install_system_deps() {
    detect_os
    info "Detected OS: ${OS_ID} (like: ${OS_ID_LIKE:-none})"

    if is_debian_based; then
        info "Installing build dependencies via apt ..."
        sudo apt update -qq
        sudo apt install -y \
            python3-pip python3-venv python3-dev \
            build-essential \
            libssl-dev libbz2-dev libffi-dev zlib1g-dev \
            libreadline-dev libsqlite3-dev libncurses-dev \
            liblzma-dev tk-dev \
            git curl wget

        # Check if Python 3.12+ is available
        if ! command -v python${PYTHON_VERSION} &>/dev/null; then
            info "Python ${PYTHON_VERSION} not found. Adding deadsnakes PPA ..."
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update -qq
            sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
        fi

    elif is_fedora_based; then
        info "Installing build dependencies via dnf/yum ..."
        if command -v dnf &>/dev/null; then
            PKG_MGR="dnf"
        else
            PKG_MGR="yum"
        fi
        sudo ${PKG_MGR} install -y \
            gcc make \
            openssl-devel bzip2-devel libffi-devel zlib-devel \
            readline-devel sqlite-devel xz-devel tk-devel \
            git curl wget

        # Fedora 39+ ships Python 3.12; older versions may need compilation
        if ! command -v python${PYTHON_VERSION} &>/dev/null; then
            warn "Python ${PYTHON_VERSION} not available via package manager."
            warn "Will attempt to compile from source."
            compile_python
        fi

    elif [[ "${OS_ID}" == "macos" ]]; then
        info "macOS detected. Checking for Homebrew ..."
        if ! command -v brew &>/dev/null; then
            error "Homebrew not found. Install it first: https://brew.sh"
        fi
        brew install python@${PYTHON_VERSION} 2>/dev/null || true

    else
        warn "Unknown OS: ${OS_ID}. Attempting to continue with existing Python."
    fi
}

# ---------- Compile Python from source (fallback) ----------
compile_python() {
    local PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION_FULL}/Python-${PYTHON_VERSION_FULL}.tgz"
    local PREFIX="/usr/local"

    if "${PREFIX}/bin/python${PYTHON_VERSION}" --version 2>/dev/null | grep -q "${PYTHON_VERSION_FULL}"; then
        info "Python ${PYTHON_VERSION_FULL} already compiled at ${PREFIX} -- skipping build."
        return
    fi

    info "Downloading Python ${PYTHON_VERSION_FULL} source ..."
    cd /tmp
    curl -sO "${PYTHON_URL}"
    tar xzf "Python-${PYTHON_VERSION_FULL}.tgz"
    cd "Python-${PYTHON_VERSION_FULL}"

    info "Configuring ..."
    ./configure --enable-optimizations --with-ensurepip=install --prefix="${PREFIX}" >/dev/null

    info "Building ($(nproc) cores) -- this takes a few minutes ..."
    make -j"$(nproc)" >/dev/null 2>&1

    info "Installing (altinstall to avoid overwriting system python) ..."
    sudo make altinstall >/dev/null 2>&1

    cd "${REPO_DIR}"
    info "Python ${PYTHON_VERSION_FULL} compiled and installed to ${PREFIX}"
}

# ---------- Find the best available Python 3.12+ ----------
find_python() {
    # Try exact version first, then minor versions
    for candidate in \
        "python${PYTHON_VERSION}" \
        "python3.12" \
        "python3.13" \
        "python3.14" \
        "/usr/local/bin/python${PYTHON_VERSION}" \
        "/usr/local/bin/python3.12" \
        "python3"; do
        if command -v "${candidate}" &>/dev/null; then
            local ver
            ver=$("${candidate}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
            local major minor
            major=$(echo "${ver}" | cut -d. -f1)
            minor=$(echo "${ver}" | cut -d. -f2)
            if [[ "${major}" -eq 3 && "${minor}" -ge 12 ]]; then
                PYTHON_BIN="${candidate}"
                info "Found Python ${ver} at $(command -v ${candidate})"
                return
            fi
        fi
    done

    error "Python 3.12+ not found. Please install Python 3.12 or newer.
On Linux Mint / Ubuntu:
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.12 python3.12-venv python3.12-dev

On Fedora:
    sudo dnf install python3.12

On macOS:
    brew install python@3.12"
}

# ---------- Create / activate virtual environment ----------
setup_venv() {
    find_python

    if [[ -d "${VENV_DIR}" && -f "${VENV_DIR}/bin/python" ]]; then
        # Verify existing venv is still valid
        if "${VENV_DIR}/bin/python" --version &>/dev/null; then
            info "Virtual environment already exists at ${VENV_DIR}"
        else
            warn "Existing venv is broken. Recreating ..."
            rm -rf "${VENV_DIR}"
            "${PYTHON_BIN}" -m venv "${VENV_DIR}"
        fi
    else
        info "Creating virtual environment at ${VENV_DIR} ..."
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi

    # Use the venv's pip and python from here on
    PIP="${VENV_DIR}/bin/pip"
    PYTHON="${VENV_DIR}/bin/python"

    info "Upgrading pip ..."
    "${PYTHON}" -m pip install --upgrade pip --quiet

    info "Python: $(${PYTHON} --version)"
    info "Pip:    $(${PIP} --version)"
}

# ---------- Stage installers ----------
install_stage1() {
    info "Stage 1/4: Core libraries (numpy, pandas, scipy, matplotlib, pytest) ..."
    "${PIP}" install --timeout 300 -r "${REPO_DIR}/requirements/stage1-core.txt"
    info "Stage 1 complete."
}

install_stage2() {
    info "Stage 2/4: ML & statistics (scikit-learn, xgboost, statsmodels, shap) ..."
    "${PIP}" install --timeout 300 -r "${REPO_DIR}/requirements/stage2-ml.txt"
    info "Stage 2 complete."
}

install_stage3() {
    info "Stage 3/4: Deep learning & Bayesian (PyTorch ~2 GB, PyMC) ..."
    info "This stage downloads large files. If it times out, just re-run: ./install.sh 3"
    "${PIP}" install --timeout 300 -r "${REPO_DIR}/requirements/stage3-deeplearning.txt"
    info "Stage 3 complete."
}

install_stage4() {
    info "Stage 4/4: Data source wrappers (edgartools, yfinance, dart-fss, etc.) ..."
    "${PIP}" install --timeout 300 -r "${REPO_DIR}/requirements/stage4-wrappers.txt"
    info "Stage 4 complete."
}

# ---------- Smoke test ----------
run_smoke_test() {
    info "Running import smoke test ..."
    "${PYTHON}" -c "
import sys
print(f'Python {sys.version}')

# Stage 1
import numpy, pandas, scipy, requests, yaml, matplotlib
print('[OK] Stage 1: Core libraries')

# Stage 2
try:
    import sklearn, statsmodels, xgboost, ruptures, hmmlearn, arch, shap, mapie, dtaidistance
    print('[OK] Stage 2: ML & statistics')
except ImportError as e:
    print(f'[SKIP] Stage 2: {e}')

# Stage 3
try:
    import torch, arviz, pymc
    print('[OK] Stage 3: Deep learning & Bayesian')
except ImportError as e:
    print(f'[SKIP] Stage 3: {e}')

# Stage 4
try:
    import edgar, yfinance, wbgapi, fredapi
    print('[OK] Stage 4: Data source wrappers')
except ImportError as e:
    print(f'[SKIP] Stage 4: {e}')

print()
print('Import test complete. Any [SKIP] stages can be installed with: ./install.sh <stage_number>')
"
    info "Smoke test complete."
}

# ---------- Main ----------
main() {
    local stage="${1:-all}"

    case "${stage}" in
        all)
            install_system_deps
            setup_venv
            install_stage1
            install_stage2
            install_stage3
            install_stage4
            run_smoke_test
            echo ""
            info "All done. Activate the virtual environment with:"
            info "    source venv/bin/activate"
            info "Then run: python run.py"
            ;;
        1)
            setup_venv
            install_stage1
            ;;
        2)
            setup_venv
            install_stage2
            ;;
        3)
            setup_venv
            install_stage3
            ;;
        4)
            setup_venv
            install_stage4
            ;;
        venv)
            install_system_deps
            setup_venv
            info "Virtual environment ready. Activate with: source venv/bin/activate"
            ;;
        check|test|smoke)
            setup_venv
            run_smoke_test
            ;;
        *)
            echo "Usage: $0 [all|1|2|3|4|venv|check]"
            echo ""
            echo "  all   -- Install everything (default)"
            echo "  1     -- Stage 1: Core libraries (~30s)"
            echo "  2     -- Stage 2: ML & statistics (~1-2min)"
            echo "  3     -- Stage 3: Deep learning + Bayesian (~5-8min)"
            echo "  4     -- Stage 4: Data source wrappers (~1-2min)"
            echo "  venv  -- Just create the virtual environment"
            echo "  check -- Run import smoke test"
            exit 1
            ;;
    esac
}

main "$@"
