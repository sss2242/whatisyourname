#!/usr/bin/env bash
# Staged dependency installer for Operator 1
# Installs in 4 stages to avoid pip timeouts on slow connections.
# Each stage is independent -- if one fails, re-run just that stage.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# Or run individual stages:
#   ./install.sh 3    # only install stage 3 (deep learning)

set -e

PIP_TIMEOUT=300
PIP_RETRIES=3

install_stage() {
    local stage=$1
    local file=$2
    local desc=$3
    echo ""
    echo "============================================"
    echo "  Stage ${stage}: ${desc}"
    echo "============================================"
    pip install --timeout ${PIP_TIMEOUT} --retries ${PIP_RETRIES} -r "${file}"
    echo "  Stage ${stage} complete."
}

# If a specific stage number is passed, only run that stage
if [ -n "$1" ]; then
    case "$1" in
        1) install_stage 1 requirements/stage1-core.txt "Core libraries (~30s)" ;;
        2) install_stage 2 requirements/stage2-ml.txt "ML and statistics (~1-2min)" ;;
        3) install_stage 3 requirements/stage3-deeplearning.txt "Deep learning + Bayesian (~5-8min)" ;;
        4) install_stage 4 requirements/stage4-wrappers.txt "Data source wrappers (~1-2min)" ;;
        *) echo "Usage: $0 [1|2|3|4]"; exit 1 ;;
    esac
    exit 0
fi

echo "Installing Operator 1 dependencies in 4 stages..."
echo "Python: $(python3 --version)"
echo ""

install_stage 1 requirements/stage1-core.txt "Core libraries (~30s)"
install_stage 2 requirements/stage2-ml.txt "ML and statistics (~1-2min)"
install_stage 3 requirements/stage3-deeplearning.txt "Deep learning + Bayesian (~5-8min)"
install_stage 4 requirements/stage4-wrappers.txt "Data source wrappers (~1-2min)"

echo ""
echo "============================================"
echo "  All stages complete!"
echo "============================================"
echo ""
echo "Verify with: python3 -c \"import pandas, numpy, torch; print('OK')\""
echo "Run tests:   python3 -m pytest tests/test_phase1_smoke.py"
