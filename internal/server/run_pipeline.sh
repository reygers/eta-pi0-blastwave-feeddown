#!/bin/bash
set -euo pipefail

# Convenience wrapper for python/run_pipeline.py
#
# Usage:
#   scripts/run_pipeline.sh <energy> <centrality> [extra args...]
# Example:
#   scripts/run_pipeline.sh 2.76 0-10 --kernels kernels/kernels_scan06_alice.h5 --n-thermal-terms 20 --n-quantum-terms 10
#   scripts/run_pipeline.sh 2.76 0-10 --run-phi-validation

if [ $# -lt 2 ]; then
  echo "Usage: $0 <energy> <centrality> [extra args...]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ENERGY="$1"
CENTRALITY="$2"
shift 2

"$PYTHON_BIN" python/run_pipeline.py --energy "$ENERGY" --centrality "$CENTRALITY" "$@"
