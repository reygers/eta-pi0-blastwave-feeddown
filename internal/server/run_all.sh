#!/bin/bash
set -euo pipefail

# Full paper-style pipeline for the configured energies/centralities.
# Run from repo root:
#   bash scripts/run_all.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
KERNELS="kernels/kernels_scan06_alice.h5"
NTHERM=10
NQUANT=10
OUTDIR="figures/current"

"$PYTHON_BIN" python/run_pipeline.py \
  --energy 2.76 \
  --centralities 0-10,20-50 \
  --kernels "$KERNELS" \
  --n-thermal-terms "$NTHERM" \
  --n-quantum-terms "$NQUANT" \
  --preset paper \
  --outdir "$OUTDIR"

"$PYTHON_BIN" python/run_pipeline.py \
  --energy 5.02 \
  --centralities 0-20,20-40,40-60,60-80,0-10,30-50 \
  --kernels "$KERNELS" \
  --n-thermal-terms "$NTHERM" \
  --n-quantum-terms "$NQUANT" \
  --preset paper \
  --outdir "$OUTDIR"
