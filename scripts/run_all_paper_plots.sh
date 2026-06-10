#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root, regardless of caller location.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
NTHERM="${NTHERM:-10}"
NQUANT="${NQUANT:-10}"
KERNELS="${KERNELS:-kernels/kernels_scan09_alice.h5}"

echo "[1/10] blast-wave fit"
"$PYTHON_BIN" python/fit_blastwave_combined.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels "$KERNELS" \
  --n-thermal-terms "$NTHERM" \
  --n-quantum-terms "$NQUANT" \
  --outdir paper/figures

echo "[2/10] double-ratio plot"
"$PYTHON_BIN" python/plot_double_ratio_feeddown.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels "$KERNELS" \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --n-thermal-terms "$NTHERM" \
  --n-quantum-terms "$NQUANT" \
  --pt-min 0.05 --pt-max 3.0 --pt-step 0.05 \
  --show-components \
  --outdir paper/figures

echo "[3/10] flow-scaling collapse figure"
"$PYTHON_BIN" python/fit_flow_scaling.py \
  --energy 2.76 \
  --centrality 0-10 \
  --n-thermal-terms "$NTHERM" \
  --smooth-f tsallis \
  --logy-collapse \
  --collapse-x-min 0.2 \
  --collapse-x-max 1.4 \
  --paper-style \
  --collapse-only \
  --outdir paper/figures

echo "[4/10] flow-scaling ratio CSV"
"$PYTHON_BIN" python/fit_flow_scaling.py \
  --energy 2.76 \
  --centrality 0-10 \
  --n-thermal-terms "$NTHERM" \
  --smooth-f tsallis \
  --eta-norm-mode fit-anchor \
  --anchor-min-pt 2.0 \
  --anchor-max-pt 4.0 \
  --write-ratio-csv \
  --outdir outputs/current

echo "[5/10] systematics summary"
"$PYTHON_BIN" python/systematics_double_ratio_scan.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels "$KERNELS" \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --flow-csv outputs/current/flow_scaling_eta_ratios_2p76TeV_0_10.csv \
  --fluidum-csv data/FluiduM/fluidum_double_ratio_points.csv \
  --n-thermal-terms "$NTHERM" \
  --n-quantum-terms "$NQUANT" \
  --tchem-list 0.136,0.156,0.176 \
  --run-no-feeddown-refit \
  --other-outdir outputs/current \
  --outdir paper/figures

echo "[6/10] final eta/pi0 figure"
"$PYTHON_BIN" python/predict_eta_to_pi0.py \
  --energy 2.76 \
  --centrality 0-10 \
  --kernels "$KERNELS" \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --n-thermal-terms "$NTHERM" \
  --n-quantum-terms "$NQUANT" \
  --show-estimated-band \
  --systematics-curves-csv outputs/current/systematics_double_ratio_curves_2p76TeV_0_10.csv \
  --main-plot-only \
  --outdir paper/figures

echo "[7/10] feeddown appendix figures"
"$PYTHON_BIN" python/plot_feeddown_fractions.py \
  --kernels "$KERNELS" \
  --fit-results data/fit_results/blastwave_fit_2p76TeV_0_10.json \
  --species k_plus,eta \
  --top-n 10 \
  --paper-style \
  --outdir paper/figures

echo "[8/10] decay-photon cocktail fractions"
DECAY_PHOTON_KERNELS="${DECAY_PHOTON_KERNELS:-kernels/kernels_scan01_decay_photon_components.h5}"
"$PYTHON_BIN" python/plot_decay_photons.py \
  --kernels "$DECAY_PHOTON_KERNELS" \
  --energy 2.76 \
  --centrality 0-10 \
  --eta-pi0-curve-csv paper/figures/eta_to_pi0_fitcurve_2p76TeV_0_10.csv \
  --outdir paper/figures \
  --n-thermal-terms "$NTHERM"

echo "[9/10] pp+pA eta/pi0 fit reference"
"$PYTHON_BIN" python/fit_eta_to_pi0_pp_pA.py \
  --outdir paper/figures

echo "[10/10] mT-scaling appendix figure"
"$PYTHON_BIN" python/plot_eta_pi0_mt_vs_flow.py \
  --energy 2.76 \
  --centrality 0-10 \
  --pseudo-csv paper/figures/eta_to_pi0_pseudo_2p76TeV_0_10.csv \
  --outdir paper/figures

echo "Done. Paper figures regenerated in paper/figures."
