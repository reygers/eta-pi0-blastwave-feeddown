#!/usr/bin/env bash
set -euo pipefail

# A/B check for old vs Thermal-FIST-extended decay setup.
# Run from repo root.

NEV="${NEV:-20000}"
OUTDIR="${OUTDIR:-kernels}"
FIT_JSON="${FIT_JSON:-python/blastwave_fit_2p76TeV_0_10.json}"

OLD_DECAYTABLE="${OLD_DECAYTABLE:-data/decaytables/decaytable_pythia8.dat}"
NEW_DECAYTABLE="${NEW_DECAYTABLE:-data/decaytables/decaytable_pythia8_full_from_thermalfist.dat}"
OVERLAY_CMND="${OVERLAY_CMND:-data/Thermal-FIST 1.6/missing_particles_eta_k_p_auto.cmnd}"

OLD_KERNEL="${OUTDIR}/ab_old_small.h5"
NEW_KERNEL="${OUTDIR}/ab_new_small.h5"

mkdir -p "${OUTDIR}"

echo "[1/4] Build old-mode kernel -> ${OLD_KERNEL}"
./kernel_factory/kernel_factory \
  --out "${OLD_KERNEL}" \
  --nev "${NEV}" \
  --decaytable "${OLD_DECAYTABLE}" \
  --decaytable-reset 0

echo "[2/4] Build new-mode kernel -> ${NEW_KERNEL}"
./kernel_factory/kernel_factory \
  --out "${NEW_KERNEL}" \
  --nev "${NEV}" \
  --decaytable "${NEW_DECAYTABLE}" \
  --decaytable-reset 1 \
  --overlay-cmnd "${OVERLAY_CMND}"

echo "[3/4] Compare old-mode kernel to Thermal-FIST"
.venv/bin/python python/compare_thermal_fist.py \
  --kernels "${OLD_KERNEL}" \
  --fit-results "${FIT_JSON}"

echo "[4/4] Compare new-mode kernel to Thermal-FIST"
.venv/bin/python python/compare_thermal_fist.py \
  --kernels "${NEW_KERNEL}" \
  --fit-results "${FIT_JSON}"

echo "Done. Kernels:"
echo "  ${OLD_KERNEL}"
echo "  ${NEW_KERNEL}"

