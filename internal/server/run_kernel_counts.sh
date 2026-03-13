#!/bin/bash

## Resource Request
#SBATCH --job-name=kernel_counts
#SBATCH --array=1-150
#SBATCH --output=logs/kernel_counts_%A_%a.out
#SBATCH --error=logs/kernel_counts_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --partition=main

# --- environment ---
# Update these paths for your cluster
ROOT_SETUP=/misc/alidata131/alice_u/reygers/root-6.32.10/bin/thisroot.sh
PYTHIA8_ROOT=${PYTHIA8_ROOT:-/misc/alidata131/alice_u/reygers/pythia8317}
PYTHIA_XMLDOC=$PYTHIA8_ROOT/share/Pythia8/xmldoc
HDF5_ROOT=/misc/alidata131/alice_u/reygers/hdf5
KERNEL_BASE=/misc/alidata131/alice_u/reygers/kernel_factory_run

if [ -f "$ROOT_SETUP" ]; then
  source "$ROOT_SETUP"
else
  echo "ROOT setup not found at $ROOT_SETUP" >&2
  exit 1
fi

export PYTHIA8="$PYTHIA8_ROOT"
export PYTHIA8DATA="$PYTHIA_XMLDOC"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:$LD_LIBRARY_PATH"

# --- run ---
NEVENTS=1000000
OUTDIR=$KERNEL_BASE/outputs
OUTFILE="$OUTDIR/counts_${SLURM_ARRAY_TASK_ID}.h5"
DECAYTABLE_PATH="${DECAYTABLE_PATH:-$KERNEL_BASE/inputs/decaytable_pythia8.dat}"
DECAYTABLE_RESET="${DECAYTABLE_RESET:-0}"
OVERLAY_CMND_PATH="${OVERLAY_CMND_PATH:-$KERNEL_BASE/inputs/missing_particles_all_lt2gev_tau1mm.cmnd}"

mkdir -p "$OUTDIR" "$KERNEL_BASE/logs"

CMD=(
  "$KERNEL_BASE/bin/kernel_factory"
  --out "$OUTFILE"
  --nev "$NEVENTS"
  --counts-only
  --decaytable "$DECAYTABLE_PATH"
  --decaytable-reset "$DECAYTABLE_RESET"
  --pt-max-mother 15
  --n-pt-mother 300
)

if [ -f "$OVERLAY_CMND_PATH" ]; then
  CMD+=(--overlay-cmnd "$OVERLAY_CMND_PATH")
fi

"${CMD[@]}"
