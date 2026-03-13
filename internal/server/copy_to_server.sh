#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/copy_to_server.sh <user@host> <remote_base_dir>
# Example:
#   scripts/copy_to_server.sh user@my.server.edu /path/to/kernel_factory_run

if [ $# -lt 2 ]; then
  echo "Usage: $0 <user@host> <remote_base_dir>" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_BASE="$2"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Remote paths
REMOTE_SRC="$REMOTE_BASE/src"
REMOTE_INPUTS="$REMOTE_BASE/inputs"
REMOTE_SCRIPTS="$REMOTE_BASE/scripts"

# Local files needed for kernel scan production only
SRC_CC="$LOCAL_ROOT/kernel_factory/kernel_factory.cc"
SRC_MAKE="$LOCAL_ROOT/kernel_factory/Makefile.server"
DECAYTABLE="$LOCAL_ROOT/data/decaytables/decaytable_pythia8.dat"
OVERLAY_CMND="${OVERLAY_CMND:-$LOCAL_ROOT/data/Thermal-FIST 1.6/missing_particles_all_lt2gev_tau1mm.cmnd}"
RUN_SCRIPT="$LOCAL_ROOT/scripts/run_kernel_counts.sh"
RUN_FINE_SCRIPT="$LOCAL_ROOT/scripts/run_kernel_counts_fine.sh"
RUN_DECAY_PHOTON_SCRIPT="$LOCAL_ROOT/scripts/run_kernel_counts_decay_photon.sh"
MERGE_SCRIPT="$LOCAL_ROOT/scripts/merge_kernels.sh"
MERGE_FINE_SCRIPT="$LOCAL_ROOT/scripts/merge_kernels_fine.sh"

# Ensure remote dirs exist
ssh "$REMOTE" "mkdir -p '$REMOTE_SRC' '$REMOTE_INPUTS' '$REMOTE_SCRIPTS'"

# Copy source as explicit files (no local executable or unrelated files)
scp "$SRC_CC" "$REMOTE:$REMOTE_SRC/kernel_factory.cc"
scp "$SRC_MAKE" "$REMOTE:$REMOTE_SRC/Makefile"
scp "$DECAYTABLE" "$REMOTE:$REMOTE_INPUTS/"
if [ -f "$OVERLAY_CMND" ]; then
  scp "$OVERLAY_CMND" "$REMOTE:$REMOTE_INPUTS/"
else
  echo "Warning: overlay cmnd not found locally: $OVERLAY_CMND"
fi
scp "$RUN_SCRIPT" "$REMOTE:$REMOTE_SCRIPTS/"
scp "$RUN_FINE_SCRIPT" "$REMOTE:$REMOTE_SCRIPTS/"
scp "$RUN_DECAY_PHOTON_SCRIPT" "$REMOTE:$REMOTE_SCRIPTS/"
scp "$MERGE_SCRIPT" "$REMOTE:$REMOTE_SCRIPTS/"
scp "$MERGE_FINE_SCRIPT" "$REMOTE:$REMOTE_SCRIPTS/"

echo "Uploaded scan-production files to $REMOTE_BASE"
