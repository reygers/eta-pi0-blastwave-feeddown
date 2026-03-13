#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/copy_from_server.sh <user@host> <remote_base_dir> <local_output_dir> [filename]
# Example:
#   scripts/copy_from_server.sh user@my.server.edu /path/to/kernel_factory_run kernels
#   scripts/copy_from_server.sh user@my.server.edu /path/to/kernel_factory_run kernels kernels_scan02.h5

if [ $# -lt 3 ]; then
  echo "Usage: $0 <user@host> <remote_base_dir> <local_output_dir> [filename]" >&2
  exit 1
fi

REMOTE="$1"
REMOTE_BASE="$2"
LOCAL_OUT="$3"
FILENAME="${4:-kernels.h5}"

mkdir -p "$LOCAL_OUT"

scp "$REMOTE:$REMOTE_BASE/outputs/$FILENAME" "$LOCAL_OUT/"

echo "Downloaded $FILENAME to $LOCAL_OUT"
