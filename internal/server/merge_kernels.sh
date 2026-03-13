#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_glob> <output_file>" >&2
  echo "Example: $0 'kernels/counts_*.h5' kernels/kernels.h5" >&2
  exit 1
fi

INPUT_GLOB="$1"
OUTPUT_FILE="$2"

python python/merge_kernels.py --inputs "$INPUT_GLOB" --out "$OUTPUT_FILE"
