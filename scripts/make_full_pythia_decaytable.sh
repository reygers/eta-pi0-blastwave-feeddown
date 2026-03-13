#!/usr/bin/env bash
set -euo pipefail

OUT_FILE="${1:-data/decaytables/decaytable_pythia8_full.dat}"

root -l -b -q "scripts/export_full_pythia_decaytable.C(\"${OUT_FILE}\")"

