#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Note: scripts/sync_to_server.sh is deprecated. Using scripts/copy_to_server.sh."
exec "$SCRIPT_DIR/copy_to_server.sh" "$@"
