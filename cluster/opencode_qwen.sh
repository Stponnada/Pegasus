#!/bin/bash
# Backward-compatible alias retained for existing qwen-code installations.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/opencode_gemma.sh" "$@"
