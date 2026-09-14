#!/bin/bash
# Copies the standalone Python engine (../engine, this repo's single source of
# truth) into plugin/engine/ so it ships inside the npm package. plugin/engine/
# is generated (gitignored) -- never edit it directly, edit ../engine and rerun
# this script (or `npm run sync-engine`, wired as `prepack` so a real `npm
# pack`/`npm publish` always ships a fresh copy).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PLUGIN_DIR="$ROOT/plugin"
ENGINE_SRC="$ROOT/engine"
ENGINE_DEST="$PLUGIN_DIR/engine"

rm -rf "$ENGINE_DEST"
mkdir -p "$ENGINE_DEST"

cp -R "$ENGINE_SRC/src" "$ENGINE_DEST/src"
cp "$ENGINE_SRC/pyproject.toml" "$ENGINE_DEST/pyproject.toml"
[ -f "$ENGINE_SRC/uv.lock" ] && cp "$ENGINE_SRC/uv.lock" "$ENGINE_DEST/uv.lock"

# Drop bytecode/pycache picked up by a wildcard cp -- not needed at runtime
# and just bloats the published tarball.
find "$ENGINE_DEST" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "synced $(basename "$ENGINE_SRC") -> $ENGINE_DEST"
