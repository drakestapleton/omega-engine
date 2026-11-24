#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN="${OUT_DIR:-$ROOT/build}/omega-engine"
MODEL="${1:?usage: sh scripts/inspect.sh /path/to/model-dir}"
exec "$BIN" inspect "$MODEL"
