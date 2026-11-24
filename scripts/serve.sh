#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN="${OUT_DIR:-$ROOT/build}/omega-engine"
MODEL="${1:?usage: sh scripts/serve.sh /path/to/model-dir [--port PORT] [--self-test]}"
shift
exec "$BIN" serve "$MODEL" "$@"
