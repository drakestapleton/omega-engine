#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
OMEGA_MV="${OMEGA_MV:-$ROOT/../omega-mv}"
OUT_DIR="${OUT_DIR:-$ROOT/build}"
PIN=$(tr -d ' \t\r\n' < "$ROOT/omega-mv.lock")

if [ ! -d "$OMEGA_MV/.git" ]; then
  echo "ERROR: omega-mv checkout not found at $OMEGA_MV" >&2
  echo "Clone it first: git clone https://interplane.io/drakestapleton/omega-mv.git \"$OMEGA_MV\" && git -C \"$OMEGA_MV\" checkout $PIN" >&2
  exit 1
fi

sh "$OMEGA_MV/om/gpu/engine/build.sh" "$OUT_DIR"
ls -lh "$OUT_DIR/omega-engine"
