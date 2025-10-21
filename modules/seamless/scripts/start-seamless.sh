#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONUNBUFFERED=1
export SEAMLESS_MODEL=${SEAMLESS_MODEL:-facebook/seamless-m4t-v2-large}
export DEVICE=${DEVICE:-cpu}

cd "$ROOT_DIR"

if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PY_BIN="$CONDA_PREFIX/bin/python"
else
  PY_BIN="${PYTHON:-$(command -v python3 || command -v python || true)}"
fi
if [ -z "$PY_BIN" ]; then
  echo "[ERROR] No python interpreter found."
  exit 1
fi

PYVER=$("$PY_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "0.0")
MAJ="${PYVER%%.*}"
MIN_TMP="${PYVER#*.}"
MIN="${MIN_TMP%%.*}"
if [ -z "$MAJ" ] || [ -z "$MIN" ]; then
  MAJ=0; MIN=0
fi
if [ "$MAJ" -lt 3 ] || { [ "$MAJ" -eq 3 ] && [ "$MIN" -lt 8 ]; }; then
  echo "[ERROR] Python >= 3.8 required. Current: $PYVER"
  echo "[HINT] Create and activate the conda env:"
  echo "       conda env create -f environment.yml && conda activate livetranslate-seamless"
  exit 1
fi

echo "[INFO] Using python: $PY_BIN (version $MAJ.$MIN)"
echo "[INFO] Starting Seamless demo service on :5007"
exec "$PY_BIN" -m uvicorn src.server:app --host 0.0.0.0 --port 5007


