#!/usr/bin/env bash
# One-command local setup for genome_codex (macOS and Linux).
# Creates .venv, installs pinned dependencies, and verifies that the
# historical experiments reproduce on this machine.
#
#   ./setup.sh            setup + verification
#   ./setup.sh --no-verify   setup only
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  # Take the first candidate that is actually 3.11+; a bare python3 can be
  # an older system interpreter (macOS ships 3.9).
  for c in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3.11+ not found. Install it (macOS: brew install python@3.13) and re-run." >&2
  exit 1
fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Need Python 3.11 or newer, found $("$PY" --version)." >&2; exit 1; }

echo "==> Using $("$PY" --version) at $(command -v "$PY")"
if [ ! -d .venv ]; then
  echo "==> Creating virtual environment in .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
echo "==> Installing pinned dependencies (torch is large, this can take a few minutes)"
python -m pip install --quiet -r requirements.txt

python - <<'PY'
import torch, transformers, sklearn, numpy
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"==> numpy {numpy.__version__}, scikit-learn {sklearn.__version__}, "
      f"torch {torch.__version__}, transformers {transformers.__version__}")
print(f"==> Accelerator available to torch: {dev}")
PY

if [ "${1:-}" != "--no-verify" ]; then
  echo "==> Verifying that the stored results reproduce on this machine"
  python -I scripts/verify.py
fi
echo "==> Done. Activate later with: source .venv/bin/activate"
