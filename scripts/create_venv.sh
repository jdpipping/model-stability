#!/usr/bin/env bash
# Create .venv with Python 3.12 and install deps. Run from repo root.
set -e
PY312=""
for candidate in python3.12 \
  /opt/homebrew/opt/python@3.12/bin/python3.12 \
  /usr/local/opt/python@3.12/bin/python3.12; do
  if command -v "$candidate" &>/dev/null && "$candidate" -c 'import sys; exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
    PY312="$candidate"
    break
  fi
done
if [ -z "$PY312" ]; then
  echo "Python 3.12 not found. Install it first, e.g.:"
  echo "  brew install python@3.12"
  echo "Then run this script again."
  exit 1
fi
echo "Using: $PY312 ($($PY312 --version))"
rm -rf .venv
"$PY312" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"
