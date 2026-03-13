#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate a virtualenv first (expected .venv)." >&2
  exit 1
fi

mkdir -p results/logs environment
python scripts/check_imports.py
python -m pip freeze | sort > environment/lock.txt
python -m pip freeze | sort > results/logs/pip_freeze.txt

echo "Wrote environment/lock.txt and results/logs/pip_freeze.txt"
