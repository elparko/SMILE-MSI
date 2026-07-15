#!/bin/bash
# Double-click launcher (macOS) for the SMILE MSI desktop app.
# First run builds a local environment; afterwards it opens instantly.
cd "$(dirname "$0")" || exit 1

if [ ! -d .venv ]; then
  echo "First run: building the environment (a minute or two)…"
  if command -v uv >/dev/null 2>&1; then
    uv venv && ./.venv/bin/python -m ensurepip >/dev/null 2>&1
    uv pip install --python ./.venv/bin/python -e '.[gui]'
  else
    python3 -m venv .venv
    ./.venv/bin/pip install -e '.[gui]'
  fi
fi

exec ./.venv/bin/python -m smile_msi.gui
