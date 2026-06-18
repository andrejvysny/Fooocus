#!/usr/bin/env bash
# Fooocus launcher for macOS (Apple Silicon / MPS) and Linux (NVIDIA / CUDA, AMD / ROCm, CPU).
# Uses the uv package manager: `uv run` creates and syncs the virtual environment from
# pyproject.toml + uv.lock before launching, so no manual install step is required.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install it (https://docs.astral.sh/uv/getting-started/installation/):"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# entry_with_update.py git-pulls the latest Fooocus, then launches launch.py.
# Use launch.py directly (uv run python launch.py) if you do not want auto-update.
exec uv run python entry_with_update.py "$@"
