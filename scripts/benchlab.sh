#!/usr/bin/env bash
set -euo pipefail

: "${BENCHLAB_PROJECT:?Set BENCHLAB_PROJECT to the external local-llm-serving-lab project directory (see scripts/README.md)}"
if [[ ! -f "$BENCHLAB_PROJECT/pyproject.toml" ]]; then
    echo "error: BENCHLAB_PROJECT must contain pyproject.toml" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "error: install uv and run uv sync --project \"$BENCHLAB_PROJECT\" first" >&2
    exit 1
fi
exec uv run --project "$BENCHLAB_PROJECT" benchlab "$@"
