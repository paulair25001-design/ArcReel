#!/usr/bin/env bash
# Run this fork against its isolated local runtime. No engine/service lifecycle actions.
set -euo pipefail
ARCREEL_CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ARCREEL_CHECKOUT"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Missing .venv: install the locked Python dependencies first.' >&2
  exit 1
fi
if [[ ! -f frontend/dist/index.html ]]; then
  echo 'Missing frontend build: run pnpm --dir frontend build first.' >&2
  exit 1
fi
export ARCREEL_LOCAL_ONLY=true
export ARCREEL_OFFICIAL_SERVICE_URL=''
ARCREEL_LOCAL_PORT="${LISTEN_PORT:-1241}"
export ARCREEL_HERMES_MCP_URL="http://127.0.0.1:${ARCREEL_LOCAL_PORT}/mcp"
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port "$ARCREEL_LOCAL_PORT" --timeout-graceful-shutdown 5
