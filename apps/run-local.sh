#!/usr/bin/env bash
set -euo pipefail

APPS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_ROOT="${APPS_ROOT}/api"
WEB_ROOT="${APPS_ROOT}/web"

if [[ -x "${API_ROOT}/.venv/bin/python" ]]; then
  API_PYTHON="${API_ROOT}/.venv/bin/python"
  API_PYTHONPATH="${API_ROOT}:${APPS_ROOT}/../streamtimelens/src:${APPS_ROOT}/../src"
elif [[ -d "${API_ROOT}/.deps/fastapi" ]]; then
  API_PYTHON="python3"
  API_PYTHONPATH="${API_ROOT}/.deps:${API_ROOT}:${APPS_ROOT}/../streamtimelens/src:${APPS_ROOT}/../src"
else
  echo "Backend dependencies are missing. Run 'make setup' from the repository root."
  exit 1
fi
if [[ ! -d "${WEB_ROOT}/node_modules/next" ]]; then
  echo "Frontend dependencies are missing. Run 'make setup' from the repository root."
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd "${API_ROOT}"
  PYTHONPATH="${API_PYTHONPATH}" \
    "${API_PYTHON}" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
) &
API_PID=$!

cd "${WEB_ROOT}"
# Listen on all interfaces so Codex/SSH port-forwarders can discover the web
# server. FastAPI remains loopback-only and is reached through Next rewrites.
npm run dev -- --hostname 0.0.0.0
