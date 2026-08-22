#!/usr/bin/env bash
# ResumeFit AI — one-command local run.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
VENV=.venv

if [ ! -d "$VENV" ]; then
  echo "→ Creating virtualenv…"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  echo "→ Installing dependencies…"
  "$VENV/bin/pip" install --quiet -r requirements.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo
  echo "  Created .env from .env.example."
  echo "  Add your ANTHROPIC_API_KEY to .env, then run this script again."
  echo "  (Or set LLM_PROVIDER=mock in .env to explore the UI with no API key.)"
  echo
  exit 1
fi

PORT=$(grep -E '^PORT=' .env | cut -d= -f2 | tr -d '[:space:]' || true)
PORT=${PORT:-8000}
HOST=$(grep -E '^HOST=' .env | cut -d= -f2 | tr -d '[:space:]' || true)
HOST=${HOST:-127.0.0.1}

echo "→ ResumeFit AI on http://${HOST}:${PORT}"
exec "$VENV/bin/uvicorn" backend.main:app --host "$HOST" --port "$PORT" --reload
