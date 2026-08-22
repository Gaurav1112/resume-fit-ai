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
  echo "→ Created .env (defaults to the local rules engine — no API key needed)"
fi

PORT=$(grep -E '^PORT=' .env | cut -d= -f2 | tr -d '[:space:]' || true)
PORT=${PORT:-8000}
HOST=$(grep -E '^HOST=' .env | cut -d= -f2 | tr -d '[:space:]' || true)
HOST=${HOST:-127.0.0.1}

echo "→ ResumeFit AI on http://${HOST}:${PORT}"
exec "$VENV/bin/uvicorn" backend.main:app --host "$HOST" --port "$PORT" --reload
