"""Vercel serverless entry point.

Vercel's Python runtime looks for a module-level ASGI app called `app` under
`api/`. Everything else is the same application that runs locally.

On this host the filesystem is read-only, so `backend.db` degrades to
unavailable: analysis, generation, scoring, validation and export all still work
(they are pure functions of their inputs), while version history and the
application tracker return empty. The frontend posts the analysis back to
`/api/generate` and the document back to `/api/render/...`, so no request depends
on state left behind by a previous one.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# /tmp is the only writable path on Vercel. It is per-instance and ephemeral,
# which is fine: nothing correctness-critical is stored there.
os.environ.setdefault("DB_PATH", "/tmp/resumefit.db")
os.environ.setdefault("LLM_PROVIDER", "local")

from backend.main import app  # noqa: E402

__all__ = ["app"]
