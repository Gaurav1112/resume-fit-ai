"""Vercel serverless entry point.

Vercel's Python runtime looks for a module-level ASGI app called `app` under
`api/`. Everything else is the same application that runs locally.

On this host the filesystem is read-only, so `backend.db` degrades to
unavailable: analysis, generation, scoring, validation and export all still work
(they are pure functions of their inputs), while version history and the
application tracker return empty. The frontend posts the analysis back to
`/api/generate` and the document back to `/api/render/...`, so no request depends
on state left behind by a previous one.

Two constraints shape the structure below, both learned the hard way:

1. The builder locates the entrypoint by parsing this file's AST, not by
   importing it. It needs a *top-level* binding called `app` — a definition
   nested inside `try:` is invisible to a static scan and fails the build with
   "Could not find a top-level app". Hence the plain assignment at the end.
2. A failed import otherwise surfaces as a bare FUNCTION_INVOCATION_FAILED with
   the traceback only in the dashboard, costing a deploy cycle per guess. So the
   import is caught and served as JSON by a framework-free ASGI app — the
   framework being one of the candidates for what broke.
"""

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# /tmp is the only writable path on Vercel. It is per-instance and ephemeral,
# which is fine: nothing correctness-critical is stored there.
os.environ.setdefault("DB_PATH", "/tmp/resumefit.db")
os.environ.setdefault("LLM_PROVIDER", "local")

_IMPORT_ERROR: dict | None = None
_application = None

try:
    from backend.main import app as _application
except BaseException:  # noqa: BLE001 - anything at all must be reportable
    _IMPORT_ERROR = {
        "error": "the application failed to import",
        "python": sys.version,
        "cwd": os.getcwd(),
        "entrypoint_dir": str(Path(__file__).resolve().parent),
        "root_on_path": str(ROOT),
        "root_exists": ROOT.exists(),
        "root_contents": sorted(p.name for p in ROOT.iterdir()) if ROOT.exists() else [],
        "backend_package_found": (ROOT / "backend" / "__init__.py").exists(),
        "sys_path": sys.path,
        "traceback": traceback.format_exc().splitlines(),
    }


async def _report_import_error(scope, receive, send):
    """Minimal ASGI app — no framework, since the framework may be what broke."""
    if scope["type"] != "http":
        return
    body = json.dumps(_IMPORT_ERROR, indent=2).encode()
    await send({
        "type": "http.response.start",
        "status": 500,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({"type": "http.response.body", "body": body})


# Top-level and unconditional: this is the name the builder scans for.
app = _application if _application is not None else _report_import_error

__all__ = ["app"]
