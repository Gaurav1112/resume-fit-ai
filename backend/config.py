"""Configuration loaded from environment / .env.

Deliberately dependency-free (no pydantic-settings) so the config surface is one
readable file. API keys are read here and never leave the backend process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader. Existing environment variables always win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    """Like `os.getenv`, but treats a blank value as absent.

    `os.getenv(name, default)` only returns the default when the variable is
    unset. A variable that is *present but empty* returns "" — and hosting
    platforms and dashboard-pasted .env files produce empty variables routinely.
    Vercel sets an empty PORT, which made `int(os.getenv("PORT", "8000"))` raise
    at import time and took the entire application down before it served a
    single request.
    """
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    """Blank-tolerant integer read that also survives a non-numeric value.

    Config is read at import time, so a bad value here is not a bad request —
    it is a process that never starts.
    """
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


# Default model per provider. These are the current flagship ids as of the
# claude-api reference bundled with this project; override with LLM_MODEL.
DEFAULT_MODELS = {
    "local": "local-rules",
    "ollama": "qwen2.5:7b",
    "anthropic": "claude-opus-5",
    "openai": "gpt-4.1",
    "gemini": "gemini-2.5-pro",
    "mock": "mock-1",
}


@dataclass(frozen=True)
class Settings:
    provider: str = _env("LLM_PROVIDER", "local").lower()
    model: str = ""
    effort: str = _env("LLM_EFFORT", "high").lower()

    anthropic_api_key: str = _env("ANTHROPIC_API_KEY", "")
    openai_api_key: str = _env("OPENAI_API_KEY", "")
    google_api_key: str = _env("GOOGLE_API_KEY", "")

    host: str = _env("HOST", "127.0.0.1")
    port: int = _env_int("PORT", 8000)

    db_path: Path = field(
        default_factory=lambda: Path(_env("DB_PATH", str(ROOT / "data" / "resumefit.db")))
    )
    max_upload_bytes: int = _env_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    llm_trace: bool = _env("LLM_TRACE", "0") == "1"

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(
                self, "model", os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(self.provider, "")
            )
        object.__setattr__(self, "db_path", Path(self.db_path))
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Read-only filesystem (serverless). Storage is optional — db.init()
            # will mark itself unavailable and every call degrades gracefully.
            # Crashing here would take the whole app down at import time for a
            # feature that is not needed to produce a resume.
            pass

    @property
    def api_key(self) -> str:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.google_api_key,
        }.get(self.provider, "")

    @property
    def configured(self) -> bool:
        """True when the selected provider can actually be called."""
        if self.provider in ("local", "mock", "ollama"):
            return True    # no credential required
        if self.provider == "anthropic":
            # The Anthropic SDK also resolves credentials from `ant auth login`
            # profiles, so an unset key does not necessarily mean unconfigured.
            return True
        return bool(self.api_key)


settings = Settings()
