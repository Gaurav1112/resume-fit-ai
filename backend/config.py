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


# Default model per provider. These are the current flagship ids as of the
# claude-api reference bundled with this project; override with LLM_MODEL.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4.1",
    "gemini": "gemini-2.5-pro",
    "mock": "mock-1",
}


@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    model: str = ""
    effort: str = os.getenv("LLM_EFFORT", "high").strip().lower()

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))

    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("DB_PATH", str(ROOT / "data" / "resumefit.db")))
    )
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    llm_trace: bool = os.getenv("LLM_TRACE", "0") == "1"

    def __post_init__(self) -> None:
        if not self.model:
            object.__setattr__(
                self, "model", os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(self.provider, "")
            )
        object.__setattr__(self, "db_path", Path(self.db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

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
        if self.provider == "mock":
            return True
        if self.provider == "anthropic":
            # The Anthropic SDK also resolves credentials from `ant auth login`
            # profiles, so an unset key does not necessarily mean unconfigured.
            return True
        return bool(self.api_key)


settings = Settings()
