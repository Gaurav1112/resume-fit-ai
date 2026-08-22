"""Provider-agnostic LLM interface.

Every pipeline stage calls `provider.json(...)` with a JSON Schema and gets back a
validated dict. Providers that support server-side schema enforcement use it;
the rest fall back to schema-in-prompt plus a parse-and-retry loop, so the
contract at the call site is identical either way.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ROOT, settings


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    """The provider's safety classifiers declined the request."""

    def __init__(self, category: str = "", explanation: str = "") -> None:
        super().__init__(
            f"The model declined this request (category={category or 'unspecified'}). "
            f"{explanation}".strip()
        )
        self.category = category
        self.explanation = explanation


class LLMTruncated(LLMError):
    """Generation hit the output cap before producing complete JSON."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.calls += other.calls

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


@dataclass
class Call:
    """One structured-output request."""

    stage: str
    system: str
    user: str
    schema: dict[str, Any]
    max_tokens: int = 16000
    cacheable_prefix: str = ""       # large stable content, cached where supported
    effort: str = ""


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON recovery from a model response."""
    text = (text or "").strip()
    if not text:
        raise LLMError("empty response from model")

    for candidate in (text, *(m.strip() for m in _JSON_BLOCK.findall(text))):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise LLMError(f"could not parse JSON from model response (first 200 chars): {text[:200]}")


class Provider(ABC):
    name: str = "abstract"
    supports_native_schema: bool = False

    def __init__(self, model: str, effort: str = "high") -> None:
        self.model = model
        self.effort = effort
        self.usage = Usage()

    @abstractmethod
    def _complete(self, call: Call) -> tuple[str, Usage]:
        """Return (raw text, usage)."""

    def json(self, call: Call, *, retries: int = 2) -> dict[str, Any]:
        """Run a call and return parsed JSON, retrying on parse failure."""
        last: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                text, usage = self._complete(call)
                self.usage.add(usage)
                _trace(call, text)
                return extract_json(text)
            except LLMRefusal:
                raise
            except (LLMError, json.JSONDecodeError) as exc:
                last = exc
                if attempt <= retries:
                    call = Call(
                        **{
                            **call.__dict__,
                            "user": call.user
                            + "\n\nYour previous response was not valid JSON matching the "
                            "schema. Respond with the JSON object only — no prose, no "
                            "markdown fences.",
                        }
                    )
                    time.sleep(0.8 * attempt)
        raise LLMError(f"stage '{call.stage}' failed after {retries + 1} attempts: {last}")


def _trace(call: Call, response: str) -> None:
    if not settings.llm_trace:
        return
    path = ROOT / "data" / "llm_trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": call.stage,
        "system": call.system[:4000],
        "user": call.user[:20000],
        "response": response[:20000],
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def schema_instruction(schema: dict[str, Any]) -> str:
    """Prompt fragment for providers without native schema enforcement."""
    return (
        "Respond with a single JSON object and nothing else. It MUST validate "
        "against this JSON Schema:\n\n"
        + json.dumps(schema, indent=2)
        + "\n\nDo not wrap the JSON in markdown fences. Do not add commentary."
    )
