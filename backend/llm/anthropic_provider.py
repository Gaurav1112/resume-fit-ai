"""Anthropic provider — the default, and the only one with server-side schema
enforcement in this app.

Notes that matter for correctness on current Claude models:

* `temperature` / `top_p` / `top_k` are **rejected with a 400** on Claude Opus 5,
  Fable 5, Opus 4.8 and 4.7. We never send them.
* Structured JSON goes through `output_config.format` with a `json_schema`.
  Assistant-turn prefills (the old way to force JSON) return a 400 on every
  current model.
* `effort` also lives inside `output_config`. Combining `effort` and `format` is
  supported; we degrade gracefully if a given model rejects the pairing.
* Thinking is on by default on Opus 5, and `max_tokens` caps thinking **plus**
  response text — so the stage budgets are generous.
* `stop_reason` must be checked before reading `content`: a refusal returns
  HTTP 200 with an empty/partial content array.
* The large stable prefix (master resume + JD) is sent as a cached system block,
  so repeated stages against the same documents read from cache at ~0.1x.
"""

from __future__ import annotations

from typing import Any

from .base import Call, LLMError, LLMRefusal, LLMTruncated, Provider, Usage


class AnthropicProvider(Provider):
    name = "anthropic"
    supports_native_schema = True

    def __init__(self, model: str, effort: str = "high", api_key: str = "") -> None:
        super().__init__(model, effort)
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment issue
            raise LLMError(
                "The `anthropic` package is not installed. Run: pip install anthropic"
            ) from exc
        # A bare client also resolves credentials from an `ant auth login` profile,
        # so we only pass the key when one was explicitly configured.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._anthropic = anthropic
        self._effort_supported = True

    def _build_kwargs(self, call: Call, with_effort: bool) -> dict[str, Any]:
        system: list[dict[str, Any]] = [{"type": "text", "text": call.system}]
        if call.cacheable_prefix:
            # Stable, large content goes last in the system array with a cache
            # breakpoint, so tools+system cache together for later stages.
            system.append(
                {
                    "type": "text",
                    "text": call.cacheable_prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": call.schema}
        }
        effort = call.effort or self.effort
        if with_effort and effort in {"low", "medium", "high", "xhigh", "max"}:
            output_config["effort"] = effort

        return {
            "model": self.model,
            "max_tokens": call.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": call.user}],
            "output_config": output_config,
        }

    def _complete(self, call: Call) -> tuple[str, Usage]:
        kwargs = self._build_kwargs(call, with_effort=self._effort_supported)
        try:
            response = self._client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as exc:
            message = str(exc)
            # Degrade gracefully if this model rejects effort alongside format,
            # or rejects structured outputs entirely.
            if self._effort_supported and "effort" in message:
                self._effort_supported = False
                response = self._client.messages.create(
                    **self._build_kwargs(call, with_effort=False)
                )
            elif "output_config" in message or "json_schema" in message:
                raise LLMError(
                    f"Model '{self.model}' does not support structured outputs. "
                    "Use claude-opus-5, claude-sonnet-5, claude-opus-4-8, "
                    "claude-fable-5 or claude-haiku-4-5."
                ) from exc
            else:
                raise LLMError(f"Anthropic rejected the request: {message}") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError(
                "Rate limited by the Anthropic API. Wait a moment and retry."
            ) from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API — check your network.") from exc
        except self._anthropic.AuthenticationError as exc:
            raise LLMError(
                "Anthropic authentication failed. Set ANTHROPIC_API_KEY in .env, "
                "or run `ant auth login`."
            ) from exc

        # Refusals return HTTP 200 — check stop_reason before touching content.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(
                category=getattr(details, "category", "") or "",
                explanation=getattr(details, "explanation", "") or "",
            )
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise LLMTruncated(
                f"Stage '{call.stage}' hit the {call.max_tokens}-token output cap. "
                "The input document may be unusually long."
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

        u = response.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            calls=1,
        )
        return text, usage
