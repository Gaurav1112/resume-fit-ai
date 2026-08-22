"""OpenAI provider (optional). Uses JSON mode + schema-in-prompt.

Deliberately not using OpenAI's `strict: true` json_schema mode: it requires every
property to appear in `required` and forbids optional fields, which would mean
maintaining a second, divergent set of schemas. JSON mode plus the shared
schema-in-prompt instruction and the base class's parse-retry gets equivalent
reliability from one schema definition.
"""

from __future__ import annotations

from .base import Call, LLMError, Provider, Usage, schema_instruction


class OpenAIProvider(Provider):
    name = "openai"
    supports_native_schema = False

    def __init__(self, model: str, effort: str = "high", api_key: str = "") -> None:
        super().__init__(model, effort)
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError(
                "The `openai` package is not installed. Run: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def _complete(self, call: Call) -> tuple[str, Usage]:
        system = call.system
        if call.cacheable_prefix:
            system = f"{system}\n\n{call.cacheable_prefix}"
        system = f"{system}\n\n{schema_instruction(call.schema)}"

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=call.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": call.user},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - SDK exception surface varies
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        text = response.choices[0].message.content or ""
        u = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            calls=1,
        )
        return text, usage
