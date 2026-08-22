"""Gemini provider (optional). JSON MIME type + schema-in-prompt."""

from __future__ import annotations

from .base import Call, LLMError, Provider, Usage, schema_instruction


class GeminiProvider(Provider):
    name = "gemini"
    supports_native_schema = False

    def __init__(self, model: str, effort: str = "high", api_key: str = "") -> None:
        super().__init__(model, effort)
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError(
                "The `google-genai` package is not installed. Run: pip install google-genai"
            ) from exc
        self._genai = genai
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _complete(self, call: Call) -> tuple[str, Usage]:
        system = call.system
        if call.cacheable_prefix:
            system = f"{system}\n\n{call.cacheable_prefix}"
        system = f"{system}\n\n{schema_instruction(call.schema)}"

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=call.user,
                config={
                    "system_instruction": system,
                    "response_mime_type": "application/json",
                    "max_output_tokens": call.max_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001 - SDK exception surface varies
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = getattr(response, "text", "") or ""
        meta = getattr(response, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            calls=1,
        )
        return text, usage
