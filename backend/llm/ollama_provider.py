"""Optional local LLM via Ollama — still no API key, still fully offline.

Hybrid by design: the rules engine handles every stage it does deterministically
well (parsing, JD analysis, positioning, the recruiter scan), and the model is
used for the one thing rules genuinely cannot do — rewriting a bullet into
action + technology + outcome form.

Even then the model is kept on a short leash. It rewrites one bullet at a time,
is shown only that bullet, and its output is rejected and the original restored
if it introduces a number that was not already there. A small local model *will*
hallucinate a metric given the chance; this makes that failure mode inert rather
than trusting the prompt.

Setup:
    brew install ollama          (or: curl -fsSL https://ollama.com/install.sh | sh)
    ollama pull qwen2.5:7b
    LLM_PROVIDER=ollama in .env
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from ..services import local_engine
from .base import Call, LLMError, Provider, Usage
from .local_provider import LocalProvider

_NUM = re.compile(r"\d[\d,.]*")

REWRITE_SYSTEM = """You rewrite a single resume bullet to be stronger and more scannable.

Rules, in priority order:
1. Introduce no new facts. No number, technology, team size, job title or outcome
   may appear in your output unless it appears in the input. This is absolute.
2. Start with a strong past-tense action verb.
3. Keep every technology named in the input.
4. One sentence, under 240 characters.
5. If you cannot improve it without breaking rule 1, return it unchanged.

Reply with the rewritten bullet as plain text. No quotes, no preamble, no explanation."""


class OllamaProvider(Provider):
    name = "ollama"
    supports_native_schema = False

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        effort: str = "",
        host: str = "http://localhost:11434",
    ) -> None:
        super().__init__(model or "qwen2.5:7b", effort)
        self.host = host.rstrip("/")
        self._fallback = LocalProvider()

    # -- transport ---------------------------------------------------------
    def _post(self, path: str, body: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.host}. Start it with `ollama serve`, "
                f"and make sure the model is pulled: `ollama pull {self.model}`. ({exc})"
            ) from exc

    def _generate(self, system: str, prompt: str, max_tokens: int = 400) -> str:
        body = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        result = self._post("/api/generate", body)
        self.usage.add(
            Usage(
                input_tokens=result.get("prompt_eval_count", 0) or 0,
                output_tokens=result.get("eval_count", 0) or 0,
                calls=1,
            )
        )
        return (result.get("response") or "").strip()

    def available(self) -> bool:
        try:
            tags = self._post("/api/tags", {}, timeout=5)
        except LLMError:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return any(n == self.model or n.startswith(self.model.split(":")[0]) for n in names)

    # -- rewriting ---------------------------------------------------------
    @staticmethod
    def _safe(original: str, rewritten: str) -> bool:
        """Reject a rewrite that invents a number or drops a technology."""
        candidate = rewritten.strip().strip('"').strip()
        if not (25 <= len(candidate) <= 300):
            return False
        if "\n" in candidate:
            return False
        source_numbers = {n.replace(",", "") for n in _NUM.findall(original)}
        for number in _NUM.findall(candidate):
            if number.replace(",", "") not in source_numbers:
                return False
        from ..services import ontology

        if not ontology.extract_known_terms(original) <= ontology.extract_known_terms(candidate):
            return False
        return True

    def _rewrite_bullets(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = local_engine.write_resume(
            payload.get("profile", {}),
            payload.get("jd", {}),
            payload.get("matrix", []),
            payload.get("positioning", {}),
            payload.get("master_text", ""),
        )
        jd_title = payload.get("jd", {}).get("job_title", "the role")
        rewritten_count = rejected_count = 0

        for role in base["roles"]:
            for bullet in role["bullets"]:
                original = bullet["text"]
                try:
                    candidate = self._generate(
                        REWRITE_SYSTEM,
                        f"Target role: {jd_title}\n\nBullet:\n{original}",
                        max_tokens=200,
                    )
                except LLMError:
                    continue
                candidate = candidate.strip().strip('"')
                if candidate and candidate != original and self._safe(original, candidate):
                    bullet["text"] = local_engine.clean_bullet(candidate)
                    rewritten_count += 1
                elif candidate and candidate != original:
                    rejected_count += 1

        base["changes"].append({
            "change": f"Reworded {rewritten_count} bullet(s) into action + technology + "
                      f"outcome form using a local model.",
            "reason": f"{rejected_count} further rewrite(s) were rejected and the original "
                      "restored, because they introduced a number or dropped a technology "
                      "that was not in your resume.",
            "source": "Master resume → experience bullets",
            "category": "rewritten",
        })
        return base

    # -- Provider contract -------------------------------------------------
    def _complete(self, call: Call) -> tuple[str, Usage]:  # pragma: no cover
        raise NotImplementedError("OllamaProvider overrides json() directly")

    def json(self, call: Call, *, retries: int = 0) -> dict[str, Any]:
        if call.stage == "writer":
            return self._rewrite_bullets(call.payload or {})
        # Every other stage is handled better by rules than by a 7B model.
        return self._fallback.json(call, retries=retries)
