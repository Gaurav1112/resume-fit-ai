"""The default provider: no API key, no network, no model.

Every stage is dispatched to the deterministic rules engine in
`services/local_engine.py`. It satisfies the same `Provider` contract as the
model-backed providers, so the graph, the loops, both validators, the scoring
engine and the exporters all run completely unchanged.

Two stages are deliberate no-ops here:

* `refine` — its whole job is to adjudicate rows the deterministic matcher was
  unsure about. With no model there is nothing better to ask, and guessing would
  be worse than the matcher's own answer, so it returns no changes.
* `truth` — the LLM claim audit is a *second* opinion layered on the
  deterministic gate. The deterministic gate has already run and is authoritative;
  since this engine cannot fabricate by construction, there is no second class of
  error for a claim audit to catch.
"""

from __future__ import annotations

from typing import Any

from ..services import local_engine
from .base import Call, Provider, Usage


class LocalProvider(Provider):
    name = "local"
    supports_native_schema = True
    deterministic = True

    def __init__(self, model: str = "local-rules", effort: str = "") -> None:
        super().__init__(model or "local-rules", effort)

    def _complete(self, call: Call) -> tuple[str, Usage]:  # pragma: no cover
        raise NotImplementedError("LocalProvider overrides json() directly")

    def json(self, call: Call, *, retries: int = 0) -> dict[str, Any]:
        p = call.payload or {}
        self.usage.add(Usage(calls=1))

        if call.stage == "profile":
            return local_engine.parse_resume(p.get("resume_text", ""))

        if call.stage == "jd":
            return local_engine.analyse_jd(
                p.get("jd_text", ""), p.get("market", "global")
            )

        if call.stage == "refine":
            return {"rows": []}

        if call.stage == "positioning":
            return local_engine.decide_positioning(
                p.get("profile", {}), p.get("jd", {}), p.get("matrix", [])
            )

        if call.stage == "writer":
            return local_engine.write_resume(
                p.get("profile", {}),
                p.get("jd", {}),
                p.get("matrix", []),
                p.get("positioning", {}),
                p.get("master_text", ""),
            )

        if call.stage == "recruiter":
            return local_engine.simulate_recruiter(
                p.get("resume_text", ""), p.get("jd", {}), p.get("matrix", [])
            )

        if call.stage == "truth":
            return {
                "claims": [],
                "verdict": "pass",
                "notes": "Local engine: bullets are reproduced verbatim from the master "
                         "resume, so there is no generative step that could introduce an "
                         "unsupported claim. The deterministic gate remains authoritative.",
            }

        raise ValueError(f"LocalProvider has no handler for stage '{call.stage}'")
