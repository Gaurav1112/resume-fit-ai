"""Offline provider for tests and for exercising the scoring/validation engine.

Synthesises a schema-shaped response from the input text using simple heuristics.
It is not intelligent — its job is to let the deterministic half of the system
(matching, scoring, ATS checks, truth gate, loops, exporters) be tested end-to-end
with no network and no API key.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..services import ontology
from .base import Call, Provider, Usage

_BULLET = re.compile(r"^\s*[-•*•]\s*(.+)$", re.M)


def _sample_bullets(text: str, limit: int = 6) -> list[str]:
    found = [b.strip() for b in _BULLET.findall(text)]
    if not found:
        found = [ln.strip() for ln in text.splitlines() if 40 < len(ln.strip()) < 220]
    return found[:limit]


def _default_for(schema: dict[str, Any], ctx: dict[str, Any]) -> Any:
    """Walk a JSON Schema and produce a minimally valid instance."""
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        out: dict[str, Any] = {}
        for key, sub in (schema.get("properties") or {}).items():
            if key in ctx:
                out[key] = ctx[key]
            else:
                out[key] = _default_for(sub, ctx)
        return out
    if t == "array":
        item = schema.get("items", {"type": "string"})
        return [_default_for(item, ctx)] if item.get("type") == "object" else []
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    if isinstance(t, list) and "null" in t:
        return None
    return ""


class MockProvider(Provider):
    name = "mock"
    supports_native_schema = True

    def _complete(self, call: Call) -> tuple[str, Usage]:
        source = f"{call.cacheable_prefix}\n{call.user}"
        terms = sorted(ontology.extract_known_terms(source))
        bullets = _sample_bullets(source)

        ctx: dict[str, Any] = {}
        stage = call.stage

        if stage == "profile":
            ctx = {
                "contact": {
                    "name": "Sample Candidate",
                    "email": "candidate@example.com",
                    "phone": "",
                    "location": "",
                    "linkedin": "",
                    "github": "",
                    "portfolio": "",
                },
                "current_title": "Software Engineer",
                "previous_titles": [],
                "total_years_experience": 4.0,
                "primary_domain": "software",
                "secondary_domains": [],
                "has_leadership_experience": False,
                "leadership_summary": "",
                "skill_groups": [
                    {"category": "Languages", "skills": terms[:12]},
                ],
                "roles": [
                    {
                        "company": "Example Corp",
                        "title": "Software Engineer",
                        "start_date": "Jan 2022",
                        "end_date": "Present",
                        "location": "",
                        "employment_type": "Full-time",
                        "bullets": bullets or ["Built and shipped backend services."],
                        "technologies": terms[:10],
                        "achievements": [],
                        "metrics": [],
                        "leadership": [],
                        "business_impact": [],
                    }
                ],
                "education": [],
                "certifications": [],
                "projects": [],
                "achievements": [],
                "domains": [],
                "evidence": [
                    {
                        "skill": t,
                        "evidence": [f"Referenced in the master resume ({t})"],
                        "sources": ["Master resume"],
                        "confidence": "MEDIUM",
                        "years": None,
                    }
                    for t in terms[:25]
                ],
            }
        elif stage == "jd":
            reqs = []
            for i, t in enumerate(terms[:20], start=1):
                reqs.append(
                    {
                        "id": f"R{i}",
                        "text": t,
                        "canonical": t,
                        "category": ontology.category_of(t).lower(),
                        "priority": "P0" if i <= 5 else ("P1" if i <= 12 else "P2"),
                        "kind": "REQUIRED" if i <= 12 else "PREFERRED",
                        "years_required": None,
                        "rationale": "Mentioned in the job description.",
                    }
                )
            ctx = {
                "job_title": "Software Engineer",
                "company": "",
                "seniority": "mid",
                "years_required": None,
                "location": "",
                "work_mode": "unspecified",
                "work_authorization": "",
                "domain": "software",
                "leadership_expected": False,
                "requirements": reqs,
                "responsibilities": bullets[:4],
                "qualifications": [],
                "soft_skills": [],
                "certifications": [],
                "key_phrases": terms[:10],
            }
        elif stage == "positioning":
            ctx = {
                "target_title": "Software Engineer",
                "target_seniority": "mid",
                "identity_statement": "Backend-leaning software engineer.",
                "supported": True,
                "support_reasoning": "Titles and evidence align with the target role.",
                "differentiators": terms[:4],
                "emphasise": terms[:6],
                "de_emphasise": [],
                "section_order": [
                    "summary", "skills", "experience", "projects", "education", "certifications",
                ],
            }
        elif stage == "writer":
            ctx = {
                "headline": "Software Engineer",
                "summary": (
                    "Software engineer with experience building and shipping backend "
                    "services."
                ),
                "skill_groups": [{"category": "Languages", "skills": terms[:12]}],
                "roles": [
                    {
                        "company": "Example Corp",
                        "title": "Software Engineer",
                        "start_date": "Jan 2022",
                        "end_date": "Present",
                        "location": "",
                        "bullets": [
                            {"text": b, "source_ref": "Example Corp", "keywords": []}
                            for b in (bullets or ["Built and shipped backend services."])
                        ],
                    }
                ],
                "selected_projects": [],
                "changes": [
                    {
                        "change": "Generated a baseline tailored resume.",
                        "reason": "Mock provider — no LLM was called.",
                        "source": "Master resume",
                        "category": "rewritten",
                    }
                ],
            }
        elif stage == "recruiter":
            ctx = {
                "score": 78.0,
                "who_is_this": "A software engineer.",
                "what_level": "Mid-level",
                "specialisation": "Backend services",
                "technologies_visible": terms[:8],
                "relevance_to_role": "Broadly relevant.",
                "top_strengths": terms[:5],
                "top_weaknesses": ["Mock provider — not a real assessment."],
            }
        elif stage == "truth":
            ctx = {"claims": [], "verdict": "pass", "notes": "Mock provider — no LLM review."}

        payload = _default_for(call.schema, ctx)
        return json.dumps(payload), Usage(calls=1)
