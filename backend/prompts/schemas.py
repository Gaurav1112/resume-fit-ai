"""JSON Schemas for every LLM stage.

Constraints these must respect (Anthropic structured outputs):
  * every object needs `additionalProperties: false`
  * no numeric constraints (minimum/maximum/multipleOf)
  * no string-length constraints (minLength/maxLength)
  * no recursive schemas
Optional values are expressed with `anyOf: [T, null]`, which is supported.

Dictionaries with arbitrary keys are not expressible, so `skills` is modelled as
a list of {category, skills} pairs and converted to a dict in the pipeline.
"""

from __future__ import annotations

from typing import Any


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


def _arr(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


STR = {"type": "string"}
BOOL = {"type": "boolean"}
STRS = _arr(STR)
NULLABLE_NUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}

CONFIDENCE = {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}
PRIORITY = {"type": "string", "enum": ["P0", "P1", "P2", "P3"]}
KIND = {"type": "string", "enum": ["REQUIRED", "PREFERRED", "OPTIONAL", "NICE_TO_HAVE"]}

SKILL_GROUP = _obj({"category": STR, "skills": STRS})

# --------------------------------------------------------------------------- #
# Stage 1-2: resume parse -> candidate profile + evidence database
# --------------------------------------------------------------------------- #
PROFILE_SCHEMA = _obj(
    {
        "contact": _obj(
            {
                "name": STR,
                "email": STR,
                "phone": STR,
                "location": STR,
                "linkedin": STR,
                "github": STR,
                "portfolio": STR,
            }
        ),
        "current_title": STR,
        "previous_titles": STRS,
        "total_years_experience": NULLABLE_NUM,
        "primary_domain": STR,
        "secondary_domains": STRS,
        "has_leadership_experience": BOOL,
        "leadership_summary": STR,
        "skill_groups": _arr(SKILL_GROUP),
        "roles": _arr(
            _obj(
                {
                    "company": STR,
                    "title": STR,
                    "start_date": STR,
                    "end_date": STR,
                    "location": STR,
                    "employment_type": STR,
                    "bullets": STRS,
                    "technologies": STRS,
                    "achievements": STRS,
                    "metrics": STRS,
                    "leadership": STRS,
                    "business_impact": STRS,
                }
            )
        ),
        "education": _arr(
            _obj(
                {
                    "institution": STR,
                    "degree": STR,
                    "field_of_study": STR,
                    "start_date": STR,
                    "end_date": STR,
                    "location": STR,
                    "details": STRS,
                }
            )
        ),
        "certifications": _arr(
            _obj({"name": STR, "issuer": STR, "date": STR, "credential_id": STR})
        ),
        "projects": _arr(
            _obj(
                {
                    "name": STR,
                    "description": STR,
                    "technologies": STRS,
                    "bullets": STRS,
                    "url": STR,
                }
            )
        ),
        "achievements": STRS,
        "domains": STRS,
        "evidence": _arr(
            _obj(
                {
                    "skill": STR,
                    "evidence": STRS,
                    "sources": STRS,
                    "confidence": CONFIDENCE,
                    "years": NULLABLE_NUM,
                }
            )
        ),
    }
)

# --------------------------------------------------------------------------- #
# Stage 3-4: JD intelligence + priority classification
# --------------------------------------------------------------------------- #
JD_SCHEMA = _obj(
    {
        "job_title": STR,
        "company": STR,
        "seniority": STR,
        "years_required": NULLABLE_NUM,
        "location": STR,
        "work_mode": {
            "type": "string",
            "enum": ["remote", "hybrid", "onsite", "unspecified"],
        },
        "work_authorization": STR,
        "domain": STR,
        "leadership_expected": BOOL,
        "requirements": _arr(
            _obj(
                {
                    "id": STR,
                    "text": STR,
                    "canonical": STR,
                    "category": STR,
                    "priority": PRIORITY,
                    "kind": KIND,
                    "years_required": NULLABLE_NUM,
                    "rationale": STR,
                }
            )
        ),
        "responsibilities": STRS,
        "qualifications": STRS,
        "soft_skills": STRS,
        "certifications": STRS,
        "key_phrases": STRS,
    }
)

# --------------------------------------------------------------------------- #
# Stage 5b: LLM refinement of ambiguous matrix rows
# --------------------------------------------------------------------------- #
REFINE_SCHEMA = _obj(
    {
        "rows": _arr(
            _obj(
                {
                    "requirement_id": STR,
                    "match_type": {
                        "type": "string",
                        "enum": ["EXACT", "STRONG_SEMANTIC", "PARTIAL", "WEAK", "NONE"],
                    },
                    "score": {"type": "number"},
                    "confidence": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW", "NONE"],
                    },
                    "matched_via": STR,
                    "evidence": STRS,
                    "notes": STR,
                }
            )
        )
    }
)

# --------------------------------------------------------------------------- #
# Stage 7: positioning
# --------------------------------------------------------------------------- #
POSITIONING_SCHEMA = _obj(
    {
        "target_title": STR,
        "target_seniority": STR,
        "identity_statement": STR,
        "supported": BOOL,
        "support_reasoning": STR,
        "differentiators": STRS,
        "emphasise": STRS,
        "de_emphasise": STRS,
        "section_order": STRS,
    }
)

# --------------------------------------------------------------------------- #
# Stage 8: resume writer
# --------------------------------------------------------------------------- #
RESUME_BULLET = _obj({"text": STR, "source_ref": STR, "keywords": STRS})

WRITER_SCHEMA = _obj(
    {
        "headline": STR,
        "summary": STR,
        "skill_groups": _arr(SKILL_GROUP),
        "roles": _arr(
            _obj(
                {
                    "company": STR,
                    "title": STR,
                    "start_date": STR,
                    "end_date": STR,
                    "location": STR,
                    "bullets": _arr(RESUME_BULLET),
                }
            )
        ),
        "selected_projects": _arr(
            _obj({"name": STR, "description": STR, "bullets": _arr(RESUME_BULLET)})
        ),
        "changes": _arr(
            _obj(
                {
                    "change": STR,
                    "reason": STR,
                    "source": STR,
                    "category": {
                        "type": "string",
                        "enum": [
                            "added", "removed", "rewritten", "reordered", "repositioned",
                        ],
                    },
                }
            )
        ),
    }
)

# --------------------------------------------------------------------------- #
# Stage 10b: LLM claim validation (second gate after the deterministic one)
# --------------------------------------------------------------------------- #
TRUTH_SCHEMA = _obj(
    {
        "claims": _arr(
            _obj(
                {
                    "claim": STR,
                    "supported": BOOL,
                    "source_quote": STR,
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "warning", "info"],
                    },
                    "explanation": STR,
                }
            )
        ),
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "notes": STR,
    }
)

# --------------------------------------------------------------------------- #
# Stage 11: recruiter simulation
# --------------------------------------------------------------------------- #
RECRUITER_SCHEMA = _obj(
    {
        "score": {"type": "number"},
        "who_is_this": STR,
        "what_level": STR,
        "specialisation": STR,
        "technologies_visible": STRS,
        "relevance_to_role": STR,
        "top_strengths": STRS,
        "top_weaknesses": STRS,
    }
)
