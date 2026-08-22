"""Explainable weighted scoring engine.

Every number the UI shows is produced here from inspectable inputs. No score is
ever asked of an LLM, so two runs on the same inputs give the same answer, and
every component carries the reason it scored what it did.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.schemas import (
    CandidateProfile,
    JDAnalysis,
    MatchRow,
    Positioning,
    RecruiterView,
    ScoreComponent,
    ScoreReport,
    ValidationReport,
)
from . import ontology

# --------------------------------------------------------------------------- #
# Weights (configurable per request)
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS: dict[str, float] = {
    "keyword_coverage": 0.25,
    "semantic_alignment": 0.20,
    "required_skills": 0.15,
    "experience_alignment": 0.15,
    "title_alignment": 0.05,
    "evidence_strength": 0.05,
    "ats_format": 0.10,
    "recruiter_readability": 0.05,
}

LABELS = {
    "keyword_coverage": "JD keyword coverage",
    "semantic_alignment": "Semantic requirement alignment",
    "required_skills": "Required (P0) skills coverage",
    "experience_alignment": "Experience & relevance alignment",
    "title_alignment": "Job title / role alignment",
    "evidence_strength": "Achievement & evidence strength",
    "ats_format": "ATS parsing & format compliance",
    "recruiter_readability": "Recruiter readability",
}

# Priority weights used when averaging requirement scores.
PRIORITY_WEIGHT = {"P0": 3.0, "P1": 2.0, "P2": 1.0, "P3": 0.5}


def band_for(total: float) -> str:
    if total >= 95:
        return "Excellent"
    if total >= 90:
        return "Strong"
    if total >= 80:
        return "Good"
    if total >= 70:
        return "Needs improvement"
    return "Poor alignment"


@dataclass
class ScoringInputs:
    jd: JDAnalysis
    profile: CandidateProfile
    matrix: list[MatchRow]
    resume_text: str = ""
    ats_report: ValidationReport | None = None
    recruiter: RecruiterView | None = None
    positioning: Positioning | None = None


# --------------------------------------------------------------------------- #
# Component scorers — each returns (raw 0-100, explanation, details)
# --------------------------------------------------------------------------- #
def _keyword_coverage(inp: ScoringInputs) -> tuple[float, str, list[str]]:
    """Literal presence of JD keywords in the resume text.

    Deliberately literal: this is the component that models a naive keyword-count
    ATS. It only looks at the produced document, not at the evidence index.
    """
    if not inp.resume_text:
        # Pre-generation baseline: fall back to exact matches in the matrix.
        exact = [r for r in inp.matrix if r.match_type == "EXACT"]
        pct = 100.0 * len(exact) / len(inp.matrix) if inp.matrix else 0.0
        return pct, "Baseline: exact requirement matches before tailoring", []

    resume_terms = ontology.extract_known_terms(inp.resume_text)
    resume_norm = " " + ontology.normalise(inp.resume_text) + " "

    hit, miss = [], []
    weight_hit = weight_total = 0.0
    for row in inp.matrix:
        w = PRIORITY_WEIGHT.get(row.priority, 1.0)
        weight_total += w
        present = row.canonical in resume_terms or (
            len(row.requirement) >= 4
            and ontology.normalise(row.requirement) in resume_norm
        )
        if present:
            weight_hit += w
            hit.append(row.requirement)
        elif row.score >= 0.6:
            # Supported by evidence but absent from the document: recoverable miss.
            miss.append(f"{row.requirement} (supported but not surfaced)")
        else:
            miss.append(row.requirement)

    pct = 100.0 * weight_hit / weight_total if weight_total else 0.0
    details = [f"Present: {len(hit)}/{len(inp.matrix)} requirement keywords"]
    if miss:
        details.append("Missing from document: " + ", ".join(miss[:8]))
    return pct, "Priority-weighted presence of JD keywords in the resume text", details


def _semantic_alignment(inp: ScoringInputs) -> tuple[float, str, list[str]]:
    """Priority-weighted mean of the requirement match scores."""
    if not inp.matrix:
        return 0.0, "No requirements extracted from the JD", []
    num = sum(PRIORITY_WEIGHT.get(r.priority, 1.0) * r.score for r in inp.matrix)
    den = sum(PRIORITY_WEIGHT.get(r.priority, 1.0) for r in inp.matrix)
    pct = 100.0 * num / den
    tiers: dict[str, int] = {}
    for r in inp.matrix:
        tiers[r.match_type] = tiers.get(r.match_type, 0) + 1
    details = [f"{k}: {v}" for k, v in sorted(tiers.items(), key=lambda kv: -kv[1])]
    return pct, "Priority-weighted mean of requirement↔evidence match scores", details


def _required_skills(inp: ScoringInputs) -> tuple[float, str, list[str]]:
    """P0 coverage only. Optional requirements never drag this down."""
    p0 = [r for r in inp.matrix if r.priority == "P0"]
    if not p0:
        return 100.0, "JD declares no mandatory (P0) requirements", []
    covered = [r for r in p0 if r.score >= 0.6]
    missing = [r for r in p0 if r.score < 0.6]
    pct = 100.0 * sum(min(r.score / 0.85, 1.0) for r in p0) / len(p0)
    details = [f"Covered {len(covered)}/{len(p0)} mandatory requirements"]
    if missing:
        details.append("Unmet: " + ", ".join(r.requirement for r in missing[:8]))
    return pct, "Coverage of mandatory requirements only (optional ones excluded)", details


def _experience_alignment(inp: ScoringInputs) -> tuple[float, str, list[str]]:
    details: list[str] = []
    score = 100.0

    have = inp.profile.total_years_experience
    want = inp.jd.years_required
    if want and have is not None:
        ratio = have / want
        if ratio >= 1.0:
            details.append(f"{have:g}y experience meets the {want:g}y requirement")
        elif ratio >= 0.75:
            score -= 12
            details.append(f"{have:g}y vs {want:g}y required — slightly short")
        elif ratio >= 0.5:
            score -= 30
            details.append(f"{have:g}y vs {want:g}y required — materially short")
        else:
            score -= 50
            details.append(f"{have:g}y vs {want:g}y required — large shortfall")
    elif want and have is None:
        score -= 10
        details.append("Could not determine total years of experience from the resume")

    # Domain overlap.
    jd_domain = ontology.normalise(inp.jd.domain)
    if jd_domain:
        candidate_domains = {
            ontology.normalise(d)
            for d in [inp.profile.primary_domain, *inp.profile.secondary_domains, *inp.profile.domains]
            if d
        }
        if any(jd_domain in d or d in jd_domain for d in candidate_domains if d):
            details.append(f"Domain overlap: {inp.jd.domain}")
        else:
            score -= 8
            details.append(f"No direct domain experience in {inp.jd.domain}")

    # Leadership expectation.
    if inp.jd.leadership_expected:
        if inp.profile.has_leadership_experience:
            details.append("Leadership expectation is supported")
        else:
            score -= 15
            details.append("JD expects leadership; no leadership evidence in resume")

    return max(0.0, score), "Years, domain and leadership fit against the JD", details


def _title_alignment(inp: ScoringInputs) -> tuple[float, str, list[str]]:
    jd_title = ontology.normalise(inp.jd.job_title)
    if not jd_title:
        return 75.0, "JD title not detected", []

    titles = [inp.profile.current_title, *inp.profile.previous_titles]
    if inp.positioning and inp.positioning.target_title:
        titles.insert(0, inp.positioning.target_title)
    norm_titles = [ontology.normalise(t) for t in titles if t]

    jd_tokens = set(jd_title.split())
    best, best_title = 0.0, ""
    for t in norm_titles:
        tokens = set(t.split())
        if not tokens:
            continue
        overlap = len(jd_tokens & tokens) / max(1, len(jd_tokens))
        if overlap > best:
            best, best_title = overlap, t
    pct = 100.0 * best
    detail = (
        f"Closest title: '{best_title}' vs JD '{jd_title}'"
        if best_title
        else "No comparable title found"
    )
    return pct, "Overlap between candidate titles and the JD title", [detail]


def _evidence_strength(inp: ScoringInputs) -> tuple[float, str, list[str]]:
    """Rewards quantified, outcome-shaped bullets — but only real ones."""
    bullets: list[str] = []
    for role in inp.profile.roles:
        bullets.extend(role.bullets)
    if not bullets:
        return 30.0, "No experience bullets found to assess", []

    import re

    metric_re = re.compile(r"(\d+(\.\d+)?\s?%|\$\s?\d|\b\d{2,}\b|\b\d+(\.\d+)?[kmb]\b)", re.I)
    outcome_words = (
        "reduc", "improv", "increas", "decreas", "cut", "saved", "accelerat",
        "scaled", "eliminat", "enabled", "delivered", "launched", "migrat",
        "optimis", "optimiz", "unblock", "prevent",
    )
    action_words = (
        "designed", "built", "architected", "led", "implemented", "developed",
        "owned", "drove", "shipped", "automated", "refactored", "introduced",
    )

    quantified = sum(1 for b in bullets if metric_re.search(b))
    outcomes = sum(1 for b in bullets if any(w in b.lower() for w in outcome_words))
    actions = sum(1 for b in bullets if any(b.lower().lstrip().startswith(w) for w in action_words))

    n = len(bullets)
    pct = (
        45.0 * min(1.0, quantified / max(1, n * 0.4))
        + 30.0 * min(1.0, outcomes / max(1, n * 0.5))
        + 25.0 * min(1.0, actions / max(1, n * 0.6))
    )
    details = [
        f"{quantified}/{n} bullets carry a metric",
        f"{outcomes}/{n} bullets state an outcome",
        f"{actions}/{n} bullets open with a strong action verb",
    ]
    return pct, "Quality of achievement evidence in the source material", details


def _ats_format(inp: ScoringInputs) -> tuple[float, str, list[str]]:
    if inp.ats_report is None:
        return 85.0, "Format not yet validated (pre-generation baseline)", []
    failures = [c.label for c in inp.ats_report.checks if not c.passed]
    return (
        inp.ats_report.score(),
        "Deterministic ATS formatting checks on the generated document",
        (["Failed: " + ", ".join(failures[:8])] if failures else ["All format checks passed"]),
    )


def _recruiter(inp: ScoringInputs) -> tuple[float, str, list[str]]:
    if inp.recruiter is None:
        return 80.0, "Recruiter simulation not yet run", []
    return (
        inp.recruiter.score,
        "Simulated 10-second recruiter scan",
        inp.recruiter.top_weaknesses[:3],
    )


SCORERS = {
    "keyword_coverage": _keyword_coverage,
    "semantic_alignment": _semantic_alignment,
    "required_skills": _required_skills,
    "experience_alignment": _experience_alignment,
    "title_alignment": _title_alignment,
    "evidence_strength": _evidence_strength,
    "ats_format": _ats_format,
    "recruiter_readability": _recruiter,
}


def compute(inp: ScoringInputs, weights: dict[str, float] | None = None) -> ScoreReport:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: v for k, v in weights.items() if k in DEFAULT_WEIGHTS})
    total_weight = sum(w.values()) or 1.0

    components: list[ScoreComponent] = []
    total = 0.0
    for key, scorer in SCORERS.items():
        raw, explanation, details = scorer(inp)
        raw = max(0.0, min(100.0, raw))
        weight = w[key] / total_weight
        weighted = raw * weight
        total += weighted
        components.append(
            ScoreComponent(
                key=key,
                label=LABELS[key],
                weight=round(weight, 4),
                raw=round(raw, 1),
                weighted=round(weighted, 2),
                explanation=explanation,
                details=details,
            )
        )

    total = round(total, 1)
    return ScoreReport(total=total, band=band_for(total), components=components)


def jd_match_score(matrix: list[MatchRow]) -> float:
    """Standalone JD-match headline number (semantic alignment only)."""
    if not matrix:
        return 0.0
    num = sum(PRIORITY_WEIGHT.get(r.priority, 1.0) * r.score for r in matrix)
    den = sum(PRIORITY_WEIGHT.get(r.priority, 1.0) for r in matrix)
    return round(100.0 * num / den, 1)
