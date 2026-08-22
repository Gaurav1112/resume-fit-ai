"""Typed contracts for every stage of the pipeline.

Every LLM stage returns JSON that is validated into one of these models before
the next stage sees it. No stage ever receives raw unstructured text from a
previous stage (only the original documents are raw).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Priority = Literal["P0", "P1", "P2", "P3"]
Kind = Literal["REQUIRED", "PREFERRED", "OPTIONAL", "NICE_TO_HAVE"]
Confidence = Literal["HIGH", "MEDIUM", "LOW", "NONE"]
MatchType = Literal["EXACT", "STRONG_SEMANTIC", "PARTIAL", "WEAK", "NONE"]


class Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# --------------------------------------------------------------------------- #
# Candidate profile (stages 1–2)
# --------------------------------------------------------------------------- #
class Contact(Base):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""


class Role(Base):
    company: str = ""
    title: str = ""
    start_date: str = ""          # verbatim from the resume, e.g. "Mar 2023"
    end_date: str = ""            # verbatim, e.g. "Present"
    location: str = ""
    employment_type: str = ""
    bullets: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    leadership: list[str] = Field(default_factory=list)
    business_impact: list[str] = Field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.title} @ {self.company}"


class Education(Base):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_date: str = ""
    end_date: str = ""
    location: str = ""
    details: list[str] = Field(default_factory=list)


class Certification(Base):
    name: str = ""
    issuer: str = ""
    date: str = ""
    credential_id: str = ""


class ProjectItem(Base):
    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    url: str = ""


class EvidenceItem(Base):
    """One skill and every place in the master resume that supports it."""

    skill: str
    canonical: str = ""
    evidence: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)   # e.g. "Company X → Project Y"
    confidence: Confidence = "MEDIUM"
    years: Optional[float] = None


class CandidateProfile(Base):
    contact: Contact = Field(default_factory=Contact)
    current_title: str = ""
    previous_titles: list[str] = Field(default_factory=list)
    total_years_experience: Optional[float] = None
    primary_domain: str = ""
    secondary_domains: list[str] = Field(default_factory=list)
    has_leadership_experience: bool = False
    leadership_summary: str = ""
    skills: dict[str, list[str]] = Field(default_factory=dict)   # category -> skills
    roles: list[Role] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)

    def all_skill_names(self) -> list[str]:
        out: list[str] = []
        for group in self.skills.values():
            out.extend(group)
        out.extend(e.skill for e in self.evidence)
        for role in self.roles:
            out.extend(role.technologies)
        for project in self.projects:
            out.extend(project.technologies)
        return out


# --------------------------------------------------------------------------- #
# JD analysis (stages 3–4)
# --------------------------------------------------------------------------- #
class Requirement(Base):
    id: str
    text: str
    canonical: str = ""
    category: str = "other"      # language | framework | cloud | database | ...
    priority: Priority = "P1"
    kind: Kind = "REQUIRED"
    years_required: Optional[float] = None
    rationale: str = ""


class JDAnalysis(Base):
    job_title: str = ""
    company: str = ""
    seniority: str = ""
    years_required: Optional[float] = None
    location: str = ""
    work_mode: str = ""              # remote | hybrid | onsite | unspecified
    work_authorization: str = ""
    domain: str = ""
    leadership_expected: bool = False
    requirements: list[Requirement] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    key_phrases: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Requirement ↔ evidence matrix (stage 5) and gaps (stage 6)
# --------------------------------------------------------------------------- #
class MatchRow(Base):
    requirement_id: str
    requirement: str
    canonical: str = ""
    priority: Priority = "P1"
    kind: Kind = "REQUIRED"
    match_type: MatchType = "NONE"
    score: float = 0.0                  # 0.0 – 1.0
    confidence: Confidence = "NONE"
    matched_via: str = ""               # which candidate skill produced the match
    evidence: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    notes: str = ""


class Gap(Base):
    requirement_id: str
    requirement: str
    priority: Priority
    kind: Kind
    risk: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    evidence_status: str = "No supporting evidence in master resume"
    recommendation: str = ""


# --------------------------------------------------------------------------- #
# Positioning (stage 7)
# --------------------------------------------------------------------------- #
class Positioning(Base):
    target_title: str = ""
    target_seniority: str = ""
    identity_statement: str = ""
    supported: bool = True
    support_reasoning: str = ""
    differentiators: list[str] = Field(default_factory=list)
    emphasise: list[str] = Field(default_factory=list)
    de_emphasise: list[str] = Field(default_factory=list)
    section_order: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Generated resume (stage 8) — structured, never a blob of markdown
# --------------------------------------------------------------------------- #
class ResumeBullet(Base):
    text: str
    source_ref: str = ""          # where in the master resume this came from
    keywords: list[str] = Field(default_factory=list)


class ResumeRole(Base):
    company: str
    title: str
    start_date: str
    end_date: str
    location: str = ""
    bullets: list[ResumeBullet] = Field(default_factory=list)


class ResumeSection(Base):
    heading: str
    kind: Literal[
        "summary", "skills", "experience", "projects", "education",
        "certifications", "achievements",
    ]
    paragraphs: list[str] = Field(default_factory=list)
    bullets: list[ResumeBullet] = Field(default_factory=list)
    skill_groups: dict[str, list[str]] = Field(default_factory=dict)
    roles: list[ResumeRole] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)


class TailoredResume(Base):
    contact: Contact = Field(default_factory=Contact)
    headline: str = ""
    sections: list[ResumeSection] = Field(default_factory=list)


class ChangeExplanation(Base):
    change: str
    reason: str
    source: str = ""
    category: Literal["added", "removed", "rewritten", "reordered", "repositioned"] = "rewritten"


# --------------------------------------------------------------------------- #
# Validation + scoring
# --------------------------------------------------------------------------- #
class ValidationCheck(Base):
    id: str
    label: str
    passed: bool
    severity: Literal["critical", "warning", "info"] = "warning"
    detail: str = ""
    offenders: list[str] = Field(default_factory=list)


class ValidationReport(Base):
    checks: list[ValidationCheck] = Field(default_factory=list)

    @property
    def critical_failures(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed and c.severity == "critical"]

    @property
    def passed(self) -> bool:
        return not self.critical_failures

    def score(self) -> float:
        """0–100 formatting/compliance score derived from the checks."""
        if not self.checks:
            return 0.0
        weights = {"critical": 3.0, "warning": 1.0, "info": 0.25}
        total = sum(weights[c.severity] for c in self.checks)
        earned = sum(weights[c.severity] for c in self.checks if c.passed)
        return round(100.0 * earned / total, 1) if total else 0.0


class ScoreComponent(Base):
    key: str
    label: str
    weight: float                 # 0.0 – 1.0
    raw: float                    # 0 – 100
    weighted: float               # raw * weight
    explanation: str = ""
    details: list[str] = Field(default_factory=list)


class ScoreReport(Base):
    total: float = 0.0
    band: str = ""
    components: list[ScoreComponent] = Field(default_factory=list)


class RecruiterView(Base):
    score: float = 0.0
    who_is_this: str = ""
    what_level: str = ""
    specialisation: str = ""
    technologies_visible: list[str] = Field(default_factory=list)
    relevance_to_role: str = ""
    top_strengths: list[str] = Field(default_factory=list)
    top_weaknesses: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Top-level API payloads
# --------------------------------------------------------------------------- #
class AnalysisResult(Base):
    analysis_id: str
    created_at: str
    target_market: str = "global"
    profile: CandidateProfile
    jd: JDAnalysis
    matrix: list[MatchRow] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    positioning: Positioning = Field(default_factory=Positioning)
    baseline_scores: ScoreReport = Field(default_factory=ScoreReport)
    warnings: list[str] = Field(default_factory=list)


class GenerationResult(Base):
    version_id: str
    analysis_id: str
    created_at: str
    version_name: str
    resume: TailoredResume
    plain_text: str
    scores: ScoreReport
    ats_report: ValidationReport
    truth_report: ValidationReport
    recruiter: RecruiterView
    changes: list[ChangeExplanation] = Field(default_factory=list)
    diff: dict[str, Any] = Field(default_factory=dict)
    status: Literal["optimized", "needs_review"] = "needs_review"
    status_reasons: list[str] = Field(default_factory=list)


class Application(Base):
    id: str = ""
    company: str = ""
    job_title: str = ""
    jd_excerpt: str = ""
    version_id: str = ""
    version_name: str = ""
    positioning: str = ""
    applied_on: str = ""
    ats_score: float = 0.0
    jd_match_score: float = 0.0
    url: str = ""
    status: str = "saved"
    recruiter: str = ""
    interview_stage: str = ""
    notes: str = ""
    result: str = ""
