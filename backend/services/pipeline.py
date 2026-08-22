"""The pipeline: a DAG of stages, with a convergence loop around generation.

    ┌─ parse_resume ─┐                        (LLM, parallel)
    │                ├─ evidence_index ─┐
    ├─ analyse_jd ───┘                  ├─ matrix ─ refine ─┬─ gaps ────┐
    └────────────────────────────────────┘                  └─ positioning
                                                                   │
                                    ┌──────────────────────────────┘
                                    ▼
        ╭─────────── repair loop (≤3 iterations) ───────────╮
        │  write ─▶ ATS checks + truth gate ─▶ pass? ─┐     │
        │     ▲                                       │     │
        │     └────── feedback: exact offenders ◀─────┘     │
        ╰───────────────────────────────────────────────────╯
                                    │
                     lift loop (until dry) ─▶ claim audit ─▶ recruiter ─▶ score

`parse_resume` and `analyse_jd` are independent, so the graph runs them
concurrently. `/generate` re-runs the graph with the analysis context already
populated, so those nodes report `cached` and cost nothing.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from ..graph import Context, Graph
from ..llm import Call, Provider, get_provider
from ..models.schemas import (
    AnalysisResult,
    CandidateProfile,
    Certification,
    ChangeExplanation,
    Education,
    GenerationResult,
    JDAnalysis,
    MatchRow,
    Positioning,
    RecruiterView,
    ResumeBullet,
    ResumeRole,
    ResumeSection,
    TailoredResume,
    ValidationReport,
)
from ..prompts import (
    FEEDBACK_BLOCK,
    JD_SYSTEM,
    JD_USER,
    POSITIONING_SYSTEM,
    POSITIONING_USER,
    PROFILE_SYSTEM,
    PROFILE_USER,
    RECRUITER_SYSTEM,
    RECRUITER_USER,
    REFINE_SYSTEM,
    REFINE_USER,
    TRUTH_SYSTEM,
    TRUTH_USER,
    WRITER_SYSTEM,
    WRITER_USER,
    schemas,
)
from . import ats_validator, diffing, matching, ontology, scoring, truth_validator
from .loops import RepairLoop, lift_loop, merge_reports
from .render import to_plain_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _groups_to_dict(groups: list[dict[str, Any]] | None) -> dict[str, list[str]]:
    """JSON Schema can't express arbitrary-key objects, so groups arrive as pairs."""
    out: dict[str, list[str]] = {}
    for g in groups or []:
        category = (g.get("category") or "Other").strip() or "Other"
        skills = [s for s in (g.get("skills") or []) if s and s.strip()]
        if skills:
            out.setdefault(category, [])
            for s in skills:
                if s not in out[category]:
                    out[category].append(s)
    return out


# --------------------------------------------------------------------------- #
# Stage implementations
# --------------------------------------------------------------------------- #
def _matrix_of(ctx: Context) -> list[MatchRow]:
    """The refined matrix when the optional refine stage succeeded, else the
    deterministic one. Every consumer goes through here so an optional-stage
    failure degrades quality instead of breaking the run."""
    return ctx.get("matrix_final") or ctx["matrix"]


def _stage_profile(ctx: Context) -> CandidateProfile:
    provider: Provider = ctx["provider"]
    resume_text: str = ctx["resume_text"]
    payload = provider.json(
        Call(
            stage="profile",
            system=PROFILE_SYSTEM,
            user=PROFILE_USER.format(resume_text=resume_text),
            schema=schemas.PROFILE_SCHEMA,
            max_tokens=16000,
        )
    )
    payload["skills"] = _groups_to_dict(payload.pop("skill_groups", None))
    profile = CandidateProfile.model_validate(payload)
    for ev in profile.evidence:
        ev.canonical = ontology.canonicalise(ev.skill)
    if not profile.roles:
        ctx.warn(
            "No work experience could be parsed from the resume. Check that the "
            "upload is a text-based document, not a scan."
        )
    return profile


def _stage_jd(ctx: Context) -> JDAnalysis:
    provider: Provider = ctx["provider"]
    payload = provider.json(
        Call(
            stage="jd",
            system=JD_SYSTEM,
            user=JD_USER.format(jd_text=ctx["jd_text"], market=ctx.get("market", "global")),
            schema=schemas.JD_SCHEMA,
            max_tokens=12000,
        )
    )
    jd = JDAnalysis.model_validate(payload)
    for req in jd.requirements:
        req.canonical = ontology.canonicalise(req.canonical or req.text)
    if not jd.requirements:
        ctx.warn("No requirements could be extracted from the job description.")
    return jd


def _stage_evidence_index(ctx: Context):
    return matching.build_evidence_index(ctx["profile"], ctx["resume_text"])


def _stage_matrix(ctx: Context) -> list[MatchRow]:
    return matching.build_matrix(ctx["jd"], ctx["evidence_index"])


def _stage_refine(ctx: Context) -> list[MatchRow]:
    """Ask the model to adjudicate only the rows the deterministic matcher was
    unsure about. Exact matches and clear misses are left alone — the matcher is
    more reliable there, and every row sent costs tokens."""
    matrix: list[MatchRow] = ctx["matrix"]
    ambiguous = [
        r for r in matrix
        if r.match_type in ("PARTIAL", "WEAK", "NONE") and r.priority in ("P0", "P1")
    ][:25]
    if not ambiguous:
        return matrix

    index = ctx["evidence_index"]
    evidence_blob = json.dumps(
        [
            {
                "skill": item.skill or canon,
                "evidence": item.evidence[:3],
                "sources": item.sources[:2],
                "confidence": item.confidence,
            }
            for canon, item in list(index.by_canonical.items())[:160]
        ],
        indent=1,
    )
    rows_blob = json.dumps(
        [
            {
                "requirement_id": r.requirement_id,
                "requirement": r.requirement,
                "priority": r.priority,
                "deterministic_guess": r.match_type,
                "deterministic_score": r.score,
            }
            for r in ambiguous
        ],
        indent=1,
    )

    payload = ctx["provider"].json(
        Call(
            stage="refine",
            system=REFINE_SYSTEM,
            user=REFINE_USER.format(evidence=evidence_blob, rows=rows_blob),
            schema=schemas.REFINE_SCHEMA,
            max_tokens=12000,
        )
    )

    by_id = {r.requirement_id: r for r in matrix}
    for update in payload.get("rows", []):
        row = by_id.get(update.get("requirement_id", ""))
        if row is None:
            continue
        score = float(update.get("score", row.score) or 0.0)
        score = max(0.0, min(1.0, score))
        # The LLM may raise or lower a score, but never invent an EXACT match the
        # deterministic matcher didn't find — that tier means a literal string hit.
        if update.get("match_type") == "EXACT" and row.match_type != "EXACT":
            update["match_type"] = "STRONG_SEMANTIC"
            score = min(score, 0.95)
        row.match_type = update.get("match_type", row.match_type)
        row.score = round(score, 3)
        row.confidence = update.get("confidence", row.confidence)
        row.matched_via = update.get("matched_via") or row.matched_via
        if update.get("evidence"):
            row.evidence = update["evidence"][:3]
        if update.get("notes"):
            row.notes = update["notes"]

    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    matrix.sort(key=lambda r: (order.get(r.priority, 9), -r.score))
    return matrix


def _stage_gaps(ctx: Context):
    return matching.build_gaps(_matrix_of(ctx))


def _stage_positioning(ctx: Context) -> Positioning:
    jd: JDAnalysis = ctx["jd"]
    profile: CandidateProfile = ctx["profile"]
    matrix: list[MatchRow] = _matrix_of(ctx)

    strengths = [r.requirement for r in matrix if r.score >= 0.85][:15]
    gap_names = [g.requirement for g in ctx["gaps"] if g.risk in ("HIGH", "MEDIUM")][:12]
    titles = ", ".join(
        t for t in [profile.current_title, *profile.previous_titles] if t
    ) or "not stated"

    jd_summary = json.dumps(
        {
            "job_title": jd.job_title,
            "seniority": jd.seniority,
            "years_required": jd.years_required,
            "domain": jd.domain,
            "leadership_expected": jd.leadership_expected,
            "responsibilities": jd.responsibilities[:8],
            "key_phrases": jd.key_phrases[:12],
        },
        indent=1,
    )

    payload = ctx["provider"].json(
        Call(
            stage="positioning",
            system=POSITIONING_SYSTEM,
            user=POSITIONING_USER.format(
                job_title=jd.job_title or "unspecified",
                seniority=jd.seniority or "unspecified",
                company=jd.company or "unspecified",
                domain=jd.domain or "unspecified",
                titles=titles,
                strengths=", ".join(strengths) or "none identified",
                gaps=", ".join(gap_names) or "none",
                jd_summary=jd_summary,
            ),
            schema=schemas.POSITIONING_SCHEMA,
            max_tokens=6000,
        )
    )
    positioning = Positioning.model_validate(payload)
    if not positioning.supported:
        ctx.warn(
            f"The JD targets '{jd.job_title}' but the evidence supports "
            f"'{positioning.target_title}'. {positioning.support_reasoning}"
        )
    return positioning


def _stage_baseline_scores(ctx: Context):
    return scoring.compute(
        scoring.ScoringInputs(
            jd=ctx["jd"],
            profile=ctx["profile"],
            matrix=_matrix_of(ctx),
            resume_text="",
            positioning=ctx.get("positioning"),
        ),
        weights=ctx.get("weights"),
    )


# --------------------------------------------------------------------------- #
# Writer + assembly
# --------------------------------------------------------------------------- #
def _assemble(payload: dict[str, Any], ctx: Context) -> tuple[TailoredResume, list[ChangeExplanation]]:
    profile: CandidateProfile = ctx["profile"]
    positioning: Positioning = ctx["positioning"]

    resume = TailoredResume(
        contact=profile.contact.model_copy(deep=True),
        headline=payload.get("headline") or positioning.target_title or profile.current_title,
    )

    sections: list[ResumeSection] = []

    summary = (payload.get("summary") or "").strip()
    if summary:
        sections.append(
            ResumeSection(
                heading="Professional Summary",
                kind="summary",
                paragraphs=[s.strip() for s in summary.split("\n\n") if s.strip()],
            )
        )

    skill_groups = _groups_to_dict(payload.get("skill_groups"))
    if skill_groups:
        ordered = dict(
            sorted(
                skill_groups.items(),
                key=lambda kv: ontology.CATEGORY_ORDER.index(kv[0])
                if kv[0] in ontology.CATEGORY_ORDER
                else 99,
            )
        )
        sections.append(
            ResumeSection(heading="Core Skills", kind="skills", skill_groups=ordered)
        )

    roles: list[ResumeRole] = []
    for r in payload.get("roles") or []:
        roles.append(
            ResumeRole(
                company=r.get("company", ""),
                title=r.get("title", ""),
                start_date=r.get("start_date", ""),
                end_date=r.get("end_date", ""),
                location=r.get("location", ""),
                bullets=[
                    ResumeBullet(
                        text=(b.get("text") or "").strip(),
                        source_ref=b.get("source_ref", ""),
                        keywords=b.get("keywords") or [],
                    )
                    for b in (r.get("bullets") or [])
                    if (b.get("text") or "").strip()
                ],
            )
        )
    if roles:
        sections.append(
            ResumeSection(heading="Professional Experience", kind="experience", roles=roles)
        )

    projects = payload.get("selected_projects") or []
    project_bullets: list[ResumeBullet] = []
    for proj in projects:
        name = (proj.get("name") or "").strip()
        desc = (proj.get("description") or "").strip()
        if name or desc:
            project_bullets.append(
                ResumeBullet(text=f"{name} — {desc}".strip(" —"), source_ref=name)
            )
        for b in proj.get("bullets") or []:
            text = (b.get("text") or "").strip()
            if text:
                project_bullets.append(ResumeBullet(text=text, source_ref=name))
    if project_bullets:
        sections.append(
            ResumeSection(
                heading="Selected Projects", kind="projects", bullets=project_bullets
            )
        )

    # Education and certifications are copied verbatim from the master profile —
    # the writer is never given the opportunity to reword them.
    if profile.education:
        sections.append(
            ResumeSection(
                heading="Education",
                kind="education",
                education=[Education.model_validate(e.model_dump()) for e in profile.education],
            )
        )
    if profile.certifications:
        sections.append(
            ResumeSection(
                heading="Certifications",
                kind="certifications",
                certifications=[
                    Certification.model_validate(c.model_dump())
                    for c in profile.certifications
                ],
            )
        )

    resume.sections = sections
    changes = [
        ChangeExplanation.model_validate(c) for c in (payload.get("changes") or [])
    ]
    return resume, changes


def _write_resume(ctx: Context, feedback: list[str], iteration: int) -> TailoredResume:
    jd: JDAnalysis = ctx["jd"]
    profile: CandidateProfile = ctx["profile"]
    matrix: list[MatchRow] = _matrix_of(ctx)
    positioning: Positioning = ctx["positioning"]

    requirements_blob = "\n".join(
        f"- [{r.priority}] {r.requirement} — match={r.match_type} "
        f"({r.score:.2f}) via {r.matched_via or 'n/a'}"
        for r in matrix
        if r.score >= 0.6
    ) or "- (none matched)"

    gaps_blob = "\n".join(
        f"- [{g.priority}/{g.risk} risk] {g.requirement} — {g.evidence_status}"
        for g in ctx["gaps"]
    ) or "- (no gaps)"

    profile_blob = json.dumps(
        profile.model_dump(exclude={"evidence"}), indent=1, ensure_ascii=False
    )

    jd_summary = json.dumps(
        {
            "job_title": jd.job_title,
            "company": jd.company,
            "seniority": jd.seniority,
            "domain": jd.domain,
            "responsibilities": jd.responsibilities[:10],
            "key_phrases": jd.key_phrases[:15],
        },
        indent=1,
    )

    feedback_block = (
        FEEDBACK_BLOCK.format(feedback="\n".join(f"- {f}" for f in feedback))
        if feedback
        else ""
    )

    payload = ctx["provider"].json(
        Call(
            stage="writer",
            system=WRITER_SYSTEM,
            user=WRITER_USER.format(
                positioning=json.dumps(positioning.model_dump(), indent=1),
                jd_summary=jd_summary,
                requirements=requirements_blob,
                gaps=gaps_blob,
                profile=profile_blob,
                feedback_block=feedback_block,
            ),
            schema=schemas.WRITER_SCHEMA,
            max_tokens=16000,
        )
    )
    resume, changes = _assemble(payload, ctx)
    ctx.set("last_changes", changes)
    return resume


def _validate_candidate(ctx: Context, resume: TailoredResume) -> ValidationReport:
    ats = ats_validator.validate(resume, ctx["jd"], _matrix_of(ctx))
    truth = truth_validator.validate(resume, ctx["profile"], ctx["resume_text"])
    ctx.set("last_ats", ats)
    ctx.set("last_truth", truth)
    return merge_reports(ats, truth)


def _score_candidate(ctx: Context, resume: TailoredResume) -> float:
    return scoring.compute(
        scoring.ScoringInputs(
            jd=ctx["jd"],
            profile=ctx["profile"],
            matrix=_matrix_of(ctx),
            resume_text=to_plain_text(resume),
            ats_report=ctx.get("last_ats"),
            positioning=ctx.get("positioning"),
        ),
        weights=ctx.get("weights"),
    ).total


def _stage_write_loop(ctx: Context) -> TailoredResume:
    """Repair loop, then an optional loop-until-dry keyword lift."""
    loop = RepairLoop[TailoredResume](
        produce=lambda feedback, i: _write_resume(ctx, feedback, i),
        validate=lambda r: _validate_candidate(ctx, r),
        score=lambda r: _score_candidate(ctx, r),
        max_iterations=int(ctx.get("max_repair_iterations", 3)),
        min_gain=0.75,
    )
    result = loop.run()
    ctx.set("loop_report", result.to_dict())
    if not result.converged:
        ctx.warn(
            f"Generation stopped without clearing every critical check "
            f"({result.stop_reason}). The document is returned for review."
        )

    resume = result.value

    lift_rounds = int(ctx.get("lift_rounds", 1))
    if result.converged and lift_rounds > 0:
        def find_missing(candidate: TailoredResume) -> list[str]:
            report = ats_validator.validate(candidate, ctx["jd"], _matrix_of(ctx))
            for check in report.checks:
                if check.id == "supported_keywords_surfaced" and not check.passed:
                    return check.offenders
            return []

        holder = {"current": resume}

        def produce(instruction: list[str], _round: int) -> TailoredResume:
            if not instruction:
                return holder["current"]
            candidate = _write_resume(ctx, instruction, 90)
            report = _validate_candidate(ctx, candidate)
            # Never trade a truth/ATS regression for keyword coverage.
            if report.passed:
                holder["current"] = candidate
            return holder["current"]

        lifted, rounds = lift_loop(produce, find_missing, max_rounds=lift_rounds)
        ctx.set("lift_report", rounds)
        resume = lifted
        _validate_candidate(ctx, resume)   # refresh cached reports for the final one

    return resume


def _stage_truth_audit(ctx: Context) -> ValidationReport:
    resume: TailoredResume = ctx["resume"]
    payload = ctx["provider"].json(
        Call(
            stage="truth",
            system=TRUTH_SYSTEM,
            user=TRUTH_USER.format(
                generated=to_plain_text(resume), master=ctx["resume_text"]
            ),
            schema=schemas.TRUTH_SCHEMA,
            max_tokens=10000,
        )
    )
    return truth_validator.report_from_llm_audit(payload)


def _stage_recruiter(ctx: Context) -> RecruiterView:
    jd: JDAnalysis = ctx["jd"]
    payload = ctx["provider"].json(
        Call(
            stage="recruiter",
            system=RECRUITER_SYSTEM,
            user=RECRUITER_USER.format(
                job_title=jd.job_title or "the role",
                company=jd.company or "the company",
                resume=to_plain_text(ctx["resume"]),
            ),
            schema=schemas.RECRUITER_SCHEMA,
            max_tokens=5000,
        )
    )
    view = RecruiterView.model_validate(payload)
    view.score = max(0.0, min(100.0, float(view.score or 0.0)))
    return view


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph() -> Graph:
    g = Graph("resumefit", max_workers=4)
    g.add("profile", _stage_profile, note="LLM: parse resume → profile + evidence DB")
    g.add("jd", _stage_analyse_jd_alias, note="LLM: JD → classified requirements")
    g.add("evidence_index", _stage_evidence_index, deps=["profile"],
          note="Deterministic: build evidence index")
    g.add("matrix", _stage_matrix, deps=["evidence_index", "jd"],
          note="Deterministic: requirement ↔ evidence matching")
    g.add("matrix_final", _stage_refine, deps=["matrix"], optional=True,
          note="LLM: adjudicate ambiguous rows")
    g.add("gaps", _stage_gaps, deps=["matrix_final"], note="Deterministic: gap analysis")
    g.add("positioning", _stage_positioning, deps=["gaps"], note="LLM: target positioning")
    g.add("baseline_scores", _stage_baseline_scores, deps=["positioning"],
          note="Deterministic: baseline scoring")
    g.add("resume", _stage_write_loop, deps=["positioning"],
          note="LLM + loop: write → validate → repair")
    g.add("truth_audit", _stage_truth_audit, deps=["resume"], optional=True,
          note="LLM: independent claim audit")
    g.add("recruiter", _stage_recruiter, deps=["resume"], optional=True,
          note="LLM: 10-second recruiter scan")
    return g


def _stage_analyse_jd_alias(ctx: Context) -> JDAnalysis:
    # Separate function so the node name ("jd") and the produced key match.
    return _stage_jd(ctx)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def analyse(
    resume_text: str,
    jd_text: str,
    *,
    market: str = "global",
    weights: dict[str, float] | None = None,
    provider: Provider | None = None,
) -> tuple[AnalysisResult, Context]:
    ctx = Context()
    ctx.set("resume_text", resume_text)
    ctx.set("jd_text", jd_text)
    ctx.set("market", market)
    ctx.set("weights", weights or {})
    ctx.set("provider", provider or get_provider())

    graph = build_graph()
    graph.run(ctx, only={"baseline_scores"})

    result = AnalysisResult(
        analysis_id=_sid("an"),
        created_at=_now(),
        target_market=market,
        profile=ctx["profile"],
        jd=ctx["jd"],
        matrix=_matrix_of(ctx),
        gaps=ctx["gaps"],
        positioning=ctx["positioning"],
        baseline_scores=ctx["baseline_scores"],
        warnings=ctx.warnings,
    )
    ctx.set("analysis_id", result.analysis_id)
    return result, ctx


def generate(
    ctx: Context,
    *,
    max_repair_iterations: int = 3,
    lift_rounds: int = 1,
) -> GenerationResult:
    ctx.set("max_repair_iterations", max_repair_iterations)
    ctx.set("lift_rounds", lift_rounds)
    ctx.trace.clear()

    graph = build_graph()
    graph.run(ctx, only={"truth_audit", "recruiter"})

    resume: TailoredResume = ctx["resume"]
    plain = to_plain_text(resume)

    ats_report: ValidationReport = ctx.get("last_ats") or ValidationReport()
    truth_deterministic: ValidationReport = ctx.get("last_truth") or ValidationReport()
    truth_llm: ValidationReport = ctx.get("truth_audit") or ValidationReport()
    truth_report = merge_reports(truth_deterministic, truth_llm)
    recruiter: RecruiterView = ctx.get("recruiter") or RecruiterView(
        score=75.0, top_weaknesses=["Recruiter simulation did not run."]
    )

    scores = scoring.compute(
        scoring.ScoringInputs(
            jd=ctx["jd"],
            profile=ctx["profile"],
            matrix=_matrix_of(ctx),
            resume_text=plain,
            ats_report=ats_report,
            recruiter=recruiter,
            positioning=ctx["positioning"],
        ),
        weights=ctx.get("weights"),
    )

    status_reasons: list[str] = []
    for check in ats_report.critical_failures + truth_report.critical_failures:
        status_reasons.append(f"{check.label}: {check.detail}")
    status = "optimized" if not status_reasons else "needs_review"

    jd = ctx["jd"]
    company = (jd.company or "Company").replace(" ", "")
    role = (ctx["positioning"].target_title or jd.job_title or "Role").replace(" ", "_")
    version_name = f"Resume_{role}_{company}"

    return GenerationResult(
        version_id=_sid("v"),
        analysis_id=ctx.get("analysis_id", ""),
        created_at=_now(),
        version_name=version_name,
        resume=resume,
        plain_text=plain,
        scores=scores,
        ats_report=ats_report,
        truth_report=truth_report,
        recruiter=recruiter,
        changes=ctx.get("last_changes") or [],
        diff={
            **diffing.compare(ctx["profile"], resume, _matrix_of(ctx)),
            "loop": ctx.get("loop_report", {}),
            "lift": ctx.get("lift_report", []),
        },
        status=status,  # type: ignore[arg-type]
        status_reasons=status_reasons,
    )
