"""The truthfulness engine (pipeline stage 10) — the hard gate.

Two passes, in this order:

1. **Deterministic.** Every number, every named technology, every company, title
   and date in the generated resume must be findable in the master resume. This
   pass is code, it cannot be talked out of a failure, and it produces the exact
   offending strings that the repair loop feeds back to the writer.

2. **LLM claim audit.** Catches the class of fabrication the deterministic pass
   cannot see — a claim that uses only words present in the source but asserts
   something the source does not ("led a team of engineers" where the source says
   "collaborated with the team").

A critical failure in either pass blocks the "optimized" status. The document is
still returned, marked "needs_review", with the offending claims listed — hiding
the failure would defeat the point.
"""

from __future__ import annotations

import re

from ..models.schemas import (
    CandidateProfile,
    TailoredResume,
    ValidationCheck,
    ValidationReport,
)
from . import dates, ontology
from .render import all_bullet_texts, claim_texts, to_plain_text

# Numbers that carry a factual claim. Deliberately excludes bare small integers
# ("3 services") which are usually structural, and version-like tokens.
_METRIC = re.compile(
    r"""
    (?P<pct>\d{1,3}(?:\.\d+)?\s?%)                      # 40%, 99.9%
  | (?P<money>[$£€₹]\s?\d[\d,.]*\s?(?:[kmb]|million|billion|crore|lakh)?)
  | (?P<scale>\b\d[\d,]*\s?(?:k|m|b|million|billion|thousand)\b)
  | (?P<big>\b\d{3,}(?:[,.]\d+)?\b)                     # 500, 1,200
  | (?P<xfactor>\b\d+(?:\.\d+)?x\b)                     # 3x, 2.5x
  | (?P<time>\b\d+(?:\.\d+)?\s?(?:ms|seconds?|minutes?|hours?|days?)\b)
  | (?P<rate>\b\d+(?:\.\d+)?\s?(?:rps|qps|tps|req/s|requests?/s)\b)
  | (?P<years>\b\d+(?:\.\d+)?\+?\s?(?:years?|yrs?)\b)
    """,
    re.X | re.I,
)

_TEAM = re.compile(r"\bteam of (\d+)", re.I)


def _numbers_in(text: str) -> list[str]:
    out: list[str] = []
    for match in _METRIC.finditer(text or ""):
        token = match.group(0).strip()
        if token and token not in out:
            out.append(token)
    return out


def _digits(token: str) -> str:
    """Reduce a metric to comparable digits: '1,200' and '1200' both -> '1200'."""
    return re.sub(r"[^\d.]", "", token).rstrip(".")


def _mentions_word(haystack: str, term: str) -> bool:
    """Word-boundary containment on normalised text."""
    return re.search(
        rf"(?<![a-z0-9]){re.escape(ontology.normalise(term))}(?![a-z0-9])", haystack
    ) is not None


def _master_number_set(master_text: str) -> set[str]:
    digits = {_digits(t) for t in _numbers_in(master_text)}
    digits |= {d for d in re.findall(r"\d[\d,.]*", master_text)}
    digits |= {_digits(d) for d in re.findall(r"\d[\d,.]*", master_text)}
    return {d for d in digits if d}


def validate(
    resume: TailoredResume,
    profile: CandidateProfile,
    master_text: str,
) -> ValidationReport:
    generated = to_plain_text(resume)
    master_norm = ontology.normalise(master_text)
    checks: list[ValidationCheck] = []

    # ---- 1. Metrics ------------------------------------------------------
    master_numbers = _master_number_set(master_text)
    # A total-years figure computed from employment dates that are on the resume
    # is *derived* from the source, not invented — even when that exact integer
    # never appears as a literal. Admit it, and only it.
    if profile.total_years_experience:
        years = profile.total_years_experience
        master_numbers |= {
            str(int(years)), str(round(years)), f"{years:g}", str(int(years) + 1)
        }
    invented: list[str] = []
    for bullet in all_bullet_texts(resume):
        for token in _numbers_in(bullet):
            if _digits(token) and _digits(token) not in master_numbers:
                invented.append(f"'{token}' in: {bullet[:90]}")
    checks.append(
        ValidationCheck(
            id="no_invented_metrics",
            label="Every metric traces to the master resume",
            passed=not invented,
            severity="critical",
            detail=(
                "These numbers do not appear anywhere in the source. Remove them or "
                "rewrite the bullet without a metric."
                if invented
                else "All numeric claims are present in the source document."
            ),
            offenders=invented[:12],
        )
    )

    # ---- 2. Team sizes ---------------------------------------------------
    master_teams = set(_TEAM.findall(master_text))
    bad_teams = [
        f"team of {n}"
        for bullet in all_bullet_texts(resume)
        for n in _TEAM.findall(bullet)
        if n not in master_teams
    ]
    checks.append(
        ValidationCheck(
            id="no_invented_team_size",
            label="Team sizes match the source",
            passed=not bad_teams,
            severity="critical",
            detail="Team size stated that the source does not support."
            if bad_teams
            else "No unsupported team sizes.",
            offenders=bad_teams,
        )
    )

    # ---- 3. Technologies -------------------------------------------------
    # Compared as SURFACE STRINGS, not canonical ids.
    #
    # The previous version canonicalised both sides before comparing, which made
    # the check tautological for the exact error it exists to catch: the master
    # says "Playwright", the document says "Cypress", both canonicalise to
    # `cypress`, and the substitution was invisible. That shipped a resume
    # claiming three products the candidate had never used, with the gate green.
    generated_forms = ontology.extract_surface_forms(" \n".join(claim_texts(resume)))
    master_forms = ontology.extract_surface_forms(master_text)
    unsupported_tech: list[str] = []
    for canon, written in sorted(generated_forms.items()):
        if _mentions_word(master_norm, written):
            continue
        in_master = master_forms.get(canon)
        if in_master:
            unsupported_tech.append(
                f"'{written}' — your resume says '{in_master}', not this"
            )
        else:
            unsupported_tech.append(f"'{written}' (absent from your resume)")
    checks.append(
        ValidationCheck(
            id="no_invented_technologies",
            label="Every technology named appears in the master resume",
            passed=not unsupported_tech,
            severity="critical",
            detail=(
                "These product names do not appear in your resume. Where a different "
                "spelling is shown, an alias was rendered in place of your own wording "
                "— claiming a product you have not used is worse than a missed keyword."
                if unsupported_tech
                else "Every technology named appears in your resume, in your wording."
            ),
            offenders=unsupported_tech[:12],
        )
    )

    # ---- 4. Employment facts --------------------------------------------
    master_companies = {ontology.normalise(r.company) for r in profile.roles if r.company}
    master_titles = {ontology.normalise(r.title) for r in profile.roles if r.title}
    bad_companies: list[str] = []
    bad_titles: list[str] = []
    bad_dates: list[str] = []

    # Keyed by (company, title), not company alone. A promotion produces two
    # roles at one employer, and a company-only dict keeps whichever came last —
    # so the generated "Technical Analyst, Jan 2019 - Aug 2019" was measured
    # against its own predecessor's dates and reported as a fabrication. Titles
    # differ across a promotion, so this lookup stays exact rather than lenient.
    by_company_title = {
        (ontology.normalise(r.company), ontology.normalise(r.title)): r
        for r in profile.roles
    }
    roles_at: dict[str, list] = {}
    for r in profile.roles:
        roles_at.setdefault(ontology.normalise(r.company), []).append(r)
    for section in resume.sections:
        for role in section.roles:
            company_norm = ontology.normalise(role.company)
            if company_norm and company_norm not in master_companies:
                bad_companies.append(role.company)
            title_norm = ontology.normalise(role.title)
            if title_norm and title_norm not in master_titles:
                bad_titles.append(f"{role.title} @ {role.company}")
            source_role = by_company_title.get((company_norm, title_norm))
            if source_role is None:
                siblings = roles_at.get(company_norm, [])
                # One role at this employer: the title was reworded, which
                # `titles_match` reports on its own, but the dates are still
                # comparable. Several roles: the title identifies which, and an
                # unmatched title is already a critical failure — guessing a
                # counterpart here would only invent a second, misleading one.
                source_role = siblings[0] if len(siblings) == 1 else None
            if source_role:
                # Compared semantically, not as strings: rendering "03/2021" as
                # "Mar 2021" is a formatting change the ATS validator actively
                # wants. Changing 2021 to 2019 is a fabrication. Only the second
                # should fail here.
                if role.start_date and not dates.same(role.start_date, source_role.start_date):
                    bad_dates.append(
                        f"{role.company} start: '{role.start_date}' vs source "
                        f"'{source_role.start_date}'"
                    )
                if role.end_date and not dates.same(role.end_date, source_role.end_date):
                    bad_dates.append(
                        f"{role.company} end: '{role.end_date}' vs source "
                        f"'{source_role.end_date}'"
                    )

    checks.append(
        ValidationCheck(
            id="companies_match",
            label="Company names match the master resume exactly",
            passed=not bad_companies,
            severity="critical",
            detail="Employer names must never be altered." if bad_companies else "All match.",
            offenders=bad_companies,
        )
    )
    checks.append(
        ValidationCheck(
            id="titles_match",
            label="Job titles match the master resume exactly",
            passed=not bad_titles,
            severity="critical",
            detail="Job titles must never be re-levelled or reworded."
            if bad_titles
            else "All match.",
            offenders=bad_titles,
        )
    )
    checks.append(
        ValidationCheck(
            id="dates_match",
            label="Employment dates match the master resume exactly",
            passed=not bad_dates,
            severity="critical",
            detail="Employment dates must never be adjusted." if bad_dates else "All match.",
            offenders=bad_dates,
        )
    )

    # ---- 5. Certifications and education --------------------------------
    master_certs = {ontology.normalise(c.name) for c in profile.certifications if c.name}
    bad_certs = [
        cert.name
        for section in resume.sections
        for cert in section.certifications
        if cert.name and ontology.normalise(cert.name) not in master_certs
    ]
    checks.append(
        ValidationCheck(
            id="certifications_match",
            label="No certifications beyond those in the master resume",
            passed=not bad_certs,
            severity="critical",
            detail="Certifications are trivially verifiable — never add one."
            if bad_certs
            else "No unsupported certifications.",
            offenders=bad_certs,
        )
    )

    master_institutions = {
        ontology.normalise(e.institution) for e in profile.education if e.institution
    }
    bad_edu = [
        edu.institution
        for section in resume.sections
        for edu in section.education
        if edu.institution and ontology.normalise(edu.institution) not in master_institutions
    ]
    checks.append(
        ValidationCheck(
            id="education_match",
            label="Education entries match the master resume",
            passed=not bad_edu,
            severity="critical",
            detail="Education must never be added or altered." if bad_edu else "All match.",
            offenders=bad_edu,
        )
    )

    # ---- 6. Years-of-experience claims ----------------------------------
    claimed_years = [
        float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\+?\s?(?:years?|yrs?)", generated, re.I)
    ]
    have = profile.total_years_experience
    overclaimed = (
        [f"{max(claimed_years):g}+ years claimed; profile supports {have:g}"]
        if claimed_years and have is not None and max(claimed_years) > have + 0.5
        else []
    )
    checks.append(
        ValidationCheck(
            id="years_claim",
            label="Years of experience claimed is supported",
            passed=not overclaimed,
            severity="critical",
            detail="A years claim above what the employment dates support."
            if overclaimed
            else "Years claim is consistent with the employment history.",
            offenders=overclaimed,
        )
    )

    # ---- 7. Sensitive attributes ----------------------------------------
    sensitive_terms = [
        "security clearance", "top secret", "ts/sci", "green card", "us citizen",
        "h1b", "h-1b", "work permit", "permanent resident", "ead",
    ]
    # Word-boundary matched: a substring test fires "ead" inside "lead" and
    # "h1b" inside a build id, which would fail an honest resume.
    gen_norm = ontology.normalise(generated)
    sensitive_added = [
        term for term in sensitive_terms
        if _mentions_word(gen_norm, term) and not _mentions_word(master_norm, term)
    ]
    checks.append(
        ValidationCheck(
            id="no_invented_status",
            label="No work-authorisation or clearance claims added",
            passed=not sensitive_added,
            severity="critical",
            detail="Visa, citizenship and clearance status must never be introduced."
            if sensitive_added
            else "No status claims introduced.",
            offenders=sensitive_added,
        )
    )

    return ValidationReport(checks=checks)


def report_from_llm_audit(payload: dict) -> ValidationReport:
    """Turn the LLM claim audit into checks that merge with the deterministic ones."""
    checks: list[ValidationCheck] = []
    claims = payload.get("claims") or []
    unsupported = [c for c in claims if not c.get("supported", True)]

    critical = [c for c in unsupported if c.get("severity") == "critical"]
    warnings = [c for c in unsupported if c.get("severity") == "warning"]

    checks.append(
        ValidationCheck(
            id="llm_claim_audit_critical",
            label="Claim audit: no fabricated facts",
            passed=not critical,
            severity="critical",
            detail=payload.get("notes", "") or "Independent review of every factual claim.",
            offenders=[
                f"{c.get('claim', '')[:100]} — {c.get('explanation', '')[:120]}"
                for c in critical
            ][:10],
        )
    )
    checks.append(
        ValidationCheck(
            id="llm_claim_audit_stretch",
            label="Claim audit: no overstated characterisations",
            passed=not warnings,
            severity="warning",
            detail="Claims that go slightly beyond what the source states."
            if warnings
            else "No overstatements found.",
            offenders=[
                f"{c.get('claim', '')[:100]} — {c.get('explanation', '')[:120]}"
                for c in warnings
            ][:10],
        )
    )
    return ValidationReport(checks=checks)
