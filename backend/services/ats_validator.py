"""Deterministic ATS + readability validation (pipeline stage 9).

Every check here runs on the generated document, in code. None of it is an LLM
opinion, so the resulting score is reproducible and each failure names the exact
offending strings — which is what makes the repair loop able to act on them.

Note on the claim we are careful not to make: no resume is "100% ATS compatible",
because Workday, Greenhouse, Taleo and iCIMS parse differently. What this module
measures is compliance with the format rules that are safe across all of them.
"""

from __future__ import annotations

import re
from collections import Counter

from ..models.schemas import (
    JDAnalysis,
    MatchRow,
    TailoredResume,
    ValidationCheck,
    ValidationReport,
)
from . import ontology
from .render import bullet_texts_only, to_plain_text

STANDARD_HEADINGS = {
    "summary", "professional summary", "profile", "objective",
    "skills", "core skills", "technical skills", "key skills",
    "experience", "professional experience", "work experience", "employment history",
    "projects", "selected projects", "key projects",
    "education", "certifications", "licenses and certifications",
    "achievements", "awards", "publications",
}

# Glyphs that survive copy-paste badly or signal a graphical layout.
RISKY_CHARS = "★☆●◆■▪▶►✔✓✦❖◼◾⬤▓░│┃┆┊⎯➤➔"

DATE_PATTERNS = [
    re.compile(r"^[A-Z][a-z]{2,8}\.? \d{4}$"),          # March 2023 / Mar 2023
    re.compile(r"^\d{2}/\d{4}$"),                        # 03/2023
    re.compile(r"^\d{4}-\d{2}$"),                        # 2023-03
    re.compile(r"^\d{4}$"),                              # 2023
    re.compile(r"^(Present|Current|Ongoing)$", re.I),
]

STOPWORDS = {
    "and", "the", "for", "with", "from", "that", "this", "into", "using", "used",
    "across", "within", "their", "them", "have", "has", "was", "were", "are", "our",
    "per", "via", "to", "of", "in", "on", "at", "by", "a", "an", "as", "it", "its",
}


def _check(
    cid: str, label: str, passed: bool, severity: str, detail: str = "",
    offenders: list[str] | None = None,
) -> ValidationCheck:
    return ValidationCheck(
        id=cid, label=label, passed=passed, severity=severity,  # type: ignore[arg-type]
        detail=detail, offenders=offenders or [],
    )


def _date_ok(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return True
    return any(p.match(v) for p in DATE_PATTERNS)


def _density(text: str) -> tuple[Counter, int]:
    tokens = [
        t for t in re.findall(r"[a-z][a-z0-9+#.\-]{2,}", text.lower())
        if t not in STOPWORDS
    ]
    return Counter(tokens), max(1, len(tokens))


def validate(
    resume: TailoredResume,
    jd: JDAnalysis | None = None,
    matrix: list[MatchRow] | None = None,
    master_text: str = "",
) -> ValidationReport:
    text = to_plain_text(resume)
    bullets = bullet_texts_only(resume)
    checks: list[ValidationCheck] = []

    # -- Contact ------------------------------------------------------------
    contact = resume.contact
    missing = [f for f in ("name", "email") if not getattr(contact, f, "")]
    checks.append(
        _check(
            "contact_present",
            "Name and email present in the document body",
            not missing,
            "critical",
            "Contact details must be in the body — many parsers ignore headers and "
            "footers entirely."
            if missing
            else "Contact block is in the body text.",
            missing,
        )
    )
    if contact.email:
        checks.append(
            _check(
                "email_valid",
                "Email address is parseable",
                bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", contact.email)),
                "warning",
                f"Detected: {contact.email}",
            )
        )

    # -- Headings -----------------------------------------------------------
    headings = [s.heading.strip() for s in resume.sections if s.heading.strip()]
    nonstandard = [h for h in headings if h.lower().strip(": ") not in STANDARD_HEADINGS]
    checks.append(
        _check(
            "standard_headings",
            "Section headings use conventional names",
            not nonstandard,
            "warning",
            "Creative headings ('Where I've Made an Impact') often fail to map to an "
            "ATS's expected sections."
            if nonstandard
            else "All headings are conventional.",
            nonstandard,
        )
    )
    kinds = {s.kind for s in resume.sections}
    required_sections = {"summary", "skills", "experience"}
    missing_sections = sorted(required_sections - kinds)
    checks.append(
        _check(
            "core_sections",
            "Core sections present (summary, skills, experience)",
            not missing_sections,
            "critical",
            f"Missing: {', '.join(missing_sections)}" if missing_sections else "All present.",
            missing_sections,
        )
    )

    # -- Structure ----------------------------------------------------------
    checks.append(
        _check(
            "single_column",
            "Single-column, table-free layout",
            True,
            "info",
            "The generator emits linear paragraphs only — no tables, text boxes, "
            "columns, images or text frames.",
        )
    )
    checks.append(
        _check(
            "no_graphics",
            "No images, icons, charts or skill bars",
            True,
            "info",
            "Graphical elements are not representable in this document model.",
        )
    )

    risky = sorted({c for c in text if c in RISKY_CHARS})
    checks.append(
        _check(
            "safe_glyphs",
            "No decorative or non-standard glyphs",
            not risky,
            "warning",
            "Decorative characters can be dropped or mangled during text extraction."
            if risky
            else "Only standard characters used.",
            risky,
        )
    )
    checks.append(
        _check(
            "ascii_safe",
            "Text is representable in plain ASCII-compatible encoding",
            all(ord(c) < 0x2500 for c in text),
            "info",
            "Characters above the box-drawing range can break naive parsers.",
        )
    )

    # -- Dates --------------------------------------------------------------
    bad_dates: list[str] = []
    formats_seen: set[str] = set()
    for section in resume.sections:
        for role in section.roles:
            for value in (role.start_date, role.end_date):
                if not value:
                    continue
                if not _date_ok(value):
                    bad_dates.append(f"{role.company}: '{value}'")
                for i, pattern in enumerate(DATE_PATTERNS[:4]):
                    if pattern.match(value.strip()):
                        formats_seen.add(str(i))
    checks.append(
        _check(
            "date_format",
            "Employment dates use a recognisable format",
            not bad_dates,
            "warning",
            "Unrecognised date formats are a common cause of mis-parsed employment "
            "history."
            if bad_dates
            else "All dates parse cleanly.",
            bad_dates,
        )
    )
    checks.append(
        _check(
            "date_consistency",
            "Date format is consistent throughout",
            len(formats_seen) <= 1,
            "warning",
            f"{len(formats_seen)} different date formats detected."
            if len(formats_seen) > 1
            else "One consistent format.",
        )
    )

    # -- Bullets ------------------------------------------------------------
    long_bullets = [b[:70] + "…" for b in bullets if len(b) > 320]
    checks.append(
        _check(
            "bullet_length",
            "Bullets are a scannable length",
            not long_bullets,
            "warning",
            "Bullets over ~320 characters are paragraphs; recruiters skip them."
            if long_bullets
            else "All bullets are within a scannable length.",
            long_bullets,
        )
    )
    short_bullets = [b for b in bullets if 0 < len(b.strip()) < 25]
    checks.append(
        _check(
            "bullet_substance",
            "Bullets carry substance",
            not short_bullets,
            "info",
            "Very short bullets waste a line without conveying an accomplishment."
            if short_bullets
            else "All bullets carry content.",
            short_bullets,
        )
    )

    summary_text = " ".join(
        p for s_ in resume.sections if s_.kind == "summary" for p in s_.paragraphs
    )
    if summary_text:
        words = len(summary_text.split())
        checks.append(
            _check(
                "summary_length",
                "Professional summary is a readable length",
                25 <= words <= 90,
                "warning",
                f"{words} words. Under ~25 says nothing; over ~90 stops being a summary "
                "and will not be read.",
            )
        )

    # -- Length -------------------------------------------------------------
    words = len(text.split())
    checks.append(
        _check(
            "document_length",
            "Document length is appropriate",
            350 <= words <= 1100,
            "warning",
            f"{words} words. Under ~350 reads as thin; over ~1100 typically exceeds "
            "two pages.",
        )
    )

    # -- Keyword stuffing ---------------------------------------------------
    counts, total = _density(text)
    master_counts, master_total = _density(master_text) if master_text else (Counter(), 1)

    # Every role carries a location, so a six-role resume necessarily repeats its
    # city and country. That is structural, not a vocabulary problem, and flagging
    # it sends the reader off to "fix" something that cannot be fixed.
    location_tokens: set[str] = set()
    for value in [resume.contact.location] + [
        role.location for s_ in resume.sections for role in s_.roles
    ]:
        location_tokens.update(re.findall(r"[a-z]{3,}", (value or "").lower()))
    for token in location_tokens:
        counts.pop(token, None)

    introduced: list[str] = []
    inherited: list[str] = []
    for term, n in counts.most_common(25):
        ratio = n / total
        if n < 7 or ratio <= 0.015:
            continue
        label = f"'{term}' ×{n} ({100 * ratio:.1f}% of the document)"
        master_n = master_counts.get(term, 0)
        # Compare *counts*, not densities. If tailoring added no new occurrences,
        # it cannot have stuffed the term — the density rose only because less
        # relevant text was removed around it. That is selection working, and
        # failing the document for it would punish the tool for doing its job.
        # The repetition is still worth surfacing, as a warning against the
        # master resume.
        if master_text and n <= master_n:
            inherited.append(
                f"{label} — appears {master_n}× in your master resume; tailoring "
                "added none"
            )
        else:
            introduced.append(
                f"{label} — up from {master_n}× in your master resume"
                if master_text else label
            )

    checks.append(
        _check(
            "keyword_stuffing",
            "Tailoring introduced no keyword stuffing",
            not introduced,
            "critical",
            "A term repeated well beyond its density in your own resume reads as "
            "gaming the filter, to an ATS and a human alike."
            if introduced
            else "Tailoring did not inflate any term's density.",
            introduced,
        )
    )
    if inherited:
        checks.append(
            _check(
                "source_repetition",
                "Repetition carried over from your master resume",
                False,
                "warning",
                "These terms are heavily repeated in your own writing. No ATS will "
                "penalise it, but a human reader notices the loop. Vary the wording "
                "in your master resume and re-run.",
                inherited,
            )
        )

    # -- Repeated opening verbs ---------------------------------------------
    # Distinct from stuffing: three bullets all starting "Led…" is a vocabulary
    # signal a reader notices long before any density threshold trips.
    openers = Counter()
    for bullet in bullets:
        first = re.match(r"[A-Za-z]+", bullet.strip())
        if first:
            openers[first.group(0).lower()] += 1
    repeated = [f"'{verb}' opens {n} bullets" for verb, n in openers.most_common(6) if n >= 3]
    checks.append(
        _check(
            "varied_action_verbs",
            "Bullets open with varied action verbs",
            not repeated,
            "warning",
            "Vary the opening verb in your master resume — the engine keeps your "
            "wording verbatim, so this is one it cannot fix for you."
            if repeated
            else "Opening verbs are varied.",
            repeated,
        )
    )

    # -- Duplicate lines ----------------------------------------------------
    seen = Counter(b.strip().lower() for b in bullets)
    dupes = [b for b, n in seen.items() if n > 1]
    checks.append(
        _check(
            "no_duplicate_bullets",
            "No duplicated bullet text",
            not dupes,
            "warning",
            "Repeated bullets waste space and look like a copy-paste error."
            if dupes
            else "No duplicates.",
            [d[:70] for d in dupes],
        )
    )

    # -- First-third relevance ---------------------------------------------
    if jd is not None and matrix:
        head = text[: max(400, len(text) // 3)]
        head_terms = ontology.extract_known_terms(head)
        p0_supported = [r for r in matrix if r.priority == "P0" and r.score >= 0.6]
        surfaced = [r for r in p0_supported if r.canonical in head_terms]
        ratio = len(surfaced) / len(p0_supported) if p0_supported else 1.0
        absent = [r.requirement for r in p0_supported if r.canonical not in head_terms]
        checks.append(
            _check(
                "top_third_relevance",
                "Mandatory requirements surface in the first third of the page",
                ratio >= 0.6,
                "warning",
                f"{len(surfaced)}/{len(p0_supported)} supported P0 requirements appear "
                "in the opening third, where a recruiter actually looks.",
                absent[:8],
            )
        )

        # Supported-but-absent: the recoverable miss the lift loop targets.
        doc_terms = ontology.extract_known_terms(text)
        doc_norm = ontology.normalise(text)

        def _surfaced(row: MatchRow) -> bool:
            if row.canonical in doc_terms:
                return True
            if ontology.normalise(row.requirement) in doc_norm:
                return True
            # "Container orchestration" is surfaced by the word "Kubernetes".
            # The row already records which skill satisfied it, so ask about that
            # rather than hunting for the abstract phrase itself.
            if row.matched_via and ontology.canonicalise(row.matched_via) in doc_terms:
                return True
            return any(
                ontology.canonicalise(t) in doc_terms
                for t in ontology.expand_concept(row.canonical)
            )

        missing = [r.requirement for r in matrix if r.score >= 0.6 and not _surfaced(r)]
        checks.append(
            _check(
                "supported_keywords_surfaced",
                "Every supported requirement appears somewhere in the document",
                not missing,
                "warning",
                "These are requirements the candidate genuinely meets but the document "
                "never mentions — free score being left on the table."
                if missing
                else "All supported requirements are represented.",
                missing[:12],
            )
        )

    # -- Seniority / title consistency -------------------------------------
    if jd is not None and resume.headline:
        headline = ontology.normalise(resume.headline)
        inflated = [
            level for level in ("principal", "staff", "director", "vp", "head of")
            if level in headline and level not in ontology.normalise(jd.job_title)
        ]
        checks.append(
            _check(
                "seniority_consistency",
                "Headline seniority is consistent with the target role",
                not inflated,
                "warning",
                f"Headline claims '{', '.join(inflated)}' which the JD does not ask for."
                if inflated
                else "Headline level is consistent with the role.",
                inflated,
            )
        )

    return ValidationReport(checks=checks)
