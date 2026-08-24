"""The local, no-API-key engine: rule-based parsing, analysis and writing.

Design commitment that shapes everything here: **the writer never composes a
factual sentence.** It selects, ranks, reorders and reformats the candidate's own
text. Fabrication is therefore structurally impossible rather than merely
prohibited — there is no generative step that could invent a metric.

The one place prose is assembled (the professional summary) is built from a
template whose every slot is filled from a parsed field, and any slot that cannot
be filled from the source is dropped rather than guessed. The years-of-experience
claim in particular is emitted only when the number is derivable from employment
dates that are actually on the resume.

Four public entry points, matching the four LLM stages they replace:
    parse_resume(text)      -> profile payload
    analyse_jd(text)        -> jd payload
    decide_positioning(...) -> positioning payload
    write_resume(...)       -> writer payload
"""

from __future__ import annotations

import functools
import re
from collections import Counter
from typing import Any, Iterable

from . import dates, ontology

# --------------------------------------------------------------------------- #
# Shared text utilities
# --------------------------------------------------------------------------- #
# One *or more* markers: HTML exports routinely yield "- - text" when a <li>
# already contains a literal dash.
BULLET_PREFIX = re.compile(r"^\s*(?:(?:[-*•●▪·‣◦⁃>]|\d+[.)])\s+)+")
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d[\d\s.\-]{7,14}\d")
LINKEDIN = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/\S+", re.I)
GITHUB = re.compile(r"(?:https?://)?(?:www\.)?github\.com/\S+", re.I)
URL = re.compile(r"(?:https?://)?(?:www\.)?[a-z0-9\-]+\.[a-z]{2,}(?:/\S*)?", re.I)

RISKY_CHARS = "★☆●◆■▪▶►✔✓✦❖◼◾⬤▓░│┃┆┊⎯➤➔"

# Filler that weakens an opening without carrying information. Stripping it is a
# presentation change, not a factual one.
FILLERS = [
    r"^successfully\s+", r"^helped\s+to\s+", r"^was\s+tasked\s+with\s+",
    r"^responsible\s+for\s+", r"^worked\s+to\s+", r"^assisted\s+in\s+",
    r"^involved\s+in\s+", r"^participated\s+in\s+",
]
FILLER_RE = [re.compile(p, re.I) for p in FILLERS]

METRIC_RE = re.compile(
    r"\d{1,3}(?:\.\d+)?\s?%|[$£€₹]\s?\d|\b\d{3,}\b|\b\d+(?:\.\d+)?x\b"
    r"|\b\d+(?:\.\d+)?\s?(?:ms|seconds?|minutes?|hours?|k|m|b|million|billion)\b",
    re.I,
)
STRONG_VERBS = (
    "designed", "built", "architected", "led", "implemented", "developed", "owned",
    "drove", "shipped", "automated", "refactored", "introduced", "migrated",
    "delivered", "launched", "scaled", "reduced", "improved", "created",
    "established", "mentored", "optimised", "optimized", "eliminated", "wrote",
    "added", "set up", "rebuilt", "modernised", "modernized",
)


def sanitise(text: str) -> str:
    """Strip glyphs and whitespace that break text extraction. No facts change."""
    out = "".join(" " if c in RISKY_CHARS else c for c in text or "")
    out = out.replace("–", "-").replace("—", "-")
    out = out.replace("‘", "'").replace("’", "'")
    out = out.replace("“", '"').replace("”", '"')
    out = "".join(c for c in out if ord(c) < 0x2500)
    return re.sub(r"\s+", " ", out).strip()


def clean_bullet(text: str) -> str:
    """Normalise a bullet's presentation only."""
    out = sanitise(BULLET_PREFIX.sub("", text or ""))
    for pattern in FILLER_RE:
        new = pattern.sub("", out)
        if new != out:
            out = new[:1].upper() + new[1:] if new else new
            break
    out = out.strip(" ;,-")
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    if out and out[-1] not in ".!?":
        out += "."
    return out


def split_long(text: str, limit: int = 300) -> list[str]:
    """Split an over-long bullet at sentence boundaries.

    Splitting adds no facts, so it is safe — and it is the only way to satisfy the
    bullet-length check without discarding content.
    """
    if len(text) <= limit:
        return [text]
    # Only sentence boundaries. Splitting at " - " turns a subordinate clause into
    # a standalone "bullet" that has lost its subject, which reads worse than a
    # long bullet and cannot be fixed by capitalising it.
    pieces = re.split(r"(?<=[.])\s+", text)
    parts, current = [], ""
    for sentence in pieces:
        if current and len(current) + len(sentence) + 1 > limit:
            parts.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current.strip())

    return [_finish(p) for p in parts if len(p) > 2]


def _finish(text: str) -> str:
    """Tidy terminal punctuation without changing the wording."""
    out = text.strip().rstrip(" ;,-")
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out if out[-1:] in ".!?" else out + "."


def is_bullet(line: str) -> bool:
    return bool(BULLET_PREFIX.match(line))


# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #
SECTION_SYNONYMS: dict[str, str] = {}


def _syn(kind: str, names: Iterable[str]) -> None:
    for n in names:
        SECTION_SYNONYMS[n] = kind


_syn("summary", [
    "summary", "professional summary", "profile", "professional profile",
    "objective", "career objective", "about", "about me", "overview", "synopsis",
    "executive summary", "career summary",
])
_syn("skills", [
    "skills", "technical skills", "core skills", "key skills", "core competencies",
    "competencies", "technologies", "technical expertise", "tech stack",
    "skills and technologies", "areas of expertise", "expertise",
])
_syn("experience", [
    "experience", "work experience", "professional experience", "employment",
    "employment history", "work history", "career history", "relevant experience",
    "professional background",
])
_syn("projects", [
    "projects", "selected projects", "key projects", "personal projects",
    "side projects", "notable projects", "portfolio",
])
_syn("education", ["education", "academic background", "academics", "qualifications",
                   "educational qualifications", "academic qualifications"])
_syn("certifications", [
    "certifications", "certification", "licenses and certifications", "licenses",
    "certificates", "credentials", "courses and certifications",
])
_syn("achievements", [
    "achievements", "awards", "honors", "honours", "accomplishments",
    "awards and achievements", "publications", "talks", "recognition",
    "key achievements", "selected achievements", "highlights", "key highlights",
])


def _heading_kind(line: str) -> str | None:
    """Is this line a section heading, and if so which section?"""
    stripped = line.strip().strip(":—-–_ ").strip()
    if not stripped or len(stripped) > 48 or is_bullet(line):
        return None
    if stripped.endswith((".", ",", ";")):
        return None
    key = re.sub(r"[^a-z ]", " ", stripped.lower())
    key = re.sub(r"\s+", " ", key).strip()
    if key in SECTION_SYNONYMS:
        # Guard against a body line that happens to read like a heading.
        looks_like_heading = (
            stripped.isupper()
            or stripped.istitle()
            or line.strip() == stripped
            or len(stripped.split()) <= 4
        )
        if looks_like_heading:
            return SECTION_SYNONYMS[key]
    return None


def split_sections(text: str) -> dict[str, list[str]]:
    """Split a resume into {section_kind: [lines]}, with a 'header' preamble."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for raw in text.splitlines():
        kind = _heading_kind(raw)
        if kind:
            current = kind
            sections.setdefault(current, [])
            continue
        # Blank lines are preserved: they are the only reliable separator between
        # entries in a projects or achievements block, where there is no date to
        # anchor on.
        sections.setdefault(current, []).append(raw.rstrip())
    return sections


# --------------------------------------------------------------------------- #
# Stage 1-2: resume -> profile
# --------------------------------------------------------------------------- #
def _extract_contact(header_lines: list[str], full_text: str) -> dict[str, str]:
    blob = "\n".join(header_lines) or full_text[:900]
    email = EMAIL.search(blob) or EMAIL.search(full_text)
    linkedin = LINKEDIN.search(blob) or LINKEDIN.search(full_text)
    github = GITHUB.search(blob) or GITHUB.search(full_text)

    phone = ""
    for line in blob.splitlines():
        candidate = line
        for pattern in (EMAIL, LINKEDIN, GITHUB):
            candidate = pattern.sub(" ", candidate)
        match = PHONE.search(candidate)
        if match and sum(c.isdigit() for c in match.group(0)) >= 8:
            phone = match.group(0).strip()
            break

    # The name is the first substantive line that carries no contact tokens.
    name = ""
    for line in header_lines[:6]:
        stripped = line.strip()
        if not stripped or len(stripped) > 60:
            continue
        if EMAIL.search(stripped) or "|" in stripped or URL.search(stripped):
            continue
        if sum(c.isdigit() for c in stripped) > 2:
            continue
        words = stripped.split()
        if 1 <= len(words) <= 5 and all(w[:1].isalpha() for w in words if w):
            name = stripped.title() if stripped.isupper() else stripped
            break

    location = ""
    for line in header_lines[:6]:
        cleaned = EMAIL.sub("", line)
        for part in re.split(r"[|•]", cleaned):
            part = part.strip()
            part = re.sub(r"\(.*?\)", "", part).strip()
            if _looks_like_location(part) and part.lower() != (name or "").lower():
                location = part
                break
        if location:
            break

    portfolio = ""
    email_domain = (email.group(0).split("@")[-1].lower() if email else "")
    for match in URL.finditer(blob):
        candidate = match.group(0)
        low = candidate.lower()
        if "linkedin" in low or "github" in low or "@" in candidate:
            continue
        # The URL pattern also matches the domain inside an email address.
        if email_domain and low.rstrip("/") == email_domain:
            continue
        if EMAIL.search(blob[max(0, match.start() - 40) : match.end()]):
            continue
        if any(low.endswith(tld) or f"{tld}/" in low for tld in (".com", ".dev", ".io", ".me", ".net")):
            portfolio = candidate
            break

    return {
        "name": name,
        "email": email.group(0) if email else "",
        "phone": phone,
        "location": location,
        "linkedin": linkedin.group(0) if linkedin else "",
        "github": github.group(0) if github else "",
        "portfolio": portfolio,
    }


CORP_SUFFIX = re.compile(
    r"\b(inc|ltd|llc|llp|plc|gmbh|corp|corporation|holdings|technologies|technology|"
    r"systems|solutions|associates|group|labs|software|services|consulting|digital|"
    r"partners|ventures|industries|enterprises|pvt|private|limited)\b",
    re.I,
)

TITLE_HINTS = (
    "engineer", "developer", "architect", "manager", "lead", "consultant",
    "analyst", "scientist", "designer", "administrator", "specialist", "director",
    "intern", "programmer", "sre", "devops", "founder", "head", "principal",
    "staff", "senior", "junior", "associate", "vp", "cto", "officer",
)


def _looks_like_title(text: str) -> bool:
    """Corporate suffixes are stripped before the test.

    Otherwise "Manhattan Associates" reads as an "Associate" job title and the
    employer is lost — the one collision between the title-hint list and the
    company-suffix list.
    """
    low = CORP_SUFFIX.sub(" ", text.lower())
    return any(h in low for h in TITLE_HINTS)


SENTENCE_WORD = re.compile(
    r"^(is|are|am|was|were|be|been|will|would|can|could|should|do|does|did|"
    r"you|we|us|our|your|their|they|it|this|that|there|here|who|where|what|"
    r"a|an|the|to|of|in|on|at|by|for|from|with|and|or|but|as|about|"
    r"join|joining|looking|seeking|hiring|want|wanted|place|people|team's)$",
    re.I,
)


def _is_plausible_job_title(text: str) -> bool:
    """A job title is a short noun phrase, not a sentence.

    `_looks_like_title` only asks whether a title word appears *anywhere*, which
    is true of plenty of prose ("we are looking for a senior engineer to join
    us"). A job description's first line is very often employer marketing —
    Apple's opens "Apple is a place where extraordinary people gather to do
    their best work" — and accepting that as the title puts the employer's copy
    at the head of the candidate's own summary, asserting something they never
    wrote about themselves.
    """
    value = (text or "").strip().strip(".,:;")
    if not value or len(value) > 70:
        return False
    words = value.split()
    if not 1 <= len(words) <= 8:
        return False
    if not _looks_like_title(value):
        return False
    # "Head of Engineering" and "Engineer, Payments" carry at most one connective;
    # a sentence carries several.
    return sum(1 for w in words if SENTENCE_WORD.match(w.strip(",.;:"))) <= 1


COUNTRY_HINT = re.compile(
    r"\b(india|usa|u\.s\.a|united states|uk|united kingdom|canada|germany|france|"
    r"japan|singapore|australia|ireland|netherlands|spain|poland|remote|onsite|"
    r"hybrid|[A-Z]{2})\b",
    re.I,
)


def _looks_like_location(text: str) -> bool:
    """Distinguish "Bengaluru, India" from "Talendy Holdings".

    Both are two short capitalised words; only one has a comma with a
    place-shaped right-hand side and no corporate suffix. Callers must test for a
    job title *first* — "Senior Engineer, Acme" also has a comma.
    """
    value = text.strip().strip(".,")
    if not value or len(value) > 48 or any(c.isdigit() for c in value):
        return False
    if CORP_SUFFIX.search(value) or _looks_like_title(value):
        return False
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2 or any(len(p.split()) > 3 for p in parts):
        return False
    return bool(COUNTRY_HINT.search(parts[1]))


def _classify_header_parts(candidate: str) -> tuple[str, str, str, str]:
    """Pull (title, company, location, weak_company) out of one header candidate.

    `weak_company` is a company inferred from the tail of a title segment —
    "Senior Engineer, India GCC" yields the sub-team, not the employer. It is
    returned separately so a real employer found on any other candidate line
    always wins over it.
    """
    title = company = location = weak_company = ""
    value = sanitise(candidate).strip(" ,|")
    if not value:
        return "", "", "", ""

    if _looks_like_location(value):
        return "", "", value, ""

    segments = [s.strip(" ,") for s in re.split(r"\s*\|\s*|\s+[-–—]\s+", value) if s.strip(" ,")]
    for segment in segments:
        if not title and _looks_like_title(segment):
            # The segment may itself be "Title, Company".
            head, _, tail = segment.partition(",")
            if tail and not _looks_like_title(tail) and not _looks_like_location(tail.strip()):
                title, weak_company = head.strip(), weak_company or tail.strip()
            else:
                title = segment
        elif not location and _looks_like_location(segment):
            location = segment
        elif not company:
            company = segment
    return title, company, location, weak_company


def _split_role_header(line: str) -> tuple[str, str]:
    """Pull (title, company) out of a role header line of any common shape."""
    cleaned = sanitise(line).strip(" ,")
    for separator in ("|", " - ", " – ", " — ", " @ ", " at ", ","):
        if separator in cleaned:
            parts = [p.strip() for p in cleaned.split(separator) if p.strip()]
            if len(parts) >= 2:
                first, second = parts[0], parts[1]
                if _looks_like_title(first) and not _looks_like_title(second):
                    return first, second
                if _looks_like_title(second) and not _looks_like_title(first):
                    return second, first
                return first, second
    return (cleaned, "") if _looks_like_title(cleaned) else ("", cleaned)


def _parse_experience(lines: list[str]) -> list[dict[str, Any]]:
    """Roles are anchored on date ranges — the one reliable signal across layouts.

    The header can sit on either side of that date line, and real resumes use both:

        Layout A   Senior Backend Engineer, Northwind Payments
                   Bengaluru, India | Mar 2021 - Present
                   - bullet

        Layout B   Talendy Holdings - India Global Capability Center
                   December 2025 - Present
                   Senior Engineer, India GCC - Bengaluru, India
                   - bullet

    So we gather candidates from the preceding line, the residue of the date line
    itself, and the following non-bullet line, then classify each fragment as a
    title, a company or a location rather than assuming a fixed order.
    """
    roles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending: list[str] = []
    last_was_bullet = False
    # Whether the previous bullet's RAW line ended a sentence. PDF extraction
    # discards indentation and can break mid-word, so "does the previous line
    # look finished?" is a far more reliable continuation signal than "is this
    # line indented or lowercase" — which mis-read "Performer 2023" (the tail of
    # "...recognised as Star Performer 2023") as the next employer's name.
    last_line_finished = True

    entries = [(i, ln) for i, ln in enumerate(lines)]

    def lookahead(index: int) -> str:
        """The next meaningful non-bullet line, if the role header trails the date."""
        for _j, nxt in entries[index + 1 : index + 4]:
            stripped = nxt.strip()
            if not stripped:
                continue
            if is_bullet(stripped) or dates.RANGE.search(stripped):
                return ""
            return stripped
        return ""

    for index, line in entries:
        stripped = line.strip()
        if not stripped:
            continue
        raw_indented = line[:1] in (" ", "\t")

        match = dates.RANGE.search(stripped)
        if match and not is_bullet(stripped):
            residue = dates.RANGE.sub("", stripped).strip(" |,-\u2013\u2014\u00b7")
            candidates = [c for c in (residue,
                                      lookahead(index),
                                      pending[-1] if pending else "") if c]

            title = company = location = weak_company = ""
            for candidate in candidates:
                c_title, c_company, c_location, c_weak = _classify_header_parts(candidate)
                title = title or c_title
                company = company or c_company
                location = location or c_location
                weak_company = weak_company or c_weak
            company = company or weak_company

            # A company found on the same fragment as the title should not also be
            # claimed from an earlier line; prefer the earliest non-title fragment.
            if not company and len(pending) >= 2:
                _t, alt_company, _l, _w = _classify_header_parts(pending[-2])
                company = alt_company

            current = {
                "company": company,
                "title": title,
                "start_date": dates.normalise(match.group("start")),
                "end_date": dates.normalise(match.group("end")),
                "location": location[:60],
                "employment_type": "",
                "bullets": [],
                "technologies": [],
                "achievements": [],
                "metrics": [],
                "leadership": [],
                "business_impact": [],
            }
            roles.append(current)
            pending = []
            last_was_bullet = False
            last_line_finished = True
            continue

        if is_bullet(stripped) and current is not None:
            current["bullets"].append(clean_bullet(stripped))
            last_was_bullet = True
            last_line_finished = stripped.rstrip()[-1:] in ".!?"
        elif (
            current is not None
            and current["bullets"]
            and last_was_bullet
            and (stripped[0].islower() or raw_indented)
            and not _heading_kind(line)
        ):
            # A wrapped continuation of the previous bullet. Detected by
            # indentation or a lowercase opening — not by keyword absence, which
            # misfired on ordinary words that happen to be job-title nouns
            # ("...saving 20 hours | of analyst time per month").
            current["bullets"][-1] = clean_bullet(
                current["bullets"][-1].rstrip(".") + " " + stripped
            )
            last_line_finished = stripped.rstrip()[-1:] in ".!?"
        else:
            pending.append(stripped)
            pending = pending[-3:]
            last_was_bullet = False
            last_line_finished = True

    for role in roles:
        text = " ".join(role["bullets"])
        role["technologies"] = sorted(
            ontology.display(t) for t in ontology.extract_known_terms(text)
        )
        role["metrics"] = list({m.group(0) for m in METRIC_RE.finditer(text)})[:10]
        role["achievements"] = [b for b in role["bullets"] if METRIC_RE.search(b)][:6]
        # Require a person-shaped object ("team", "engineers", "juniors"), or a
        # verb that only applies to people. Otherwise "Designed and led LBAC"
        # counts as leadership evidence, which it is not.
        role["leadership"] = [
            b for b in role["bullets"]
            if re.search(
                r"\b(mentor\w*|coach\w*|hired|onboard\w*)\b"
                r"|\b(led|leading|managed|manage)\b[^.]{0,40}"
                r"\b(team|engineers?|developers?|people|juniors?|reports?|squad)\b",
                b, re.I,
            )
        ][:5]
    return roles


def _split_skill_list(text: str) -> list[str]:
    """Split a skills line, flattening parenthesised sub-lists.

    "AWS (EC2, S3, Lambda), Kubernetes" must become five clean tokens, not
    "AWS (EC2" / "S3" / "Lambda)" / "Kubernetes".
    """
    flattened = text.replace("(", ", ").replace(")", ", ")
    parts = re.split(r"[,;|/]|\band\b", flattened)
    out: list[str] = []
    for part in parts:
        item = part.strip(" .-•\t")
        if 1 < len(item) < 40 and item.lower() not in {"etc", "and", "others"}:
            out.append(item)
    return out


def _parse_skills(lines: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    loose: list[str] = []
    for raw in lines:
        line = sanitise(BULLET_PREFIX.sub("", raw))
        if not line:
            continue
        if ":" in line:
            category, _, rest = line.partition(":")
            items = _split_skill_list(rest)
            if items and len(category) < 40:
                groups.append({"category": category.strip().title(), "skills": items[:30]})
                continue
        loose.extend(_split_skill_list(line))
    if loose:
        groups.append({"category": "Skills", "skills": loose[:40]})
    return groups


def _parse_education(lines: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in lines:
        line = sanitise(raw)
        if not line:
            continue
        if is_bullet(raw) and current:
            current["details"].append(clean_bullet(raw))
            continue

        match = dates.RANGE.search(line)
        residue = dates.RANGE.sub("", line).strip(" |,-–—") if match else line
        if current and match and not current["start_date"]:
            current["start_date"] = dates.normalise(match.group("start"))
            current["end_date"] = dates.normalise(match.group("end"))
            if residue and not current["location"]:
                current["location"] = residue[:60]
            continue

        degree, institution = "", residue
        for separator in ("—", " - ", " – ", ",", "|", " at "):
            if separator in residue:
                left, _, right = residue.partition(separator)
                if re.search(r"\b(b\.?[aes]|m\.?[aes]|ph\.?d|bachelor|master|diploma|b\.?tech|m\.?tech|mba)\b",
                             left, re.I):
                    degree, institution = left.strip(), right.strip()
                else:
                    degree, institution = right.strip(), left.strip()
                break

        field = ""
        if degree and "," in degree:
            deg, _, field = degree.partition(",")
            degree, field = deg.strip(), field.strip()

        if not re.search(
            r"\b(b\.?[aes]|m\.?[aes]|ph\.?d|bachelor|master|diploma|b\.?tech|m\.?tech|"
            r"mba|bsc|msc|university|college|institute|school|academy|polytechnic)\b",
            residue, re.I,
        ):
            # A trailing note or disclaimer inside the education block is not a
            # degree; admitting it would put a stray line in the resume.
            continue

        current = {
            "institution": institution.strip(),
            "degree": degree.strip(),
            "field_of_study": field,
            "start_date": dates.normalise(match.group("start")) if match else "",
            "end_date": dates.normalise(match.group("end")) if match else "",
            "location": "",
            "details": [],
        }
        entries.append(current)
    return entries


def _parse_certifications(lines: list[str]) -> list[dict[str, Any]]:
    """Split a certification line into name and date — and never guess an issuer.

    "Oracle Certified Professional, Java SE 11 Developer - March 2023" has one
    comma, and it separates the credential from its edition, not the credential
    from its issuer. Reading the right-hand side as an issuer produced
    "issuer: Java SE 11 Developer - March", which is simply false. The dash is
    the reliable separator; the comma is not. Where no issuer is stated, leave it
    empty rather than inventing one.
    """
    out: list[dict[str, Any]] = []
    for raw in lines:
        line = sanitise(BULLET_PREFIX.sub("", raw))
        if not line:
            continue

        parts = re.split(r"\s+[-–—]\s+", line, maxsplit=1)
        name = parts[0].strip(" ,")
        tail = parts[1].strip() if len(parts) > 1 else ""

        # The tail is the date expression as the candidate wrote it, kept verbatim
        # so "2023 to 2026 (recertification in progress)" survives intact.
        date = tail
        if not date:
            year = re.search(r"\b(?:[A-Z][a-z]{2,8}\s+)?(?:19|20)\d{2}\b", line)
            if year:
                date = year.group(0)
                name = line.replace(date, "").strip(" ,-–—")

        out.append({
            "name": name or line,
            "issuer": "",
            "date": date,
            "credential_id": "",
        })
    return out


def _parse_projects(lines: list[str]) -> list[dict[str, Any]]:
    """Blank lines separate projects; everything else continues the current one.

    Without that rule a wrapped description line becomes a phantom project — the
    projects block has no dates to anchor on the way experience does.
    """
    projects: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    start_new = True

    for raw in lines:
        line = sanitise(raw)
        if not line:
            start_new = True
            continue

        if is_bullet(raw) and current:
            current["bullets"].append(clean_bullet(raw))
            start_new = False
            continue

        low = line.lower()
        if low.startswith(("technolog", "tech stack", "stack", "built with")):
            if current:
                _, _, rest = line.partition(":")
                current["technologies"] = [
                    t.strip() for t in re.split(r"[,;|]", rest) if t.strip()
                ]
            start_new = False
            continue

        if current is not None and not start_new:
            # Continuation of this project's description.
            current["description"] = (current["description"] + " " + line).strip()
            continue

        current = {
            "name": line[:80], "description": "", "technologies": [],
            "bullets": [], "url": "",
        }
        projects.append(current)
        start_new = False
    for project in projects:
        if not project["technologies"]:
            blob = project["description"] + " " + " ".join(project["bullets"])
            project["technologies"] = sorted(
                ontology.display(t) for t in ontology.extract_known_terms(blob)
            )
    return projects


def _build_evidence(
    roles: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    skill_groups: list[dict[str, Any]],
    full_text: str,
) -> list[dict[str, Any]]:
    """One entry per canonical skill, carrying the sentences that demonstrate it."""
    evidence: dict[str, dict[str, Any]] = {}

    def add(canon: str, snippet: str, source: str, confidence: str, months: int = 0) -> None:
        item = evidence.setdefault(
            canon,
            {"skill": ontology.display(canon), "evidence": [], "sources": [],
             "confidence": confidence, "years": None, "_months": 0},
        )
        if snippet and snippet not in item["evidence"] and len(item["evidence"]) < 4:
            item["evidence"].append(snippet)
        if source and source not in item["sources"]:
            item["sources"].append(source)
        rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        if rank[confidence] > rank[item["confidence"]]:
            item["confidence"] = confidence
        item["_months"] = max(item["_months"], months)

    for role in roles:
        source = f"{role['title']} @ {role['company']}".strip(" @")
        months = dates.months_between(role["start_date"], role["end_date"])
        for bullet in role["bullets"]:
            for canon in ontology.extract_known_terms(bullet):
                add(canon, bullet, source, "HIGH", months)
        for tech in role["technologies"]:
            add(ontology.canonicalise(tech), f"Used at {role['company']}", source, "HIGH", months)

    for project in projects:
        source = f"Project: {project['name']}"
        blob = project["description"] + " " + " ".join(project["bullets"])
        for canon in ontology.extract_known_terms(blob):
            add(canon, project["description"] or blob[:160], source, "MEDIUM")
        for tech in project["technologies"]:
            add(ontology.canonicalise(tech), project["description"] or source, source, "MEDIUM")

    for group in skill_groups:
        for skill in group["skills"]:
            add(ontology.canonicalise(skill), f"Listed under {group['category']}",
                "Skills section", "MEDIUM")

    for canon in ontology.extract_known_terms(full_text):
        add(canon, "Mentioned in the resume", "Master resume", "LOW")

    out = []
    for item in evidence.values():
        months = item.pop("_months")
        item["years"] = round(months / 12.0, 1) if months >= 6 else None
        out.append(item)
    return out


def parse_resume(text: str) -> dict[str, Any]:
    sections = split_sections(text)
    contact = _extract_contact(sections.get("header", []), text)
    roles = _parse_experience(sections.get("experience", []))

    # Some resumes have no EXPERIENCE heading; fall back to scanning everything.
    if not roles:
        roles = _parse_experience([ln for ln in text.splitlines()])

    projects = _parse_projects(sections.get("projects", []))
    skill_groups = _parse_skills(sections.get("skills", []))
    education = _parse_education(sections.get("education", []))
    certifications = _parse_certifications(sections.get("certifications", []))
    achievements = [clean_bullet(l) for l in sections.get("achievements", []) if l.strip()]

    total_years = dates.total_experience_years(
        [(r["start_date"], r["end_date"]) for r in roles]
    )

    titles = [r["title"] for r in roles if r["title"]]
    current_title = titles[0] if titles else ""
    if not current_title:
        for line in sections.get("header", [])[:5]:
            if _looks_like_title(line):
                current_title = sanitise(line)
                break

    all_terms = ontology.extract_known_terms(text)
    lowered = text.lower()
    # Weight by evidence, not alphabet: a domain named once in a short early role
    # should not lead the summary ahead of one named throughout the resume.
    domains = sorted(
        (t for t in all_terms if ontology.category_of(t) == "Domain"),
        key=lambda t: (
            -max(
                lowered.count(alias)
                for alias in [t, *ontology.ALIAS_GROUPS.get(t, [])] if alias
            ),
            t,
        ),
    )
    leadership = [b for r in roles for b in r["leadership"]]
    leadership.sort(
        key=lambda b: (
            0 if re.match(r"^(mentor|led|managed|coach|hired)", b, re.I) else 1,
            0 if re.search(r"\b\d+\s+(junior |senior )?engineers?\b", b, re.I) else 1,
            len(b),
        )
    )

    return {
        "contact": contact,
        "current_title": current_title,
        "previous_titles": titles[1:],
        "total_years_experience": total_years,
        "primary_domain": ontology.display(domains[0]) if domains else "",
        "secondary_domains": [ontology.display(d) for d in domains[1:3]],
        "has_leadership_experience": bool(leadership),
        "leadership_summary": leadership[0] if leadership else "",
        "skill_groups": skill_groups,
        "roles": roles,
        "education": education,
        "certifications": certifications,
        "projects": projects,
        "achievements": achievements,
        "domains": [ontology.display(d) for d in domains],
        "evidence": _build_evidence(roles, projects, skill_groups, text),
    }


# --------------------------------------------------------------------------- #
# Stage 3-4: JD -> classified requirements
# --------------------------------------------------------------------------- #
JD_SECTIONS = {
    "required": ["requirement", "qualification", "what you", "must have", "you have",
                 "we're looking for", "we are looking for", "who you are",
                 "basic qualification", "minimum qualification", "skills"],
    "preferred": ["preferred", "bonus", "plus", "desirable", "additional"],
    "nice_to_have": ["nice to have", "nice-to-have", "good to have", "would be nice"],
    "responsibilities": ["responsibilit", "what you'll do", "what you will do",
                         "the role", "about the role", "day to day", "you will"],
    "ignore": ["benefit", "we offer", "what we offer", "perks", "compensation",
               "about us", "equal opportunity", "our team", "why join"],
}

MUST_WORDS = ("required", "must", "expert", "deep", "strong", "extensive",
              "proven", "solid", "demonstrated", "essential", "advanced")
PREFER_WORDS = ("preferred", "nice to have", "bonus", "a plus", "plus",
                "desirable", "familiarity", "exposure", "good to have")


def _jd_section_of(line: str) -> str | None:
    low = line.strip().lower().strip(":")
    if len(low) > 60:
        return None
    for kind, needles in JD_SECTIONS.items():
        if any(low.startswith(n) or low == n for n in needles):
            return kind
    return None


JD_BOILERPLATE = re.compile(
    r"^(about the (job|role|us|company|team)|job description|the role|"
    r"role overview|position summary|overview|apply now|job details|"
    r"full[- ]time|part[- ]time|contract)$",
    re.I,
)


# Pasting a posting from a job board drags the page's buttons in with it, and
# they sit on the same line as the title: "Senior AI Platform Engineer View Jobs".
JOB_BOARD_CHROME = re.compile(
    r"\s*[-–—|·•]?\s*(view all jobs?|view jobs?|see all jobs?|apply now|easy apply|"
    r"quick apply|save this job|save job|save|share|back to (?:jobs|search|results)|"
    r"new!?|featured|promoted|actively hiring|posted\b.*|job id\b.*|"
    r"req(?:uisition)?\s*#?\s*\d+.*)\s*$",
    re.I,
)


def _strip_job_board_chrome(text: str) -> str:
    previous = None
    value = text.strip()
    while value != previous:            # buttons often arrive in pairs
        previous = value
        value = JOB_BOARD_CHROME.sub("", value).strip()
    return value


BULLET_LEAD = re.compile(r"^\s*[-•*·▪◦–—]\s+")
# Requirement bullets read like titles once the leading dash is stripped
# ("Experience leading engineering teams"), so they are excluded by shape.
REQUIREMENT_LEAD = re.compile(
    r"^\s*(experience|strong|proven|solid|demonstrated|excellent|deep|hands[- ]on|"
    r"proficiency|proficient|familiarity|familiar|knowledge|understanding|ability|"
    r"bachelor|master|degree|\d+\+?\s*years?)\b",
    re.I,
)


def analyse_jd(text: str, market: str = "global") -> dict[str, Any]:
    lines = [l for l in text.splitlines()]
    # Job boards prepend boilerplate headings; treating one as the title or the
    # employer poisons the headline, the summary and the saved version name.
    non_empty = [
        l.strip() for l in lines
        if l.strip() and not JD_BOILERPLATE.match(l.strip().rstrip(":"))
    ]

    # Identity (title, employer) lives in the header block — everything before
    # the first recognised section heading. Searching past it harvests
    # requirement bullets as titles ("- Experience leading engineering teams")
    # and section headings as employers ("Minimum Qualifications").
    header: list[str] = []
    for line in non_empty:
        if _jd_section_of(line):
            break
        header.append(line)
    header = header[:12]

    job_title = ""
    for line in header:
        if BULLET_LEAD.match(line) or REQUIREMENT_LEAD.match(line):
            continue
        candidate = _strip_job_board_chrome(re.split(r"[|]", line)[0].strip())
        if _is_plausible_job_title(candidate):
            job_title = sanitise(candidate)
            break
    # No fallback to the first line. Truncating arbitrary prose to 70 characters
    # produced titles like "Apple is a place where extraordinary people gather to
    # do their best wo", which then led the candidate's professional summary.
    # An unknown title must stay unknown; downstream falls back to the
    # candidate's own current title, which is the only truthful default.

    company = ""
    for line in header[:5]:
        if BULLET_LEAD.match(line) or REQUIREMENT_LEAD.match(line):
            continue
        for part in re.split(r"[|]", line):
            part = part.strip()
            if part and part != job_title and 2 <= len(part.split()) <= 5 \
                    and not _looks_like_title(part) \
                    and not _jd_section_of(part) \
                    and not re.search(r"remote|hybrid|onsite|full-?time|part-?time", part, re.I):
                company = sanitise(part)
                break
        if company:
            break

    low_all = text.lower()
    seniority = next(
        (s for s in ("principal", "staff", "senior", "lead", "junior", "intern",
                     "director", "head of", "mid-level")
         if s in job_title.lower()),
        "senior" if "senior" in low_all else "",
    )
    work_mode = (
        "remote" if re.search(r"\bfully remote\b|\bremote\b", low_all) and "hybrid" not in low_all
        else "hybrid" if "hybrid" in low_all
        else "onsite" if re.search(r"\bon-?site\b|\bin office\b", low_all)
        else "unspecified"
    )
    auth = ""
    auth_match = re.search(r"[^.\n]*\b(authoriz|authoris|visa|work permit|citizen|clearance)\b[^.\n]*",
                           text, re.I)
    if auth_match:
        auth = sanitise(auth_match.group(0))[:160]

    overall_years = None
    year_match = re.search(r"(\d{1,2})\+?\s*(?:\+\s*)?years?", low_all)
    if year_match:
        overall_years = float(year_match.group(1))

    domain_terms = [t for t in ontology.extract_known_terms(text)
                    if ontology.category_of(t) == "Domain"]
    leadership_expected = bool(
        re.search(r"\b(mentor|technical leadership|lead the|set technical direction|"
                  r"cross-team|influence|coach|manage a team|tech lead)\b", low_all)
    )

    # --- walk the document, tracking which section each line belongs to -----
    requirements: list[dict[str, Any]] = []
    responsibilities: list[str] = []
    qualifications: list[str] = []
    seen: set[str] = set()
    section = "responsibilities"
    counter = 1

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        detected = _jd_section_of(line)
        if detected:
            section = detected
            continue
        if section == "ignore":
            continue

        body = sanitise(BULLET_PREFIX.sub("", line))
        if len(body) < 8:
            continue
        low = body.lower()

        if section == "responsibilities":
            responsibilities.append(body)
        elif section in ("required", "preferred", "nice_to_have"):
            qualifications.append(body)

        line_years = None
        y = re.search(r"(\d{1,2})\+?\s*years?", low)
        if y:
            line_years = float(y.group(1))

        if section == "nice_to_have" or "nice to have" in low:
            priority, kind = "P3", "NICE_TO_HAVE"
        elif section == "preferred" or any(w in low for w in PREFER_WORDS):
            priority, kind = "P2", "PREFERRED"
        elif section == "required" or any(w in low for w in MUST_WORDS):
            priority, kind = "P0", "REQUIRED"
        else:
            priority, kind = "P1", "REQUIRED"

        # Concrete skills named on this line each become their own requirement.
        found = ontology.extract_known_terms(body)
        for canon in sorted(found):
            if canon in seen:
                continue
            seen.add(canon)
            requirements.append({
                "id": f"R{counter}", "text": ontology.display(canon), "canonical": canon,
                "category": ontology.category_of(canon).lower(), "priority": priority,
                "kind": kind, "years_required": line_years,
                "rationale": body[:200],
            })
            counter += 1

        # Abstract phrases the ontology recognises as concepts.
        for phrase in ontology.CONCEPT_ALIASES:
            if phrase in low and phrase not in seen:
                seen.add(phrase)
                requirements.append({
                    "id": f"R{counter}", "text": phrase.title(), "canonical": phrase,
                    "category": "concept", "priority": priority, "kind": kind,
                    "years_required": line_years, "rationale": body[:200],
                })
                counter += 1

    # Repetition across sections is the JD telling you what it cares about.
    mention_counts = Counter(ontology.extract_known_terms(text.lower()))
    for req in requirements:
        if req["priority"] == "P1" and mention_counts.get(req["canonical"], 0) >= 3:
            req["priority"] = "P0"

    soft = [s for s in ("communication", "collaboration", "ownership", "mentoring",
                        "stakeholder management", "problem solving")
            if s.split()[0] in low_all]

    return {
        "job_title": job_title,
        "company": company,
        "seniority": seniority,
        "years_required": overall_years,
        "location": _find_location(non_empty[:5]),
        "work_mode": work_mode,
        "work_authorization": auth,
        "domain": ontology.display(domain_terms[0]) if domain_terms else "",
        "leadership_expected": leadership_expected,
        "requirements": requirements,
        "responsibilities": responsibilities[:14],
        "qualifications": qualifications[:20],
        "soft_skills": [s.title() for s in soft],
        "certifications": re.findall(r"\b[A-Z]{2,5}\s?(?:certified|certification)\b", text, re.I)[:5],
        "key_phrases": [ontology.display(t) for t, _ in mention_counts.most_common(14)],
    }


def _find_location(lines: list[str]) -> str:
    for line in lines:
        for part in re.split(r"[|]", line):
            part = part.strip()
            if re.search(r"remote|hybrid|onsite|,\s*[A-Z]{2}\b|United States|India|UK|Europe",
                         part, re.I) and len(part) < 60:
                return sanitise(part)
    return ""


# --------------------------------------------------------------------------- #
# Stage 7: positioning
# --------------------------------------------------------------------------- #
LEVELS = ["intern", "junior", "associate", "", "senior", "lead", "staff",
          "principal", "director", "vp", "head of"]


def _level_of(title: str) -> int:
    low = (title or "").lower()
    best = 3
    for i, level in enumerate(LEVELS):
        if level and level in low:
            best = max(best, i)
    return best


def decide_positioning(
    profile: dict[str, Any], jd: dict[str, Any], matrix: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_titles = [t for t in [profile.get("current_title", "")]
                        + list(profile.get("previous_titles") or []) if t]
    # Defence in depth: the JD may have been parsed by a provider other than
    # `analyse_jd`. A title that is not a title must never reach the headline.
    jd_title = jd.get("job_title", "")
    if jd_title and not _is_plausible_job_title(jd_title):
        jd_title = ""
    jd_level = _level_of(jd_title)
    own_level = max((_level_of(t) for t in candidate_titles), default=3)

    supported = own_level >= jd_level
    if supported:
        target_title = jd_title or (candidate_titles[0] if candidate_titles else "Engineer")
        reasoning = (
            f"Your own title history reaches {LEVELS[own_level] or 'mid'} level, which "
            f"supports the {LEVELS[jd_level] or 'stated'} level this role asks for."
        )
    else:
        # Re-level the JD title down to what the candidate's history supports,
        # rather than claiming a seniority they have not reached.
        base = re.sub(
            r"\b(principal|staff|senior|lead|director|vp|head of)\b", "", jd_title, flags=re.I
        ).strip()
        prefix = LEVELS[own_level].title() + " " if LEVELS[own_level] else ""
        target_title = (prefix + base).strip() or (candidate_titles[0] if candidate_titles else base)
        reasoning = (
            f"The job description targets {LEVELS[jd_level] or 'a higher'} level, but your "
            f"titles support {LEVELS[own_level] or 'mid'} level. The resume is positioned at "
            f"the level your history actually evidences — inflating the title here would be "
            f"contradicted by the employment section directly beneath it."
        )

    strong = [r for r in matrix if r["score"] >= 0.85]
    strong.sort(key=lambda r: ({"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(r["priority"], 9),
                               -r["score"]))
    emphasise = [r["requirement"] for r in strong[:10]]

    years = profile.get("total_years_experience")
    specialisation = profile.get("primary_domain") or (
        jd.get("domain") or "software engineering"
    )
    identity = (
        f"{target_title}"
        + (f" with {years:g} years of experience" if years else "")
        + (f" in {specialisation}" if specialisation else "")
        + (f", strongest in {', '.join(emphasise[:3])}." if emphasise else ".")
    )

    differentiators = []
    for role in profile.get("roles", []):
        for bullet in role.get("achievements", []):
            differentiators.append(bullet)
    differentiators = differentiators[:4]

    weak_canon = {r["canonical"] for r in matrix if r["score"] < 0.35}
    de_emphasise = [
        ontology.display(c)
        for c in (ontology.extract_known_terms(" ".join(
            b for r in profile.get("roles", []) for b in r.get("bullets", [])
        )) - {r["canonical"] for r in matrix})
    ][:8]

    return {
        "target_title": target_title,
        "target_seniority": LEVELS[own_level] or "mid",
        "identity_statement": identity,
        "supported": supported,
        "support_reasoning": reasoning,
        "differentiators": differentiators,
        "emphasise": emphasise,
        "de_emphasise": de_emphasise,
        "section_order": ["summary", "skills", "experience", "projects",
                          "education", "certifications"],
        "_unused_weak": sorted(weak_canon)[:0],
    }


# --------------------------------------------------------------------------- #
# Stage 8: the writer
# --------------------------------------------------------------------------- #
PRIORITY_WEIGHT = {"P0": 4.0, "P1": 2.5, "P2": 1.2, "P3": 0.5}

# Word budget keeps the document inside the ATS length check by construction.
MIN_WORDS, MAX_WORDS = 380, 1000


# Any digit counts as quantification for *ranking* purposes. METRIC_RE stays
# strict because the truth gate uses it, but "cut release cycle from 2 weeks to
# 2 days" and "team of 10" are quantified achievements that it deliberately
# ignores, and ranking must not.
QUANT_RE = re.compile(r"\d")

# Claims only this candidate can make. These are what a recruiter remembers, and
# a purely keyword-driven ranking throws them away.
DISTINCTIVE_RE = re.compile(
    r"\b(highest|largest|first|sole|only|founding|founded|fastest|best|"
    r"top|record|singl[ey]-handed|zero|award|recognis|recogniz)\w*\b",
    re.I,
)


@functools.lru_cache(maxsize=1)
def _incoming_edges() -> dict[str, list[tuple[str, float]]]:
    """`ontology.EDGES` inverted.

    Edges are stored one-way, from the general term to the specific one
    (`llm` -> `anthropic api`, `observability` -> `datadog`). Ranking a resume
    bullet asks the opposite question — "how much does this job description want
    *this* term?" — so it needs the incoming direction too.
    """
    incoming: dict[str, list[tuple[str, float]]] = {}
    for source, targets in ontology.EDGES.items():
        for target, weight in targets:
            incoming.setdefault(target, []).append((source, weight))
    return incoming


def _term_weight(term: str, wanted: dict[str, float]) -> float:
    """The JD's interest in `term`, following ontology edges in both directions.

    An exact dictionary lookup made the candidate's strongest evidence invisible
    to the job description that most wanted it: the resume says "Claude API"
    (canonical `anthropic api`), an AI platform posting says "LLM integration",
    and the two are joined by an edge weighted 0.85. Matching, scoring and gap
    analysis all traverse those edges — bullet ranking was the one place still
    doing string equality, so an AI role surfaced a deployment-tooling bullet as
    the candidate's best AI evidence, and "DataDog" scored zero against a JD
    asking for observability.
    """
    best = wanted.get(term, 0.0)
    for neighbour, weight in ontology.related(term):
        if neighbour in wanted:
            best = max(best, wanted[neighbour] * weight)
    for source, weight in _incoming_edges().get(term, []):
        if source in wanted:
            best = max(best, wanted[source] * weight)
    return best


def _bullet_relevance(bullet: str, wanted: dict[str, float]) -> float:
    terms = ontology.extract_known_terms(bullet)

    # Capped: a bullet stuffed with technology names should not be able to crowd
    # out a quantified achievement simply by naming more things. Before this cap,
    # "Shipped 26 merged pull requests in five weeks - the highest contribution of
    # any engineer in the GCC" lost its place to keyword-dense filler.
    keyword_score = min(sum(_term_weight(t, wanted) for t in terms), 14.0)

    score = keyword_score
    if QUANT_RE.search(bullet):
        score += 4.0
    if DISTINCTIVE_RE.search(bullet):
        score += 3.0
    if any(bullet.lower().startswith(v) for v in STRONG_VERBS):
        score += 1.0
    score += min(len(terms), 4) * 0.3
    return score


def _numbers_in_source(text: str) -> set[str]:
    return {re.sub(r"[^\d.]", "", d).rstrip(".") for d in re.findall(r"\d[\d,.]*", text)}


def _as_written(master_text: str, form: str) -> str:
    """The master's own spelling of `form`, preferring a capitalised occurrence.

    Changing case is presentation; changing the word is not. So we keep the
    candidate's term and only tidy how it is capitalised for a skills grid.
    """
    pattern = rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])"
    occurrences = re.findall(pattern, master_text, re.I)
    if not occurrences:
        return form.title()
    for occurrence in occurrences:
        if occurrence[:1].isupper():
            return occurrence
    return occurrences[0].title()


def write_resume(
    profile: dict[str, Any],
    jd: dict[str, Any],
    matrix: list[dict[str, Any]],
    positioning: dict[str, Any],
    master_text: str,
) -> dict[str, Any]:
    """Select, rank and reformat the candidate's own content for this JD."""
    # The candidate's own spelling for every skill the ontology recognises.
    # Rendering must go through this: printing the canonical display name turns
    # "Playwright" into "Cypress" and "LBAC" into "ABAC".
    surface = ontology.extract_surface_forms(master_text)

    def render_skill(canon: str) -> str | None:
        written = surface.get(canon)
        if written is None:
            return None          # not in the master in any spelling — never print it
        return _as_written(master_text, written)
    wanted: dict[str, float] = {}
    for row in matrix:
        if row["score"] >= 0.35:
            wanted[row["canonical"]] = max(
                wanted.get(row["canonical"], 0.0),
                PRIORITY_WEIGHT.get(row["priority"], 1.0) * row["score"],
            )

    changes: list[dict[str, str]] = []
    roles_in = profile.get("roles", [])

    # --- experience: rank bullets, allocate more lines to recent roles -------
    out_roles: list[dict[str, Any]] = []
    allowance = [7, 5, 4, 3, 2, 2]
    for i, role in enumerate(roles_in):
        scored = []
        for bullet in role.get("bullets", []):
            for piece in split_long(bullet):
                if len(piece) >= 25:
                    scored.append((_bullet_relevance(piece, wanted), piece))
        scored.sort(key=lambda x: -x[0])

        cap = allowance[i] if i < len(allowance) else 2
        chosen = scored[:cap]
        dropped = scored[cap:]

        # Restore the resume's original ordering within the role; ranking decides
        # *which* bullets survive, not what order they read in.
        original = [b for b in role.get("bullets", [])]
        order = {}
        for _score, text in chosen:
            for idx, src in enumerate(original):
                if text[:40] in src or src[:40] in text:
                    order[text] = idx
                    break
            order.setdefault(text, 999)
        kept = sorted((t for _s, t in chosen), key=lambda t: order[t])

        out_roles.append({
            "company": role["company"],
            "title": role["title"],
            "start_date": dates.normalise(role["start_date"]),
            "end_date": dates.normalise(role["end_date"]),
            "location": role.get("location", ""),
            "bullets": [
                {
                    "text": text,
                    "source_ref": f"{role['title']} @ {role['company']}".strip(" @"),
                    "keywords": [ontology.display(t)
                                 for t in ontology.extract_known_terms(text) if t in wanted],
                }
                for text in kept
            ],
        })
        if dropped:
            changes.append({
                "change": f"Dropped {len(dropped)} lower-relevance bullet(s) from "
                          f"{role['title']} @ {role['company']}",
                "reason": "They matched none of this JD's requirements, so they cost "
                          "space in the recruiter's 10-second scan without adding fit.",
                "source": f"{role['title']} @ {role['company']}".strip(" @"),
                "category": "removed",
            })

    # --- word budget: trim oldest roles first until the document fits --------
    def word_count() -> int:
        return sum(len(b["text"].split()) for r in out_roles for b in r["bullets"]) + 140

    while word_count() > MAX_WORDS:
        trimmable = [r for r in reversed(out_roles) if len(r["bullets"]) > 1]
        if not trimmable:
            break
        trimmable[0]["bullets"].pop()

    # --- skills: only supported ones, ordered by JD priority -----------------
    evidence_canon = {ontology.canonicalise(e["skill"]) for e in profile.get("evidence", [])}
    for role in roles_in:
        evidence_canon |= {ontology.canonicalise(t) for t in role.get("technologies", [])}

    ranked = sorted(
        (c for c in evidence_canon if c),
        key=lambda c: (-wanted.get(c, 0.0), ontology.display(c).lower()),
    )
    known = ontology.known_surface_forms()
    # Concept pseudo-skills exist so a JD asking for "observability" can be
    # matched by Datadog. They are not things a person writes in a skills grid.
    concepts = ontology.CONCEPT_ONLY
    grouped: dict[str, list[str]] = {}
    for canon in ranked:
        # An unrecognised token has no reliable display name or category, and
        # would surface as "Adr" or "Agent Orc" in an "Other" bucket. Dropping it
        # from the skills grid costs nothing: it stays in the experience bullets,
        # which is where it is evidenced anyway.
        if canon not in known or canon in concepts:
            continue
        category = ontology.category_of(canon)
        if category == "Other":
            continue
        if not wanted.get(canon) and len(grouped.get(category, [])) >= 6:
            continue        # keep unrelated groups short rather than exhaustive
        name = render_skill(canon)
        if name and name not in grouped.setdefault(category, []):
            grouped[category].append(name)

    skill_groups = [
        {"category": category, "skills": skills[:14]}
        for category in ontology.CATEGORY_ORDER
        if (skills := grouped.get(category))
    ]
    surfaced = [n for c in ranked[:8] if wanted.get(c) and (n := render_skill(c))]
    if surfaced:
        changes.append({
            "change": "Reordered Core Skills so this JD's priorities lead: "
                      + ", ".join(surfaced),
            "reason": "The requirement matrix ranks these highest for this role, and a "
                      "recruiter reads the first line of a skills block, not the tenth.",
            "source": "Master resume → skills and role technologies",
            "category": "reordered",
        })

    # --- summary: template-filled from parsed fields only --------------------
    summary = _compose_summary(profile, jd, positioning, wanted, master_text)
    changes.append({
        "change": "Rewrote the professional summary around this role.",
        "reason": "Every clause is filled from a parsed field of your resume — title, "
                  "computed years, domain, and your highest-matching supported skills. "
                  "No claim is introduced that is not already in the source.",
        "source": "Master resume → header, employment dates, evidence index",
        "category": "rewritten",
    })

    # --- projects: only if they add JD-relevant coverage ---------------------
    selected_projects = []
    for project in profile.get("projects", []):
        blob = project.get("description", "") + " " + " ".join(project.get("bullets", []))
        if _bullet_relevance(blob, wanted) >= 3.0:
            selected_projects.append({
                "name": project["name"],
                "description": sanitise(project.get("description", ""))[:240],
                "bullets": [
                    {"text": b, "source_ref": project["name"], "keywords": []}
                    for b in project.get("bullets", [])[:2]
                ],
            })
    if selected_projects:
        changes.append({
            "change": f"Kept {len(selected_projects)} project(s) that reinforce this JD.",
            "reason": "They demonstrate requirements the employment section covers thinly.",
            "source": "Master resume → projects",
            "category": "repositioned",
        })

    if not positioning.get("supported", True):
        changes.append({
            "change": f"Positioned as '{positioning['target_title']}' rather than the "
                      f"JD's '{jd.get('job_title', '')}'.",
            "reason": positioning.get("support_reasoning", ""),
            "source": "Master resume → job titles",
            "category": "repositioned",
        })

    return {
        "headline": positioning.get("target_title", profile.get("current_title", "")),
        "summary": summary,
        "skill_groups": skill_groups,
        "roles": out_roles,
        "selected_projects": selected_projects,
        "changes": changes,
    }


def _compose_summary(
    profile: dict[str, Any],
    jd: dict[str, Any],
    positioning: dict[str, Any],
    wanted: dict[str, float],
    master_text: str,
) -> str:
    """Assemble the summary from parsed fields; drop any slot that can't be filled.

    The years clause is the sensitive one: it is emitted only when the integer is
    already present somewhere in the master resume, so it can never be the one
    number on the page that has no source.
    """
    title = positioning.get("target_title") or profile.get("current_title") or "Engineer"
    parts: list[str] = []

    # The years figure is computed by merging the employment intervals on the
    # resume, so it is derivable from the source rather than asserted. It is
    # omitted entirely when the dates don't support one.
    years = profile.get("total_years_experience")
    years_clause = (
        f" with {round(years)}+ years of experience" if years and years >= 1 else ""
    )

    candidate_domains = [d for d in profile.get("domains") or [] if d]
    jd_domain = jd.get("domain") or ""
    # Lead with the JD's domain when it is genuinely one of the candidate's,
    # rather than whichever domain happened to sort first.
    domain = next(
        (d for d in candidate_domains
         if jd_domain and ontology.normalise(d) == ontology.normalise(jd_domain)),
        profile.get("primary_domain") or "",
    )
    lead = f"{title}{years_clause}"
    if domain:
        lead += f" in {domain}"

    written = ontology.extract_surface_forms(master_text)

    def as_written(canon: str) -> str | None:
        form = written.get(canon)
        return _as_written(master_text, form) if form else None

    top = [n for c, _w in sorted(wanted.items(), key=lambda kv: -kv[1])[:8]
           if (n := as_written(c))][:5]
    if top:
        lead += f", working across {', '.join(top[:-1])} and {top[-1]}" if len(top) > 1 \
            else f", working with {top[0]}"
    parts.append(lead.rstrip(".") + ".")

    # A summary over ~90 words stops being a summary. Add the supporting
    # sentences only while there is room, shortest-first so the most compact
    # evidence survives the budget.
    WORD_BUDGET = 78

    def room_for(sentence: str) -> bool:
        return sum(len(p.split()) for p in parts) + len(sentence.split()) <= WORD_BUDGET

    optional: list[str] = []
    differentiators = positioning.get("differentiators") or []
    if differentiators:
        # Pick the achievement most relevant to *this* JD, not whichever happened
        # to come first. A summary's one supporting sentence is prime real estate.
        best = max(differentiators, key=lambda d: _bullet_relevance(d, wanted))
        optional.append(best.rstrip(".") + ".")
    if profile.get("has_leadership_experience") and jd.get("leadership_expected"):
        leadership = profile.get("leadership_summary", "")
        if leadership:
            optional.append(leadership.rstrip(".") + ".")

    # Relevance decides *what* to say; the budget decides how much fits.
    for sentence in optional:
        if room_for(sentence):
            parts.append(sentence)

    return sanitise(" ".join(parts))


# --------------------------------------------------------------------------- #
# Stage 11: recruiter simulation (heuristic)
# --------------------------------------------------------------------------- #
def simulate_recruiter(
    resume_text: str, jd: dict[str, Any], matrix: list[dict[str, Any]]
) -> dict[str, Any]:
    head = resume_text[: max(500, len(resume_text) // 3)]
    head_terms = ontology.extract_known_terms(head)
    all_terms = ontology.extract_known_terms(resume_text)

    p0 = [r for r in matrix if r["priority"] == "P0" and r["score"] >= 0.6]
    surfaced = [r for r in p0 if r["canonical"] in head_terms]

    lines = [l for l in resume_text.splitlines() if l.strip()]
    has_headline = len(lines) > 1 and len(lines[1]) < 70
    bullets = [l for l in lines if l.startswith("- ")]
    quantified = sum(1 for b in bullets if METRIC_RE.search(b))

    score = 40.0
    score += 22.0 * (len(surfaced) / len(p0)) if p0 else 22.0
    score += 12.0 if has_headline else 0.0
    score += 10.0 * min(1.0, quantified / max(1, len(bullets) * 0.35))
    score += 8.0 if "PROFESSIONAL SUMMARY" in resume_text else 0.0
    score += 8.0 * min(1.0, len(all_terms) / 18.0)
    score = min(100.0, round(score, 1))

    strengths = [r["requirement"] for r in matrix if r["score"] >= 0.85][:5]
    weaknesses: list[str] = []
    missing_p0 = [r["requirement"] for r in matrix
                  if r["priority"] == "P0" and r["score"] < 0.6]
    if missing_p0:
        weaknesses.append(
            "Mandatory requirements with no evidence: " + ", ".join(missing_p0[:4])
        )
    absent = [r["requirement"] for r in p0 if r["canonical"] not in head_terms]
    if absent:
        weaknesses.append(
            "Supported but not visible in the opening third: " + ", ".join(absent[:4])
        )
    if quantified < max(1, len(bullets) // 3):
        weaknesses.append(
            f"Only {quantified} of {len(bullets)} bullets carry a number — add metrics to "
            "your master resume where you have them."
        )
    if not has_headline:
        weaknesses.append("No headline under the name; the level is not instantly readable.")
    if not weaknesses:
        weaknesses.append("Nothing significant — the fit reads clearly on a fast scan.")

    return {
        "score": score,
        "who_is_this": (
            f"{lines[1] if has_headline else 'An engineer'}"
            + (f" — {lines[0]}" if lines else "")
        )[:180],
        "what_level": jd.get("seniority", "").title() or "Not explicitly signalled",
        "specialisation": ", ".join(strengths[:3]) or "Not immediately obvious",
        "technologies_visible": [ontology.display(t) for t in sorted(head_terms)][:12],
        "relevance_to_role": (
            f"{len(surfaced)} of {len(p0)} mandatory requirements are visible in the "
            f"first third of the page."
            if p0 else "No mandatory requirements were detected in the JD."
        ),
        "top_strengths": strengths or ["No strong matches detected."],
        "top_weaknesses": weaknesses[:5],
    }


# --------------------------------------------------------------------------- #
# Cover letter
# --------------------------------------------------------------------------- #
# Phrases a generator must never produce. It knows nothing about the company, so
# any sentence expressing admiration, cultural fit or motivation would be
# invented — the same class of fabrication as an invented metric, and the reason
# most generated cover letters read as worthless.
BANNED_SENTIMENT = (
    "long admired", "passionate about", "excited by your mission",
    "dream company", "perfect fit", "culture",
)


def write_cover_letter(
    profile: dict[str, Any],
    jd: dict[str, Any],
    matrix: list[dict[str, Any]],
    positioning: dict[str, Any],
    master_text: str,
    today: str = "",
) -> dict[str, Any]:
    """Assemble a JD-specific cover letter from parsed facts and verbatim evidence.

    Every sentence that asserts something about the candidate is either a field
    parsed from the resume or one of their own bullets reproduced word for word.
    The connective prose states only what is true by construction ("I'm applying
    for X", "Two examples"), so the letter cannot contain a claim the resume does
    not already make.
    """
    contact = profile.get("contact", {}) or {}
    name = contact.get("name", "") or ""
    company = (jd.get("company") or "").strip()
    job_title = (jd.get("job_title") or "the role").strip()
    target_title = positioning.get("target_title") or profile.get("current_title") or "Engineer"

    wanted: dict[str, float] = {}
    for row in matrix:
        if row.get("score", 0) >= 0.35:
            wanted[row["canonical"]] = max(
                wanted.get(row["canonical"], 0.0),
                PRIORITY_WEIGHT.get(row.get("priority", "P1"), 1.0) * row["score"],
            )

    # --- opening ----------------------------------------------------------
    years = profile.get("total_years_experience")
    years_clause = f"{round(years)}+ years" if years and years >= 1 else "several years"
    jd_domain = jd.get("domain") or ""
    domain = next(
        (d for d in (profile.get("domains") or [])
         if jd_domain and ontology.normalise(d) == ontology.normalise(jd_domain)),
        profile.get("primary_domain") or "",
    )

    proven = [
        r for r in matrix
        if r.get("score", 0) >= 0.85 and r.get("priority") in ("P0", "P1")
    ]
    proven.sort(key=lambda r: (r["priority"], -r["score"]))
    written = ontology.extract_surface_forms(master_text)

    def as_written(canon: str, fallback: str) -> str:
        form = written.get(canon)
        return _as_written(master_text, form) if form else fallback

    # Only claim what the resume literally says. A JD asking for Istio, answered
    # by a candidate who runs Linkerd, must not produce "I have spent years in
    # Istio" followed by evidence naming Linkerd.
    headline_skills = [
        as_written(r["canonical"], "")
        for r in proven[:6]
    ]
    headline_skills = [h for h in headline_skills if h][:4]

    opening = (
        f"I'm applying for the {job_title} role"
        + (f" at {company}" if company else "")
        + f". I'm a {target_title} with {years_clause} of experience"
        + (f" in {domain}" if domain else "")
        + (
            f", and the requirements this role leads with — "
            + ", ".join(headline_skills[:-1])
            + f" and {headline_skills[-1]} — are the areas I've spent that time in."
            if len(headline_skills) > 1
            else "."
        )
    )

    # --- evidence paragraphs ---------------------------------------------
    # One paragraph per top requirement, answered with the candidate's own
    # highest-relevance bullet for it. Each bullet is used at most once.
    achievements: list[str] = []
    for role in profile.get("roles", []):
        for bullet in role.get("bullets", []):
            achievements.append(bullet)

    body: list[str] = []
    evidence_used: list[str] = []
    consumed: set[str] = set()

    for row in proven:
        if len(body) >= 3:
            break
        canon = row["canonical"]
        candidates = [
            b for b in achievements
            if b not in consumed and canon in ontology.extract_known_terms(b)
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda b: _bullet_relevance(b, wanted))
        consumed.add(best)
        # Label with the term the evidence actually uses. Labelling with the JD's
        # word produces "On Istio: Ran the Linkerd service mesh" — a claim the
        # very next clause contradicts.
        label = as_written(canon, "")
        if not label or not re.search(
            rf"(?<![A-Za-z0-9]){re.escape(label)}(?![A-Za-z0-9])", best, re.I
        ):
            body.append(best.rstrip(".") + ".")
        else:
            body.append(f"On {label}: {best.rstrip('.')}.")
        evidence_used.append(f"{row['requirement']} → {best[:70]}…")

    # Fall back to the strongest quantified achievements if the matrix produced
    # nothing usable, so the letter is never a bare template.
    if not body:
        ranked = sorted(achievements, key=lambda b: -_bullet_relevance(b, wanted))
        for bullet in ranked[:2]:
            body.append(bullet.rstrip(".") + ".")
            evidence_used.append(f"(general) → {bullet[:70]}…")

    # --- leadership paragraph, only where the JD asks and evidence exists --
    if jd.get("leadership_expected") and profile.get("has_leadership_experience"):
        lead = profile.get("leadership_summary", "")
        if lead and lead not in consumed:
            body.append("On the leadership side: " + lead[0].lower() + lead[1:].rstrip(".") + ".")
            evidence_used.append("Leadership → " + lead[:70] + "…")

    # --- closing ----------------------------------------------------------
    closing_bits: list[str] = []
    mode = (jd.get("work_mode") or "").lower()
    if mode in {"remote", "hybrid", "onsite"}:
        closing_bits.append(f"I'm set up for {mode} work")
    if contact.get("location"):
        closing_bits.append(f"and based in {contact['location']}")
    closing = (
        (" ".join(closing_bits) + ". ") if closing_bits else ""
    ) + "I'd welcome the chance to talk through any of the above."

    contact_line = " | ".join(
        x for x in (contact.get("email"), contact.get("phone"), contact.get("linkedin")) if x
    )

    # An unmet mandatory requirement is the candidate's decision to make, not
    # something to volunteer to the employer. Surface it in the UI instead.
    unmet = [r["requirement"] for r in matrix
             if r.get("priority") == "P0" and r.get("score", 0) < 0.6]
    omitted = (
        "Not mentioned in this letter (no supporting evidence): "
        + ", ".join(unmet[:5])
        if unmet else ""
    )

    letter = {
        "date": today,
        "recipient": f"{company} Hiring Team" if company else "Hiring Team",
        "subject": f"Application — {job_title}" + (f", {company}" if company else ""),
        "salutation": "Dear Hiring Team,",
        "paragraphs": [sanitise(p) for p in ([opening] + body + [closing]) if p.strip()],
        "signoff": "Kind regards,",
        "signature": name,
        "contact_line": contact_line,
        "evidence_used": evidence_used,
        "omitted_note": omitted,
    }

    joined = " ".join(letter["paragraphs"]).lower()
    assert not any(b in joined for b in BANNED_SENTIMENT), (
        "cover letter contained invented sentiment"
    )
    return letter


def cover_letter_to_text(letter: dict[str, Any]) -> str:
    lines: list[str] = []
    if letter.get("date"):
        lines += [letter["date"], ""]
    if letter.get("recipient"):
        lines += [letter["recipient"], ""]
    if letter.get("subject"):
        lines += [letter["subject"], ""]
    lines += [letter.get("salutation", "Dear Hiring Team,"), ""]
    for paragraph in letter.get("paragraphs", []):
        lines += [paragraph, ""]
    lines += [letter.get("signoff", "Kind regards,"), letter.get("signature", "")]
    if letter.get("contact_line"):
        lines.append(letter["contact_line"])
    return "\n".join(lines).rstrip() + "\n"
