"""Render a TailoredResume to plain text.

The plain-text rendering is the canonical form: it is what the ATS validator and
the truth validator inspect, what the recruiter simulator reads, and what the
"copy plain text" button yields. The DOCX and PDF exporters render the same
structure independently — so what you validate is what you ship.
"""

from __future__ import annotations

from ..models.schemas import Contact, TailoredResume

SECTION_ORDER = [
    "summary", "skills", "experience", "projects", "achievements",
    "education", "certifications",
]


def contact_line(contact: Contact) -> str:
    parts = [p for p in (contact.email, contact.phone, contact.location) if p]
    links = [p for p in (contact.linkedin, contact.github, contact.portfolio) if p]
    return " | ".join(parts + links)


def to_plain_text(resume: TailoredResume, *, width: int = 98) -> str:
    lines: list[str] = []

    if resume.contact.name:
        lines.append(resume.contact.name)
    if resume.headline:
        lines.append(resume.headline)
    line = contact_line(resume.contact)
    if line:
        lines.append(line)
    lines.append("")

    order = {kind: i for i, kind in enumerate(SECTION_ORDER)}
    sections = sorted(resume.sections, key=lambda s: order.get(s.kind, 99))

    for section in sections:
        body: list[str] = []

        if section.kind == "summary":
            body.extend(p for p in section.paragraphs if p.strip())

        elif section.kind == "skills":
            for group, skills in section.skill_groups.items():
                if skills:
                    body.append(f"{group}: {', '.join(skills)}")

        elif section.kind == "experience":
            for i, role in enumerate(section.roles):
                if i:
                    body.append("")
                body.append(f"{role.title}, {role.company}")
                meta = " | ".join(
                    p for p in (f"{role.start_date} - {role.end_date}".strip(" -"), role.location)
                    if p
                )
                if meta:
                    body.append(meta)
                body.extend(f"- {b.text}" for b in role.bullets if b.text.strip())

        elif section.kind == "projects":
            for project_bullet in section.bullets:
                body.append(f"- {project_bullet.text}")
            for para in section.paragraphs:
                body.append(para)

        elif section.kind == "education":
            for edu in section.education:
                head = ", ".join(p for p in (edu.degree, edu.field_of_study) if p)
                body.append(f"{head} — {edu.institution}" if head else edu.institution)
                meta = " | ".join(
                    p for p in (f"{edu.start_date} - {edu.end_date}".strip(" -"), edu.location)
                    if p
                )
                if meta:
                    body.append(meta)
                body.extend(f"- {d}" for d in edu.details)

        elif section.kind == "certifications":
            for cert in section.certifications:
                bits = [cert.name]
                if cert.issuer:
                    bits.append(cert.issuer)
                if cert.date:
                    bits.append(cert.date)
                body.append(" — ".join(bits))

        else:  # achievements and anything else
            body.extend(f"- {b.text}" for b in section.bullets if b.text.strip())
            body.extend(section.paragraphs)

        if not any(b.strip() for b in body):
            continue

        lines.append(section.heading.upper())
        lines.extend(body)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def all_bullet_texts(resume: TailoredResume) -> list[str]:
    out: list[str] = []
    for section in resume.sections:
        out.extend(b.text for b in section.bullets)
        for role in section.roles:
            out.extend(b.text for b in role.bullets)
        out.extend(section.paragraphs)
    return [t for t in out if t and t.strip()]
