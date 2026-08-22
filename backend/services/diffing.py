"""Before/after comparison between the master resume and the tailored version."""

from __future__ import annotations

import difflib
from typing import Any

from ..models.schemas import CandidateProfile, MatchRow, TailoredResume
from . import ontology
from .render import to_plain_text


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compare(
    profile: CandidateProfile,
    resume: TailoredResume,
    matrix: list[MatchRow] | None = None,
) -> dict[str, Any]:
    """Classify each generated bullet as kept / rewritten / added, and list drops."""
    source_bullets: list[tuple[str, str]] = []      # (bullet, role label)
    for role in profile.roles:
        for bullet in role.bullets:
            source_bullets.append((bullet, role.label))
    for project in profile.projects:
        for bullet in project.bullets:
            source_bullets.append((bullet, f"Project: {project.name}"))

    generated: list[tuple[str, str]] = []
    for section in resume.sections:
        for role in section.roles:
            for bullet in role.bullets:
                generated.append((bullet.text, f"{role.title} @ {role.company}"))
        for bullet in section.bullets:
            generated.append((bullet.text, section.heading))

    used: set[int] = set()
    kept: list[dict[str, str]] = []
    rewritten: list[dict[str, str]] = []
    added: list[dict[str, str]] = []

    for text, where in generated:
        best_i, best_score = -1, 0.0
        for i, (src, _src_where) in enumerate(source_bullets):
            if i in used:
                continue
            score = _similar(text, src)
            if score > best_score:
                best_i, best_score = i, score

        if best_score >= 0.92 and best_i >= 0:
            used.add(best_i)
            kept.append({"text": text, "where": where})
        elif best_score >= 0.42 and best_i >= 0:
            used.add(best_i)
            rewritten.append(
                {
                    "before": source_bullets[best_i][0],
                    "after": text,
                    "where": where,
                    "similarity": f"{best_score:.0%}",
                }
            )
        else:
            added.append({"text": text, "where": where})

    removed = [
        {"text": src, "where": where}
        for i, (src, where) in enumerate(source_bullets)
        if i not in used
    ]

    # Skill reordering: which skills moved into the top of the SKILLS section.
    master_skill_order: list[str] = []
    for group in profile.skills.values():
        master_skill_order.extend(ontology.canonicalise(s) for s in group)
    new_skill_order: list[str] = []
    for section in resume.sections:
        for skills in section.skill_groups.values():
            new_skill_order.extend(ontology.canonicalise(s) for s in skills)

    promoted: list[dict[str, Any]] = []
    for new_pos, skill in enumerate(new_skill_order[:15]):
        if skill in master_skill_order:
            old_pos = master_skill_order.index(skill)
            if old_pos - new_pos >= 4:
                promoted.append({"skill": skill, "from": old_pos + 1, "to": new_pos + 1})

    master_text_len = sum(len(b) for b, _ in source_bullets)
    new_text_len = len(to_plain_text(resume))

    return {
        "kept": kept,
        "rewritten": rewritten,
        "added": added,
        "removed": removed,
        "reordered_skills": promoted,
        "stats": {
            "source_bullets": len(source_bullets),
            "generated_bullets": len(generated),
            "kept": len(kept),
            "rewritten": len(rewritten),
            "added": len(added),
            "removed": len(removed),
            "master_chars": master_text_len,
            "tailored_chars": new_text_len,
        },
    }


def unified_text_diff(before: str, after: str, context: int = 2) -> list[str]:
    return list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="master_resume",
            tofile="tailored_resume",
            lineterm="",
            n=context,
        )
    )
