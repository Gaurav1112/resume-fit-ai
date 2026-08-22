"""Deterministic requirement ↔ evidence matching (pipeline stage 5).

Builds an index of everything the candidate can actually demonstrate, then scores
each JD requirement against it. The LLM never decides whether a match exists —
it only supplies the extracted evidence that this module indexes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.schemas import (
    CandidateProfile,
    Confidence,
    EvidenceItem,
    Gap,
    JDAnalysis,
    MatchRow,
    MatchType,
    Requirement,
)
from . import ontology

# Score thresholds that turn a numeric similarity into a reportable tier.
STRONG_SEMANTIC_MIN = 0.85
PARTIAL_MIN = 0.60
WEAK_MIN = 0.35


@dataclass
class EvidenceIndex:
    """Everything the candidate can demonstrate, keyed by canonical skill."""

    by_canonical: dict[str, EvidenceItem] = field(default_factory=dict)
    corpus: str = ""                  # normalised full master-resume text
    corpus_terms: set[str] = field(default_factory=set)

    def get(self, canonical: str) -> EvidenceItem | None:
        return self.by_canonical.get(canonical)

    def has(self, canonical: str) -> bool:
        return canonical in self.by_canonical

    def canonicals(self) -> set[str]:
        return set(self.by_canonical)


def build_evidence_index(profile: CandidateProfile, master_text: str) -> EvidenceIndex:
    """Fold profile skills, role technologies, bullets and LLM evidence into one index."""
    index = EvidenceIndex(corpus=ontology.normalise(master_text))
    index.corpus_terms = ontology.extract_known_terms(master_text)

    def add(skill: str, snippet: str, source: str, confidence: Confidence) -> None:
        canon = ontology.canonicalise(skill)
        if not canon or len(canon) < 2:
            return
        item = index.by_canonical.get(canon)
        if item is None:
            item = EvidenceItem(skill=skill, canonical=canon, confidence=confidence)
            index.by_canonical[canon] = item
        if snippet and snippet not in item.evidence:
            item.evidence.append(snippet)
        if source and source not in item.sources:
            item.sources.append(source)
        # Confidence only ever ratchets up.
        rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        if rank[confidence] > rank[item.confidence]:
            item.confidence = confidence

    # 1. Explicit evidence extracted by the LLM (highest fidelity).
    for ev in profile.evidence:
        canon = ontology.canonicalise(ev.skill)
        ev.canonical = canon
        existing = index.by_canonical.get(canon)
        if existing is None:
            index.by_canonical[canon] = ev.model_copy(deep=True)
        else:
            for snippet in ev.evidence:
                if snippet not in existing.evidence:
                    existing.evidence.append(snippet)
            for src in ev.sources:
                if src not in existing.sources:
                    existing.sources.append(src)

    # 2. Role-level technologies and bullets.
    for role in profile.roles:
        source = role.label
        for tech in role.technologies:
            add(tech, f"Used at {source}", source, "HIGH")
        for bullet in role.bullets:
            for canon in ontology.extract_known_terms(bullet):
                add(canon, bullet, source, "HIGH")

    # 3. Projects.
    for project in profile.projects:
        source = f"Project: {project.name}" if project.name else "Project"
        for tech in project.technologies:
            add(tech, project.description or f"Used in {source}", source, "MEDIUM")
        for bullet in project.bullets:
            for canon in ontology.extract_known_terms(bullet):
                add(canon, bullet, source, "MEDIUM")

    # 4. Declared skills section (weakest — a list, not a demonstration).
    for group, skills in profile.skills.items():
        for skill in skills:
            add(skill, f"Listed under {group}", "Skills section", "MEDIUM")

    # 5. Anything else the ontology recognises anywhere in the resume text.
    for canon in index.corpus_terms:
        if canon not in index.by_canonical:
            add(canon, "Mentioned in resume", "Master resume", "LOW")

    return index


def _confidence_from(score: float, evidence_confidence: Confidence) -> Confidence:
    if score >= STRONG_SEMANTIC_MIN:
        return "HIGH" if evidence_confidence in ("HIGH", "MEDIUM") else "MEDIUM"
    if score >= PARTIAL_MIN:
        return "MEDIUM" if evidence_confidence != "LOW" else "LOW"
    if score >= WEAK_MIN:
        return "LOW"
    return "NONE"


def _tier(score: float) -> MatchType:
    if score >= 0.999:
        return "EXACT"
    if score >= STRONG_SEMANTIC_MIN:
        return "STRONG_SEMANTIC"
    if score >= PARTIAL_MIN:
        return "PARTIAL"
    if score >= WEAK_MIN:
        return "WEAK"
    return "NONE"


def match_requirement(req: Requirement, index: EvidenceIndex) -> MatchRow:
    """Score one JD requirement against the candidate's evidence index."""
    canon = ontology.canonicalise(req.canonical or req.text)
    row = MatchRow(
        requirement_id=req.id,
        requirement=req.text,
        canonical=canon,
        priority=req.priority,
        kind=req.kind,
    )

    best_score = 0.0
    best_via = ""
    best_item: EvidenceItem | None = None
    best_note = ""

    # --- 1. Exact canonical hit -------------------------------------------
    item = index.get(canon)
    if item is not None:
        best_score, best_via, best_item = 1.0, item.skill or canon, item
        best_note = "Exact skill match"

    # --- 2. Abstract JD phrase satisfied by a concrete skill --------------
    # A JD says "container orchestration"; the resume says "OpenShift". Expanding
    # the concept gets us to "kubernetes", which the candidate also doesn't have
    # literally — so we then walk one edge from the expanded target. Without that
    # second hop, the most common form of semantic match silently scores zero.
    if best_score < 1.0:
        for target in ontology.expand_concept(req.canonical or req.text):
            tcanon = ontology.canonicalise(target)
            titem = index.get(tcanon)
            if titem is not None:
                if 0.9 > best_score:
                    best_score, best_via, best_item = 0.9, titem.skill or tcanon, titem
                    best_note = f"'{req.text}' is demonstrated by {titem.skill or tcanon}"
                continue
            for neighbour, weight in ontology.related(tcanon):
                nitem = index.get(neighbour)
                # Discount slightly: this is a two-hop inference, not a direct one.
                hop_score = weight * 0.97
                if nitem is not None and hop_score > best_score:
                    best_score = hop_score
                    best_via = nitem.skill or neighbour
                    best_item = nitem
                    best_note = (
                        f"'{req.text}' is demonstrated by {best_via} "
                        f"(via {tcanon})"
                    )

    # --- 3. Weighted semantic edges ---------------------------------------
    if best_score < STRONG_SEMANTIC_MIN:
        for neighbour, weight in ontology.related(canon):
            nitem = index.get(neighbour)
            if nitem is not None and weight > best_score:
                best_score, best_via, best_item = weight, nitem.skill or neighbour, nitem
                best_note = f"Semantic match via {nitem.skill or neighbour}"

    # --- 4. Substring containment in the raw corpus (last resort) ---------
    if best_score < WEAK_MIN:
        needle = ontology.normalise(req.text)
        if len(needle) >= 4 and needle in index.corpus:
            best_score = max(best_score, 0.45)
            best_via = "resume text"
            best_note = "Phrase appears in the master resume but is not a demonstrated skill"

    # --- 5. Years-of-experience penalty -----------------------------------
    if best_score > 0 and req.years_required and best_item is not None:
        have = best_item.years
        if have is not None and have < req.years_required:
            shortfall = min(1.0, have / req.years_required)
            best_score *= 0.6 + 0.4 * shortfall
            best_note += (
                f" — JD asks for {req.years_required:g}y, evidence supports ~{have:g}y"
            )

    row.score = round(min(best_score, 1.0), 3)
    row.match_type = _tier(row.score)
    row.matched_via = best_via
    row.notes = best_note
    if best_item is not None and row.score >= WEAK_MIN:
        row.evidence = best_item.evidence[:3]
        row.sources = best_item.sources[:3]
        row.confidence = _confidence_from(row.score, best_item.confidence)
    else:
        row.confidence = "NONE"
    return row


def build_matrix(jd: JDAnalysis, index: EvidenceIndex) -> list[MatchRow]:
    rows = [match_requirement(r, index) for r in jd.requirements]
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    rows.sort(key=lambda r: (order.get(r.priority, 9), -r.score))
    return rows


RISK_BY_PRIORITY = {"P0": "HIGH", "P1": "MEDIUM", "P2": "LOW", "P3": "LOW"}


def build_gaps(matrix: list[MatchRow]) -> list[Gap]:
    """Any requirement below the PARTIAL threshold is a gap worth reporting."""
    gaps: list[Gap] = []
    for row in matrix:
        if row.score >= PARTIAL_MIN:
            continue
        risk = RISK_BY_PRIORITY.get(row.priority, "LOW")
        if row.score >= WEAK_MIN and risk == "HIGH":
            risk = "MEDIUM"

        if row.score < WEAK_MIN:
            status = "No supporting evidence in master resume"
            rec = (
                f"Do NOT add '{row.requirement}' to the resume — there is nothing to "
                "support it."
            )
            if row.priority == "P0":
                rec += (
                    " This is a mandatory requirement; treat this application as a "
                    "stretch unless you have off-resume experience you can add to the "
                    "master profile first."
                )
            else:
                rec += " It is not mandatory, so the gap is unlikely to be disqualifying."
        else:
            status = f"Weak/indirect evidence via {row.matched_via or 'resume text'}"
            rec = (
                f"Only claim '{row.requirement}' if you can point to concrete work. "
                f"Closest supported skill: {row.matched_via or 'n/a'}."
            )

        gaps.append(
            Gap(
                requirement_id=row.requirement_id,
                requirement=row.requirement,
                priority=row.priority,
                kind=row.kind,
                risk=risk,  # type: ignore[arg-type]
                evidence_status=status,
                recommendation=rec,
            )
        )
    return gaps
