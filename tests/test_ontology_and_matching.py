"""The matching engine is the part of the system that decides what is true about
a person's experience, so it gets the most direct tests."""

from __future__ import annotations

import pytest

from backend.models.schemas import (
    CandidateProfile,
    EvidenceItem,
    JDAnalysis,
    Requirement,
    Role,
)
from backend.services import matching, ontology


# --------------------------------------------------------------------------- #
# Ontology
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "surface,expected",
    [
        ("K8s", "kubernetes"),
        ("kubernetes", "kubernetes"),
        ("Postgres", "postgresql"),
        ("PostgreSQL", "postgresql"),
        ("Golang", "go"),
        ("Apache Kafka", "kafka"),
        ("Spring-Boot", "spring boot"),
        ("CI/CD", "cicd"),
        ("Amazon Web Services", "aws"),
        ("Express.js", "express"),
        ("Micro-services", "microservices"),
        ("event-driven architecture", "event driven architecture"),
    ],
)
def test_canonicalise_maps_aliases(surface, expected):
    assert ontology.canonicalise(surface) == expected


def test_extract_known_terms_respects_word_boundaries():
    terms = ontology.extract_known_terms(
        "Ongoing work on the register; going forward we care about R&D."
    )
    # "go" must not fire inside "Ongoing"/"going", nor "r" inside arbitrary words.
    assert "go" not in terms


def test_extract_known_terms_finds_real_mentions():
    terms = ontology.extract_known_terms(
        "Built Kafka pipelines in Go, deployed on Kubernetes with Terraform."
    )
    assert {"kafka", "go", "kubernetes", "terraform"} <= terms


def test_concept_expansion():
    assert "kubernetes" in ontology.expand_concept("container orchestration")
    assert "terraform" in ontology.expand_concept("infrastructure as code")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile(
        current_title="Senior Backend Engineer",
        total_years_experience=8.0,
        skills={"Languages": ["Java", "Go"], "Data": ["PostgreSQL"]},
        roles=[
            Role(
                company="Northwind",
                title="Senior Backend Engineer",
                start_date="Mar 2021",
                end_date="Present",
                bullets=[
                    "Designed a Kafka-based event processing pipeline for settlement.",
                    "Migrated a monolith into Spring Boot microservices on OpenShift.",
                ],
                technologies=["Java", "Spring Boot", "Kafka", "OpenShift", "PostgreSQL"],
            )
        ],
        evidence=[
            EvidenceItem(
                skill="Apache Kafka",
                evidence=["Designed a Kafka-based event processing pipeline"],
                sources=["Northwind"],
                confidence="HIGH",
            )
        ],
    )


@pytest.fixture
def index(profile):
    master = " ".join(b for r in profile.roles for b in r.bullets)
    return matching.build_evidence_index(profile, master)


def _req(rid, text, priority="P0", years=None) -> Requirement:
    return Requirement(
        id=rid,
        text=text,
        canonical=ontology.canonicalise(text),
        priority=priority,
        years_required=years,
    )


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def test_exact_match_scores_full(index):
    row = matching.match_requirement(_req("R1", "Kafka"), index)
    assert row.match_type == "EXACT"
    assert row.score == 1.0
    assert row.evidence


def test_openshift_satisfies_kubernetes_semantically(index):
    row = matching.match_requirement(_req("R2", "Kubernetes"), index)
    assert row.match_type == "STRONG_SEMANTIC"
    assert 0.85 <= row.score < 1.0
    assert "openshift" in row.matched_via.lower()


def test_abstract_concept_matches_concrete_skill(index):
    row = matching.match_requirement(_req("R3", "container orchestration"), index)
    assert row.score >= 0.85
    assert row.match_type in ("STRONG_SEMANTIC", "EXACT")


def test_event_driven_architecture_matched_via_kafka(index):
    row = matching.match_requirement(_req("R4", "event-driven architecture"), index)
    assert row.score >= 0.85


def test_unsupported_requirement_scores_none(index):
    row = matching.match_requirement(_req("R5", "Rust"), index)
    assert row.match_type == "NONE"
    assert row.score == 0.0
    assert row.evidence == []
    assert row.confidence == "NONE"


def test_gaps_never_recommend_adding_unsupported_skill(index):
    jd = JDAnalysis(requirements=[_req("R5", "Rust"), _req("R6", "COBOL")])
    matrix = matching.build_matrix(jd, index)
    gaps = matching.build_gaps(matrix)
    assert len(gaps) == 2
    for gap in gaps:
        assert "do not add" in gap.recommendation.lower()
        assert gap.risk == "HIGH"       # both are P0


def test_optional_requirements_are_lower_risk(index):
    jd = JDAnalysis(requirements=[_req("R7", "Rust", priority="P2")])
    gaps = matching.build_gaps(matching.build_matrix(jd, index))
    assert gaps[0].risk == "LOW"


def test_matrix_orders_by_priority_then_score(index):
    jd = JDAnalysis(
        requirements=[
            _req("A", "Rust", priority="P2"),
            _req("B", "Kafka", priority="P0"),
            _req("C", "Terraform", priority="P0"),
        ]
    )
    rows = matching.build_matrix(jd, index)
    assert [r.requirement_id for r in rows][:2] == ["B", "C"]
    assert rows[-1].requirement_id == "A"


def test_years_shortfall_reduces_score():
    profile = CandidateProfile(
        evidence=[
            EvidenceItem(skill="Python", evidence=["Scripts"], confidence="HIGH", years=2.0)
        ]
    )
    index = matching.build_evidence_index(profile, "Python scripting work")
    strong = matching.match_requirement(_req("R1", "Python"), index)
    short = matching.match_requirement(_req("R2", "Python", years=10.0), index)
    assert short.score < strong.score
    assert "10y" in short.notes
