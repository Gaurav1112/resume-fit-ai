"""The truthfulness gate is the promise this product makes. It gets adversarial
tests: each one asserts that a specific kind of fabrication is *caught*."""

from __future__ import annotations

import pytest

from backend.models.schemas import (
    CandidateProfile,
    Certification,
    Contact,
    Education,
    ResumeBullet,
    ResumeRole,
    ResumeSection,
    Role,
    TailoredResume,
)
from backend.services import ats_validator, truth_validator

MASTER = """
ARJUN MEHTA
arjun@example.com | Bengaluru

TECHNICAL SKILLS
Languages: Java, Go
Messaging: Apache Kafka

Senior Backend Engineer, Northwind Payments
Mar 2021 - Present
- Designed a Kafka-based event processing pipeline, cutting settlement latency
  from 90 seconds to 12 seconds.
- Migrated a monolith into 7 Spring Boot microservices on OpenShift.
- Mentored 4 engineers.

EDUCATION
B.E., Computer Science - University of Pune

CERTIFICATIONS
AWS Certified Solutions Architect - Associate, Amazon Web Services, 2022
"""


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile(
        contact=Contact(name="Arjun Mehta", email="arjun@example.com"),
        total_years_experience=8.0,
        roles=[
            Role(
                company="Northwind Payments",
                title="Senior Backend Engineer",
                start_date="Mar 2021",
                end_date="Present",
                bullets=["Designed a Kafka-based event processing pipeline."],
            )
        ],
        education=[Education(institution="University of Pune", degree="B.E.")],
        certifications=[
            Certification(name="AWS Certified Solutions Architect - Associate")
        ],
    )


def resume_with(bullets: list[str], **overrides) -> TailoredResume:
    role = ResumeRole(
        company=overrides.get("company", "Northwind Payments"),
        title=overrides.get("title", "Senior Backend Engineer"),
        start_date=overrides.get("start_date", "Mar 2021"),
        end_date=overrides.get("end_date", "Present"),
        bullets=[ResumeBullet(text=b) for b in bullets],
    )
    return TailoredResume(
        contact=Contact(name="Arjun Mehta", email="arjun@example.com"),
        headline="Senior Backend Engineer",
        sections=[
            ResumeSection(
                heading="Professional Summary",
                kind="summary",
                paragraphs=[overrides.get("summary", "Backend engineer.")],
            ),
            ResumeSection(
                heading="Core Skills",
                kind="skills",
                skill_groups=overrides.get("skills", {"Languages": ["Java"]}),
            ),
            ResumeSection(heading="Professional Experience", kind="experience", roles=[role]),
        ]
        + overrides.get("extra_sections", []),
    )


def check(report, cid):
    return next(c for c in report.checks if c.id == cid)


# --------------------------------------------------------------------------- #
# Truth gate — each test asserts a fabrication is caught
# --------------------------------------------------------------------------- #
def test_supported_metric_passes(profile):
    resume = resume_with(["Cut settlement latency from 90 seconds to 12 seconds."])
    report = truth_validator.validate(resume, profile, MASTER)
    assert check(report, "no_invented_metrics").passed


def test_invented_percentage_is_caught(profile):
    resume = resume_with(["Improved throughput by 47% across the platform."])
    report = truth_validator.validate(resume, profile, MASTER)
    failure = check(report, "no_invented_metrics")
    assert not failure.passed
    assert failure.severity == "critical"
    assert any("47%" in o for o in failure.offenders)


def test_invented_scale_is_caught(profile):
    resume = resume_with(["Served 5000 requests per second."])
    report = truth_validator.validate(resume, profile, MASTER)
    assert not check(report, "no_invented_metrics").passed


def test_inflated_team_size_is_caught(profile):
    resume = resume_with(["Led a team of 25 engineers."])
    report = truth_validator.validate(resume, profile, MASTER)
    assert not check(report, "no_invented_team_size").passed


def test_unsupported_technology_is_caught(profile):
    resume = resume_with(
        ["Built services in Rust."], skills={"Languages": ["Java", "Rust"]}
    )
    report = truth_validator.validate(resume, profile, MASTER)
    failure = check(report, "no_invented_technologies")
    assert not failure.passed
    assert any("rust" in o.lower() for o in failure.offenders)


def test_an_alias_substituted_for_the_candidates_own_tool_is_caught():
    """The regression that shipped: rendering a canonical alias in place of the
    candidate's own wording.

    The master says Playwright and LBAC; the document said Cypress and ABAC.
    Because the old check canonicalised *both* sides before comparing, the two
    were the same token and the substitution was invisible — the gate reported
    green on a resume claiming products the candidate had never used.
    """
    master = (
        "Wrote the end-to-end suite in Playwright. "
        "Designed label-based access control (LBAC) for a multi-tenant product."
    )
    profile = CandidateProfile(contact=Contact(name="A", email="a@b.co"))

    substituted = TailoredResume(
        contact=Contact(name="A", email="a@b.co"),
        sections=[
            ResumeSection(
                heading="Core Skills", kind="skills",
                skill_groups={"Testing": ["Cypress"], "Security": ["ABAC"]},
            )
        ],
    )
    failure = check(
        truth_validator.validate(substituted, profile, master), "no_invented_technologies"
    )
    assert not failure.passed
    joined = " ".join(failure.offenders).lower()
    assert "cypress" in joined and "playwright" in joined, (
        "the failure must name both the wrong term and the candidate's own word"
    )

    faithful = TailoredResume(
        contact=Contact(name="A", email="a@b.co"),
        sections=[
            ResumeSection(
                heading="Core Skills", kind="skills",
                skill_groups={"Testing": ["Playwright"], "Security": ["LBAC"]},
            )
        ],
    )
    assert check(
        truth_validator.validate(faithful, profile, master), "no_invented_technologies"
    ).passed


PROMOTION_MASTER = """
ARJUN MEHTA

Technical Analyst, Kestrel Logistics
Jan 2019 - Aug 2021
- Led a team of 10 engineers.

Associate Technical Analyst, Kestrel Logistics
Jul 2016 - Dec 2018
- Built warehouse modules in Java.
"""


def promotion_profile() -> CandidateProfile:
    return CandidateProfile(
        contact=Contact(name="Arjun Mehta", email="arjun@example.com"),
        roles=[
            Role(company="Kestrel Logistics", title="Technical Analyst",
                 start_date="Jan 2019", end_date="Aug 2021", bullets=["Led a team of 10."]),
            Role(company="Kestrel Logistics", title="Associate Technical Analyst",
                 start_date="Jul 2016", end_date="Dec 2018", bullets=["Built warehouse modules."]),
        ],
    )


def promotion_resume(**overrides) -> TailoredResume:
    senior = ResumeRole(
        company="Kestrel Logistics", title="Technical Analyst",
        start_date=overrides.get("senior_start", "Jan 2019"),
        end_date=overrides.get("senior_end", "Aug 2021"),
        bullets=[ResumeBullet(text="Led a team of 10.")],
    )
    junior = ResumeRole(
        company="Kestrel Logistics", title="Associate Technical Analyst",
        start_date="Jul 2016", end_date="Dec 2018",
        bullets=[ResumeBullet(text="Built warehouse modules.")],
    )
    return TailoredResume(
        contact=Contact(name="Arjun Mehta", email="arjun@example.com"),
        sections=[ResumeSection(heading="Professional Experience", kind="experience",
                                roles=[senior, junior])],
    )


def test_a_promotion_at_one_employer_is_not_a_date_fabrication():
    """Two roles at one employer must be matched by title, not by company.

    A company-keyed lookup kept whichever role came last, so the senior role's
    dates were compared against its own predecessor's and the gate reported a
    critical fabrication on a resume that was entirely accurate. Every candidate
    ever promoted internally would have hit this.
    """
    report = truth_validator.validate(
        promotion_resume(), promotion_profile(), PROMOTION_MASTER
    )
    assert check(report, "dates_match").passed, check(report, "dates_match").offenders
    assert check(report, "titles_match").passed


def test_a_shifted_date_on_a_promoted_role_is_still_caught():
    """The fix must not buy the promotion case at the cost of the gate."""
    report = truth_validator.validate(
        promotion_resume(senior_start="Jan 2017"), promotion_profile(), PROMOTION_MASTER
    )
    failure = check(report, "dates_match")
    assert not failure.passed
    assert failure.severity == "critical"
    assert any("Jan 2017" in o for o in failure.offenders)


def test_altered_company_name_is_caught(profile):
    resume = resume_with(["Designed pipelines."], company="Northwind Global Payments Inc")
    report = truth_validator.validate(resume, profile, MASTER)
    assert not check(report, "companies_match").passed


def test_relevelled_title_is_caught(profile):
    resume = resume_with(["Designed pipelines."], title="Principal Backend Engineer")
    report = truth_validator.validate(resume, profile, MASTER)
    assert not check(report, "titles_match").passed


def test_shifted_dates_are_caught(profile):
    resume = resume_with(["Designed pipelines."], start_date="Mar 2019")
    report = truth_validator.validate(resume, profile, MASTER)
    assert not check(report, "dates_match").passed


def test_added_certification_is_caught(profile):
    extra = ResumeSection(
        heading="Certifications",
        kind="certifications",
        certifications=[Certification(name="Google Cloud Professional Architect")],
    )
    resume = resume_with(["Designed pipelines."], extra_sections=[extra])
    report = truth_validator.validate(resume, profile, MASTER)
    assert not check(report, "certifications_match").passed


def test_overclaimed_years_is_caught(profile):
    resume = resume_with(["Designed pipelines."], summary="Engineer with 15 years of experience.")
    report = truth_validator.validate(resume, profile, MASTER)
    assert not check(report, "years_claim").passed


def test_added_work_authorisation_is_caught(profile):
    resume = resume_with(
        ["Designed pipelines."], summary="US citizen with active security clearance."
    )
    report = truth_validator.validate(resume, profile, MASTER)
    assert not check(report, "no_invented_status").passed


def test_rewording_without_new_facts_passes(profile):
    resume = resume_with(
        ["Architected an event processing pipeline on Kafka for settlement flows."]
    )
    report = truth_validator.validate(resume, profile, MASTER)
    assert report.passed, [c.label for c in report.critical_failures]


# --------------------------------------------------------------------------- #
# ATS validator
# --------------------------------------------------------------------------- #
def test_keyword_stuffing_is_caught(profile):
    stuffed = ["Kubernetes Kubernetes Kubernetes deployment with Kubernetes."] * 8
    resume = resume_with(stuffed)
    report = ats_validator.validate(resume)
    failure = check(report, "keyword_stuffing")
    assert not failure.passed
    assert failure.severity == "critical"


def test_natural_density_passes(profile):
    resume = resume_with(
        [
            "Designed a Kafka event pipeline for settlement processing.",
            "Migrated the monolith into Spring Boot microservices on OpenShift.",
            "Owned the PostgreSQL schema for the ledger service.",
        ]
    )
    assert check(ats_validator.validate(resume), "keyword_stuffing").passed


def test_missing_contact_is_critical():
    resume = resume_with(["Designed pipelines."])
    resume.contact = Contact(name="", email="")
    report = ats_validator.validate(resume)
    failure = check(report, "contact_present")
    assert not failure.passed and failure.severity == "critical"


def test_nonstandard_heading_is_flagged():
    resume = resume_with(["Designed pipelines."])
    resume.sections[0].heading = "Where I've Made an Impact"
    assert not check(ats_validator.validate(resume), "standard_headings").passed


def test_inconsistent_dates_are_flagged():
    resume = resume_with(["Designed pipelines."], start_date="03/2021", end_date="Present")
    resume.sections[2].roles.append(
        ResumeRole(
            company="Kestrel",
            title="Backend Engineer",
            start_date="Jul 2018",
            end_date="Feb 2021",
            bullets=[ResumeBullet(text="Built services.")],
        )
    )
    assert not check(ats_validator.validate(resume), "date_consistency").passed


def test_risky_glyphs_are_flagged():
    resume = resume_with(["★ Designed pipelines ★"])
    assert not check(ats_validator.validate(resume), "safe_glyphs").passed


def test_duplicate_bullets_are_flagged():
    resume = resume_with(["Designed pipelines.", "Designed pipelines."])
    assert not check(ats_validator.validate(resume), "no_duplicate_bullets").passed


def test_report_score_is_bounded_and_monotonic():
    good = ats_validator.validate(
        resume_with(["Designed a Kafka pipeline for settlement processing at Northwind."])
    )
    bad = ats_validator.validate(resume_with(["★ x ★"] * 12))
    assert 0.0 <= bad.score() <= good.score() <= 100.0
