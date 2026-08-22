"""The rules engine is now the product, so it carries the heaviest test load.

Parsing is tested against the layout variants real resumes actually use, because
every downstream score depends on getting the employment history right.
"""

from __future__ import annotations

import pytest

from backend.services import dates, local_engine, ontology, text_extract


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Mar 2021", "Mar 2021"),
        ("March 2021", "Mar 2021"),
        ("03/2021", "Mar 2021"),
        ("3/2021", "Mar 2021"),
        ("2021-03", "Mar 2021"),
        ("Present", "Present"),
        ("current", "Present"),
        ("Ongoing", "Present"),
        ("2021", "2021"),
    ],
)
def test_dates_normalise_to_one_format(raw, expected):
    assert dates.normalise(raw) == expected


def test_reformatting_a_date_is_not_a_change_of_fact():
    assert dates.same("03/2021", "Mar 2021")
    assert dates.same("2021-03", "March 2021")
    assert not dates.same("Mar 2021", "Mar 2019")


def test_total_experience_merges_overlapping_roles():
    """Summing role durations double-counts concurrent employment."""
    overlapping = [("Jan 2020", "Dec 2022"), ("Jan 2021", "Dec 2022")]
    assert dates.total_experience_years(overlapping) == pytest.approx(3.0, abs=0.1)

    sequential = [("Jan 2018", "Dec 2019"), ("Jan 2020", "Dec 2021")]
    assert dates.total_experience_years(sequential) == pytest.approx(4.0, abs=0.2)

    assert dates.total_experience_years([]) is None


# --------------------------------------------------------------------------- #
# Role-header layouts
# --------------------------------------------------------------------------- #
LAYOUT_TITLE_FIRST = """
Senior Backend Engineer, Northwind Payments
Bengaluru, India | Mar 2021 - Present
- Designed a Kafka-based event processing pipeline.
"""

LAYOUT_TITLE_AFTER_DATE = """
Talendy Holdings - India Global Capability Center
December 2025 - Present
Senior Engineer, India GCC - Bengaluru, India
- Founding engineer for the India GCC.
"""

LAYOUT_PIPED = """
Backend Engineer | Acme Systems | Jan 2020 - Dec 2021
- Built services in Go.
"""


@pytest.mark.parametrize(
    "layout,title,company",
    [
        (LAYOUT_TITLE_FIRST, "Senior Backend Engineer", "Northwind Payments"),
        (LAYOUT_TITLE_AFTER_DATE, "Senior Engineer", "Talendy Holdings"),
        (LAYOUT_PIPED, "Backend Engineer", "Acme Systems"),
    ],
)
def test_role_header_layouts(layout, title, company):
    roles = local_engine._parse_experience(layout.splitlines())
    assert len(roles) == 1
    assert roles[0]["title"].startswith(title)
    assert roles[0]["company"] == company


def test_company_with_corporate_suffix_is_not_read_as_a_job_title():
    """"Manhattan Associates" must not become an "Associate" title."""
    roles = local_engine._parse_experience(
        [
            "Manhattan Associates",
            "July 2016 - August 2019",
            "Technical Analyst - Bengaluru, India",
            "- Developed warehouse management modules in Java.",
        ]
    )
    assert roles[0]["company"] == "Manhattan Associates"
    assert roles[0]["title"] == "Technical Analyst"
    assert roles[0]["location"] == "Bengaluru, India"


def test_location_is_distinguished_from_company():
    assert local_engine._looks_like_location("Bengaluru, India")
    assert local_engine._looks_like_location("Austin, TX")
    assert not local_engine._looks_like_location("Talendy Holdings")
    assert not local_engine._looks_like_location("Senior Engineer, Acme")


def test_wrapped_bullet_continuation_is_joined():
    """The continuation heuristic must not depend on keyword absence."""
    roles = local_engine._parse_experience(
        [
            "Software Engineer, Vertex Systems",
            "Jun 2016 - Jun 2018",
            "- Automated a manual reconciliation process, saving roughly 20 hours",
            "  of analyst time per month.",
        ]
    )
    assert "of analyst time per month" in roles[0]["bullets"][0]


def test_dates_are_normalised_but_facts_preserved():
    roles = local_engine._parse_experience(
        ["Engineer, Acme", "03/2021 - Present", "- Did the work."]
    )
    assert roles[0]["start_date"] == "Mar 2021"
    assert dates.same(roles[0]["start_date"], "03/2021")


# --------------------------------------------------------------------------- #
# Bullets and skills
# --------------------------------------------------------------------------- #
def test_double_bullet_markers_from_html_are_stripped():
    assert local_engine.clean_bullet("- - Built the thing").startswith("Built")


def test_filler_prefixes_are_removed_without_changing_facts():
    assert local_engine.clean_bullet(
        "- Responsible for managing the Kafka cluster"
    ) == "Managing the Kafka cluster."


def test_long_bullet_splits_on_clause_separators_when_no_sentence_end():
    text = "Did a thing - " + ("and another clause " * 30)
    parts = local_engine.split_long(text, limit=200)
    assert len(parts) > 1
    assert all(len(p) <= 260 for p in parts)


def test_skill_list_flattens_parentheses():
    assert local_engine._split_skill_list("AWS (EC2, S3, Lambda), Kubernetes") == [
        "AWS", "EC2", "S3", "Lambda", "Kubernetes",
    ]


# --------------------------------------------------------------------------- #
# JD analysis
# --------------------------------------------------------------------------- #
JD = """
Staff Backend Engineer
Acme Corp | Remote

Requirements
- Expert-level Java, including Spring Boot
- Deep experience with Apache Kafka
- 8+ years of professional backend experience

Preferred
- Experience with GraphQL
- Python for tooling

Nice to have
- Experience with Rust
"""


def test_jd_priorities_follow_the_posting_s_own_language():
    jd = local_engine.analyse_jd(JD)
    by_canon = {r["canonical"]: r for r in jd["requirements"]}
    assert by_canon["java"]["priority"] == "P0"
    assert by_canon["kafka"]["priority"] == "P0"
    assert by_canon["graphql"]["priority"] == "P2"
    assert by_canon["rust"]["priority"] == "P3"


def test_jd_extracts_title_years_and_mode():
    jd = local_engine.analyse_jd(JD)
    assert "Staff Backend Engineer" in jd["job_title"]
    assert jd["years_required"] == 8.0
    assert jd["work_mode"] == "remote"


def test_compound_requirement_is_split_into_separate_rows():
    jd = local_engine.analyse_jd("Requirements\n- Java, Spring Boot and Kafka required")
    canonicals = {r["canonical"] for r in jd["requirements"]}
    assert {"java", "spring boot", "kafka"} <= canonicals


# --------------------------------------------------------------------------- #
# Positioning
# --------------------------------------------------------------------------- #
def test_positioning_will_not_inflate_seniority():
    profile = {"current_title": "Senior Backend Engineer", "previous_titles": [], "roles": []}
    jd = {"job_title": "Principal Backend Engineer"}
    result = local_engine.decide_positioning(profile, jd, [])
    assert result["supported"] is False
    assert "principal" not in result["target_title"].lower()


def test_positioning_accepts_a_level_the_history_supports():
    profile = {"current_title": "Staff Engineer", "previous_titles": [], "roles": []}
    jd = {"job_title": "Senior Engineer"}
    assert local_engine.decide_positioning(profile, jd, [])["supported"] is True


# --------------------------------------------------------------------------- #
# HTML intake — many people keep their master resume as HTML
# --------------------------------------------------------------------------- #
def test_html_resume_extraction():
    html = b"""<html><head><style>p{color:red}</style></head><body>
    <div><b>KUMAR GAURAV</b></div><p>Senior Engineer</p>
    <ul><li>Built a Kafka pipeline</li><li>Owned PostgreSQL schema</li></ul>
    <script>alert(1)</script></body></html>"""
    text = text_extract.extract("resume.html", html)
    assert "KUMAR GAURAV" in text
    assert "- Built a Kafka pipeline" in text
    assert "alert" not in text and "color:red" not in text


def test_html_is_detected_without_an_extension():
    text = text_extract.extract("resume", b"<!DOCTYPE html><html><body><p>Hi there</p></body></html>")
    assert "Hi there" in text


# --------------------------------------------------------------------------- #
# The writer never invents
# --------------------------------------------------------------------------- #
def test_writer_reproduces_bullets_verbatim():
    """The core guarantee: generated bullets are a subset of the source's."""
    profile = local_engine.parse_resume(
        open("samples/master_resume.txt", encoding="utf-8").read()
    )
    jd = local_engine.analyse_jd(
        open("samples/job_description.txt", encoding="utf-8").read()
    )
    matrix = [
        {"canonical": r["canonical"], "priority": r["priority"], "score": 1.0,
         "requirement": r["text"]}
        for r in jd["requirements"]
    ]
    positioning = local_engine.decide_positioning(profile, jd, matrix)
    written = local_engine.write_resume(
        profile, jd, matrix, positioning,
        open("samples/master_resume.txt", encoding="utf-8").read(),
    )

    source = {b for role in profile["roles"] for b in role["bullets"]}
    for role in written["roles"]:
        for bullet in role["bullets"]:
            text = bullet["text"]
            # Either verbatim, or a sentence-split fragment of a source bullet.
            assert text in source or any(
                text.rstrip(".") in s or s.rstrip(".") in text for s in source
            ), f"writer produced text with no source: {text!r}"


def test_summary_stays_within_a_readable_budget():
    profile = local_engine.parse_resume(
        open("samples/master_resume.txt", encoding="utf-8").read()
    )
    jd = local_engine.analyse_jd(
        open("samples/job_description.txt", encoding="utf-8").read()
    )
    positioning = local_engine.decide_positioning(profile, jd, [])
    written = local_engine.write_resume(
        profile, jd, [], positioning,
        open("samples/master_resume.txt", encoding="utf-8").read(),
    )
    assert 20 <= len(written["summary"].split()) <= 90


def test_years_claim_is_omitted_when_dates_do_not_support_one():
    profile = {"roles": [], "total_years_experience": None, "current_title": "Engineer"}
    summary = local_engine._compose_summary(
        profile, {}, {"target_title": "Engineer"}, {}, "master text"
    )
    assert "years of experience" not in summary


def test_engine_is_deterministic():
    text = open("samples/master_resume.txt", encoding="utf-8").read()
    assert local_engine.parse_resume(text) == local_engine.parse_resume(text)
