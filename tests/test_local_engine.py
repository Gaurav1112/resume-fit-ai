"""The rules engine is now the product, so it carries the heaviest test load.

Parsing is tested against the layout variants real resumes actually use, because
every downstream score depends on getting the employment history right.
"""

from __future__ import annotations

import re

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


def test_long_bullet_splits_at_sentence_boundaries():
    text = "Built the service. " + ("Then shipped it to production. " * 12)
    parts = local_engine.split_long(text, limit=200)
    assert len(parts) > 1
    assert all(p[0].isupper() and p.endswith(".") for p in parts)


def test_a_long_bullet_with_no_sentence_boundary_is_left_whole():
    """A subordinate clause is not a bullet.

    Splitting "…for a product - authored the ADR and carried it through review"
    at the dash yields "authored the ADR…" with its subject gone, and the
    fragment then gets re-ranked away from the clause it belonged to. Leaving the
    bullet long and reporting it is the better failure: the fix belongs in the
    master resume.
    """
    text = "Did a thing - " + ("and another clause " * 30)
    parts = local_engine.split_long(text, limit=200)
    assert len(parts) == 1
    assert parts[0].startswith("Did a thing")


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


EMPLOYER_MARKETING_JD = """Apple is a place where extraordinary people gather to do their best work. \
Together we create products and experiences people once couldn't have imagined.

Do you love working on challenges that no one has solved yet? As a member of our \
Software Engineering group, you'll build the next generation of services.

Minimum Qualifications
- 10+ years of experience in SaaS
- Strong Java and distributed systems background
- Experience with REST APIs and data modelling

Preferred Qualifications
- Apache Kafka, Kubernetes
- Experience leading engineering teams
"""


def test_employer_marketing_never_becomes_the_job_title():
    """The regression that shipped into a real application.

    When no line looked like a title, the parser fell back to the JD's first
    line truncated to 70 characters. Apple's posting opens with marketing copy,
    so the candidate's professional summary began "Apple is a place where
    extraordinary people gather to do their best wo with 10+ years of
    experience" — the employer's words asserted as the candidate's own.
    """
    jd = local_engine.analyse_jd(EMPLOYER_MARKETING_JD)
    assert jd["job_title"] == "", f"invented a title: {jd['job_title']!r}"
    assert "Apple" not in jd["job_title"]
    assert "Apple" not in (jd.get("company") or "")


def test_requirement_bullets_are_not_mistaken_for_the_title():
    """"Experience leading engineering teams" is a requirement, not a title.

    It survives the noun-phrase test, so identity is read only from the header
    block above the first section heading.
    """
    jd = local_engine.analyse_jd(EMPLOYER_MARKETING_JD)
    assert "Experience leading" not in jd["job_title"]
    assert (jd.get("company") or "") != "Minimum Qualifications"


@pytest.mark.parametrize(
    "text, plausible",
    [
        ("Staff Backend Engineer", True),
        ("Senior Software Engineer, Payments", True),
        ("Head of Engineering", True),
        ("Apple is a place where extraordinary people gather to do their best wo", False),
        ("We are looking for a senior engineer to join our team", False),
        ("Do you love working on challenges that no one has solved yet?", False),
        ("Experience with REST APIs and data modelling", False),  # two connectives: prose
        ("", False),
    ],
)
def test_plausible_job_title_rejects_sentences(text, plausible):
    assert local_engine._is_plausible_job_title(text) is plausible


def test_a_real_title_is_still_parsed():
    """The fix must not make the parser blind to ordinary postings."""
    jd = local_engine.analyse_jd(JD)
    assert "Staff Backend Engineer" in jd["job_title"]


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Senior AI Platform Engineer View Jobs", "Senior AI Platform Engineer"),
        ("Staff Backend Engineer  Apply Now", "Staff Backend Engineer"),
        ("Software Engineer - Save Job", "Software Engineer"),
        ("Principal Engineer | Easy Apply", "Principal Engineer"),
        ("Backend Engineer View Jobs Apply Now", "Backend Engineer"),
        ("Senior Engineer", "Senior Engineer"),
    ],
)
def test_job_board_buttons_are_stripped_from_the_title(line, expected):
    """Pasting from a job board drags the page's buttons onto the title line.

    "Senior AI Platform Engineer View Jobs" became the candidate's headline and
    the first words of their professional summary.
    """
    jd = local_engine.analyse_jd(f"{line}\nBengaluru, India\n\nRequirements\n- Java")
    assert jd["job_title"] == expected


def test_bullet_ranking_follows_ontology_edges_in_both_directions():
    """The resume says "Claude API"; the JD says "LLM integration".

    `ontology.EDGES` runs one way (llm -> anthropic api), so scoring a bullet by
    exact lookup gave the candidate's flagship AI work a weight of zero against
    the one job description that most wanted it.
    """
    wanted = {"llm": 4.0, "observability": 4.0}
    assert local_engine._term_weight("anthropic api", wanted) == pytest.approx(3.4)
    assert local_engine._term_weight("datadog", wanted) == pytest.approx(3.6)
    # An unrelated term stays at zero — expansion must not become a free pass.
    assert local_engine._term_weight("kotlin", wanted) == 0.0


def test_ai_evidence_outranks_unrelated_evidence_for_an_ai_role():
    ai_wanted = {"llm": 4.0, "rag": 4.0, "prompt engineering": 4.0}
    claude = ("Architected and solo-developed a 34,000-line production TypeScript "
              "platform on the Claude API")
    warehouse = "Developed warehouse management modules in Java serving 200+ clients"
    assert (local_engine._bullet_relevance(claude, ai_wanted)
            > local_engine._bullet_relevance(warehouse, ai_wanted))


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


# --------------------------------------------------------------------------- #
# Cover letter — same truthfulness guarantee as the resume
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def letter_fixture():
    resume_text = open("samples/master_resume.txt", encoding="utf-8").read()
    profile = local_engine.parse_resume(resume_text)
    jd = local_engine.analyse_jd(
        open("samples/job_description.txt", encoding="utf-8").read()
    )
    matrix = [
        {"canonical": r["canonical"], "priority": r["priority"], "score": 1.0,
         "requirement": r["text"]}
        for r in jd["requirements"]
    ]
    positioning = local_engine.decide_positioning(profile, jd, matrix)
    letter = local_engine.write_cover_letter(
        profile, jd, matrix, positioning, resume_text, today="1 January 2026"
    )
    return profile, jd, letter


def test_cover_letter_has_the_expected_shape(letter_fixture):
    _profile, jd, letter = letter_fixture
    assert letter["salutation"].startswith("Dear")
    assert jd["job_title"] in letter["paragraphs"][0]
    assert len(letter["paragraphs"]) >= 3
    assert letter["signature"]


def test_cover_letter_body_is_verbatim_resume_content(letter_fixture):
    """The core guarantee: no body sentence asserts anything new about the candidate."""
    profile, _jd, letter = letter_fixture
    # Case-insensitive: a paragraph reading "On the leadership side: mentored 4
    # engineers…" lowercases the source bullet's first letter to sit after the
    # colon. That is grammar, not a change of content.
    source = {b.rstrip(".").lower() for role in profile["roles"] for b in role["bullets"]}
    # Skip opening and closing, which are template prose making no factual claim.
    for paragraph in letter["paragraphs"][1:-1]:
        _label, _, evidence = paragraph.partition(": ")
        body = (evidence or paragraph).rstrip(".").lower()
        assert any(body in s or s in body for s in source), (
            f"cover letter paragraph has no source bullet: {paragraph!r}"
        )


def test_cover_letter_invents_no_numbers(letter_fixture):
    profile, _jd, letter = letter_fixture
    master = open("samples/master_resume.txt", encoding="utf-8").read()
    source_digits = set(re.findall(r"\d[\d,.]*", master))
    for paragraph in letter["paragraphs"]:
        for number in re.findall(r"\d[\d,.]*", paragraph):
            # Years-of-experience is derived from the employment dates.
            if number == str(round(profile["total_years_experience"] or 0)):
                continue
            assert number in source_digits, f"invented number {number!r} in cover letter"


def test_cover_letter_contains_no_invented_sentiment(letter_fixture):
    _profile, _jd, letter = letter_fixture
    joined = " ".join(letter["paragraphs"]).lower()
    for phrase in local_engine.BANNED_SENTIMENT:
        assert phrase not in joined


def test_cover_letter_does_not_volunteer_gaps(letter_fixture):
    """Unmet requirements are reported to the candidate, never to the employer."""
    _profile, jd, letter = letter_fixture
    unsupported = local_engine.write_cover_letter(
        {"contact": {"name": "A"}, "roles": [], "domains": [],
         "total_years_experience": 5.0, "current_title": "Engineer"},
        jd,
        [{"canonical": "rust", "priority": "P0", "score": 0.0, "requirement": "Rust"}],
        {"target_title": "Engineer"},
        "master",
    )
    assert "rust" not in " ".join(unsupported["paragraphs"]).lower()
    assert "Rust" in unsupported["omitted_note"]


def test_cover_letter_renders_to_text(letter_fixture):
    _profile, _jd, letter = letter_fixture
    text = local_engine.cover_letter_to_text(letter)
    assert "Dear" in text and "Kind regards," in text
    assert text.endswith("\n")


def test_pdf_layout_puts_the_employer_on_the_date_line():
    """PDF extraction collapses the employer onto the dates and drops indentation.

    The line before a role header is then often the tail of the *previous* role's
    last bullet — which is how "Performer 2023" (from "...recognised as Star
    Performer 2023") became an employer. The residue of the date line is
    structurally bound to this role, so it must outrank the preceding line.
    """
    roles = local_engine._parse_experience(
        [
            "- Mentored 5 junior engineers through code review - recognised as Star",
            "Performer 2023",
            "Oracle February 2022 - September 2022",
            "Software Development Engineer L3 - Bengaluru, India",
            "- Architected backend APIs reducing query response time by 50%.",
        ]
    )
    assert roles[-1]["company"] == "Oracle"
    assert roles[-1]["title"] == "Software Development Engineer L3"


def test_a_sub_team_in_the_title_does_not_become_the_employer():
    """"Senior Engineer, India GCC" names a sub-team, not the company."""
    roles = local_engine._parse_experience(
        [
            "Talendy Holdings - India Global Capability Center",
            "December 2025 - Present",
            "Senior Engineer, India GCC - Bengaluru, India",
            "- Founding engineer for the India GCC.",
        ]
    )
    assert roles[0]["company"] == "Talendy Holdings"
    assert roles[0]["title"] == "Senior Engineer"
