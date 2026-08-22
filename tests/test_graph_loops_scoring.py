"""Graph engine, convergence loops, scoring engine, and exporters."""

from __future__ import annotations

import pytest

from backend.graph import Context, Graph, GraphError
from backend.models.schemas import (
    CandidateProfile,
    Contact,
    JDAnalysis,
    MatchRow,
    Requirement,
    ResumeBullet,
    ResumeRole,
    ResumeSection,
    Role,
    TailoredResume,
    ValidationCheck,
    ValidationReport,
)
from backend.services import exporters, scoring
from backend.services.loops import RepairLoop, lift_loop
from backend.services.render import to_plain_text


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def test_independent_nodes_share_a_level():
    g = Graph()
    g.add("a", lambda c: 1)
    g.add("b", lambda c: 2)
    g.add("c", lambda c: c["a"] + c["b"], deps=["a", "b"])
    plan = g.plan()
    assert set(plan[0]) == {"a", "b"}
    assert plan[1] == ["c"]


def test_graph_executes_and_stores_results():
    g = Graph()
    g.add("a", lambda c: 2)
    g.add("b", lambda c: c["a"] * 3, deps=["a"])
    ctx = g.run()
    assert ctx["b"] == 6


def test_cycle_is_rejected():
    g = Graph()
    g.add("a", lambda c: 1, deps=["b"])
    g.add("b", lambda c: 2, deps=["a"])
    with pytest.raises(ValueError, match="cycle"):
        g.plan()


def test_unknown_dependency_is_rejected():
    g = Graph()
    g.add("a", lambda c: 1, deps=["nope"])
    with pytest.raises(ValueError, match="unknown node"):
        g.plan()


def test_cached_nodes_are_not_re_executed():
    calls = {"n": 0}

    def fn(_ctx):
        calls["n"] += 1
        return 1

    g = Graph()
    g.add("a", fn)
    ctx = Context()
    g.run(ctx)
    g.run(ctx)
    assert calls["n"] == 1
    assert [t.status for t in ctx.trace] == ["ok", "cached"]


def test_only_runs_the_requested_closure():
    calls = []
    g = Graph()
    g.add("a", lambda c: calls.append("a") or 1)
    g.add("b", lambda c: calls.append("b") or 2)
    g.add("c", lambda c: calls.append("c") or 3, deps=["a"])
    g.run(Context(), only={"c"})
    assert "b" not in calls
    assert set(calls) == {"a", "c"}


def test_required_node_failure_aborts_with_context():
    def boom(_ctx):
        raise RuntimeError("kaboom")

    g = Graph()
    g.add("bad", boom, retries=1)
    with pytest.raises(GraphError) as exc:
        g.run()
    assert exc.value.node == "bad"


def test_optional_node_failure_warns_and_continues():
    g = Graph()
    g.add("opt", lambda c: (_ for _ in ()).throw(RuntimeError("x")), optional=True, retries=1)
    g.add("after", lambda c: "done")
    ctx = g.run()
    assert ctx["after"] == "done"
    assert any("Optional stage" in w for w in ctx.warnings)


def test_retry_succeeds_on_second_attempt():
    attempts = {"n": 0}

    def flaky(_ctx):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    g = Graph()
    g.add("f", flaky, retries=3)
    assert g.run()["f"] == "ok"
    assert attempts["n"] == 2


# --------------------------------------------------------------------------- #
# Loops
# --------------------------------------------------------------------------- #
def _report(critical_failures: int, warnings: int = 0) -> ValidationReport:
    checks = [
        ValidationCheck(id=f"c{i}", label=f"critical {i}", passed=False, severity="critical",
                        detail="bad", offenders=["x"])
        for i in range(critical_failures)
    ] + [
        ValidationCheck(id=f"w{i}", label=f"warning {i}", passed=False, severity="warning",
                        detail="meh")
        for i in range(warnings)
    ] + [ValidationCheck(id="ok", label="fine", passed=True, severity="info")]
    return ValidationReport(checks=checks)


def test_repair_loop_stops_immediately_when_clean():
    calls = []
    loop = RepairLoop(
        produce=lambda fb, i: calls.append(i) or "doc",
        validate=lambda _c: _report(0),
        score=lambda _c: 90.0,
    )
    result = loop.run()
    assert result.converged
    assert result.stop_reason == "all_critical_checks_passed"
    assert calls == [1]


def test_repair_loop_feeds_failures_back_and_converges():
    seen_feedback = []

    def produce(feedback, iteration):
        seen_feedback.append(list(feedback))
        return f"doc{iteration}"

    def validate(candidate):
        return _report(0) if candidate == "doc2" else _report(2)

    loop = RepairLoop(produce, validate, score=lambda c: 70.0 if c == "doc1" else 88.0)
    result = loop.run()
    assert result.converged
    assert result.value == "doc2"
    assert seen_feedback[0] == []           # first attempt gets no feedback
    assert seen_feedback[1]                  # second attempt receives the failures
    assert "MUST FIX" in seen_feedback[1][0]


def test_repair_loop_respects_iteration_cap():
    loop = RepairLoop(
        produce=lambda fb, i: f"doc{i}",
        validate=lambda _c: _report(1),
        score=lambda _c: 50.0,
        max_iterations=3,
        min_gain=-999,          # disable plateau stop so the cap is what fires
    )
    result = loop.run()
    assert not result.converged
    assert result.stop_reason == "iteration_cap"
    assert len(result.attempts) == 3


def test_repair_loop_stops_on_score_plateau():
    loop = RepairLoop(
        produce=lambda fb, i: f"doc{i}",
        validate=lambda _c: _report(1),
        score=lambda _c: 60.0,
        max_iterations=5,
        min_gain=1.0,
    )
    result = loop.run()
    assert result.stop_reason == "score_plateau"
    assert len(result.attempts) == 2


def test_repair_loop_returns_best_candidate_not_last():
    scores = {"doc1": 95.0, "doc2": 40.0, "doc3": 41.0}
    loop = RepairLoop(
        produce=lambda fb, i: f"doc{i}",
        validate=lambda _c: _report(1),
        score=lambda c: scores[c],
        max_iterations=3,
        min_gain=-999,
    )
    assert loop.run().value == "doc1"


def test_lift_loop_stops_when_dry():
    rounds_run = []

    def produce(instruction, r):
        rounds_run.append(r)
        return "doc"

    _value, rounds = lift_loop(produce, lambda _c: [], max_rounds=3)
    assert rounds[-1]["action"].startswith("dry")
    assert rounds_run == [0]          # only the initial production


def test_lift_loop_does_not_repeat_the_same_missing_keyword():
    calls = []

    def produce(instruction, r):
        calls.append(instruction)
        return "doc"

    _value, rounds = lift_loop(produce, lambda _c: ["Kafka"], max_rounds=3)
    # "Kafka" is requested once, then deduplicated, so round 2 goes dry.
    assert len(rounds) == 2
    assert rounds[1]["missing"] == []


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _matrix(scores_by_priority: list[tuple[str, float]]) -> list[MatchRow]:
    return [
        MatchRow(
            requirement_id=f"R{i}",
            requirement=f"req{i}",
            canonical=f"req{i}",
            priority=p,  # type: ignore[arg-type]
            score=s,
            match_type="EXACT" if s >= 0.999 else "PARTIAL",
        )
        for i, (p, s) in enumerate(scores_by_priority)
    ]


def test_score_is_bounded_and_banded():
    inputs = scoring.ScoringInputs(
        jd=JDAnalysis(job_title="Backend Engineer"),
        profile=CandidateProfile(current_title="Backend Engineer"),
        matrix=_matrix([("P0", 1.0), ("P0", 1.0), ("P1", 1.0)]),
    )
    report = scoring.compute(inputs)
    assert 0 <= report.total <= 100
    assert report.band in {
        "Excellent", "Strong", "Good", "Needs improvement", "Poor alignment",
    }


def test_every_component_explains_itself():
    report = scoring.compute(
        scoring.ScoringInputs(
            jd=JDAnalysis(), profile=CandidateProfile(), matrix=_matrix([("P0", 0.5)])
        )
    )
    assert len(report.components) == len(scoring.DEFAULT_WEIGHTS)
    for component in report.components:
        assert component.explanation
        assert 0 <= component.raw <= 100


def test_optional_requirements_do_not_sink_required_coverage():
    full_p0 = scoring.ScoringInputs(
        jd=JDAnalysis(), profile=CandidateProfile(),
        matrix=_matrix([("P0", 1.0), ("P3", 0.0), ("P3", 0.0), ("P3", 0.0)]),
    )
    component = next(
        c for c in scoring.compute(full_p0).components if c.key == "required_skills"
    )
    assert component.raw == 100.0


def test_missing_mandatory_requirement_hurts():
    strong = scoring.compute(
        scoring.ScoringInputs(jd=JDAnalysis(), profile=CandidateProfile(),
                              matrix=_matrix([("P0", 1.0), ("P0", 1.0)]))
    )
    weak = scoring.compute(
        scoring.ScoringInputs(jd=JDAnalysis(), profile=CandidateProfile(),
                              matrix=_matrix([("P0", 1.0), ("P0", 0.0)]))
    )
    assert weak.total < strong.total


def test_weights_are_configurable_and_renormalised():
    inputs = scoring.ScoringInputs(
        jd=JDAnalysis(), profile=CandidateProfile(), matrix=_matrix([("P0", 1.0)])
    )
    report = scoring.compute(inputs, weights={"keyword_coverage": 0.9})
    assert sum(c.weight for c in report.components) == pytest.approx(1.0, abs=1e-6)
    kw = next(c for c in report.components if c.key == "keyword_coverage")
    assert kw.weight > scoring.DEFAULT_WEIGHTS["keyword_coverage"]


def test_scoring_is_deterministic():
    inputs = scoring.ScoringInputs(
        jd=JDAnalysis(job_title="Backend Engineer"),
        profile=CandidateProfile(current_title="Backend Engineer", total_years_experience=8),
        matrix=_matrix([("P0", 1.0), ("P1", 0.7)]),
    )
    assert scoring.compute(inputs).total == scoring.compute(inputs).total


# --------------------------------------------------------------------------- #
# Exporters
# --------------------------------------------------------------------------- #
@pytest.fixture
def resume() -> TailoredResume:
    return TailoredResume(
        contact=Contact(name="Arjun Mehta", email="a@example.com", location="Bengaluru"),
        headline="Senior Backend Engineer",
        sections=[
            ResumeSection(heading="Professional Summary", kind="summary",
                          paragraphs=["Backend engineer with 8 years of experience."]),
            ResumeSection(heading="Core Skills", kind="skills",
                          skill_groups={"Languages": ["Java", "Go"]}),
            ResumeSection(
                heading="Professional Experience", kind="experience",
                roles=[ResumeRole(
                    company="Northwind", title="Senior Backend Engineer",
                    start_date="Mar 2021", end_date="Present",
                    bullets=[ResumeBullet(text="Designed a Kafka event pipeline.")],
                )],
            ),
        ],
    )


def test_plain_text_contains_every_section(resume):
    text = to_plain_text(resume)
    for expected in ("Arjun Mehta", "PROFESSIONAL SUMMARY", "CORE SKILLS",
                     "PROFESSIONAL EXPERIENCE", "Kafka"):
        assert expected in text


def test_docx_is_a_real_docx_with_extractable_text(resume):
    import io
    import zipfile

    data = exporters.to_docx(resume)
    assert data[:2] == b"PK"                       # zip magic == real OOXML
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    assert "Arjun Mehta" in xml
    assert "Kafka" in xml
    # ATS-hostile constructs must not be present.
    assert "<w:tbl>" not in xml                    # no tables
    assert "<w:drawing>" not in xml                # no images
    assert "<w:txbxContent>" not in xml            # no text boxes


def test_pdf_has_a_real_text_layer(resume):
    import io

    from pypdf import PdfReader

    data = exporters.to_pdf(resume)
    assert data[:5] == b"%PDF-"
    text = "".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    assert "Arjun Mehta" in text
    assert "Kafka" in text


def test_filename_is_sanitised():
    assert exporters.safe_filename("Arjun Mehta / Resume: v2") == "Arjun_Mehta_Resume_v2"
