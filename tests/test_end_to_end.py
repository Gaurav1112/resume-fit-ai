"""End-to-end pipeline + HTTP API, using the offline `mock` provider.

This proves the whole deterministic half works together — graph execution, the
evidence index, matching, the repair loop, both validators, scoring, rendering,
export and persistence — with no network and no API key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("LLM_PROVIDER", "mock")

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="module")
def texts() -> tuple[str, str]:
    return (
        (SAMPLES / "master_resume.txt").read_text(encoding="utf-8"),
        (SAMPLES / "job_description.txt").read_text(encoding="utf-8"),
    )


@pytest.fixture(scope="module")
def provider():
    from backend.llm import get_provider

    return get_provider("mock", "mock-1")


def test_pipeline_runs_end_to_end(texts, provider):
    from backend.services import pipeline

    resume_text, jd_text = texts
    analysis, ctx = pipeline.analyse(
        resume_text, jd_text, market="Global Remote", provider=provider
    )

    assert analysis.analysis_id.startswith("an_")
    assert analysis.jd.requirements, "JD analysis produced no requirements"
    assert analysis.matrix, "matrix is empty"
    assert len(analysis.matrix) == len(analysis.jd.requirements)
    assert 0 <= analysis.baseline_scores.total <= 100
    assert analysis.baseline_scores.band

    # Every component must carry an explanation — the score is never a bare number.
    for component in analysis.baseline_scores.components:
        assert component.explanation

    result = pipeline.generate(ctx, max_repair_iterations=1, lift_rounds=0)
    assert result.version_id.startswith("v_")
    assert result.plain_text.strip()
    assert result.scores.components
    assert result.ats_report.checks
    assert result.truth_report.checks
    assert result.status in ("optimized", "needs_review")
    assert "loop" in result.diff and result.diff["loop"]["attempts"]


def test_graph_runs_independent_stages_in_parallel():
    from backend.services.pipeline import build_graph

    plan = build_graph().plan()
    assert set(plan[0]) == {"jd", "profile"}, "resume parse and JD analysis must be concurrent"


def test_generate_reuses_the_analysis_without_re_parsing(texts, provider):
    """The second phase must not re-run the expensive parse stages."""
    from backend.services import pipeline

    resume_text, jd_text = texts
    _analysis, ctx = pipeline.analyse(resume_text, jd_text, provider=provider)
    pipeline.generate(ctx, max_repair_iterations=1, lift_rounds=0)

    statuses = {t.name: t.status for t in ctx.trace}
    assert statuses.get("profile") == "cached"
    assert statuses.get("jd") == "cached"
    assert statuses.get("resume") == "ok"


def test_exports_produce_real_files(texts, provider):
    import io
    import zipfile

    from pypdf import PdfReader

    from backend.services import exporters, pipeline

    resume_text, jd_text = texts
    _analysis, ctx = pipeline.analyse(resume_text, jd_text, provider=provider)
    result = pipeline.generate(ctx, max_repair_iterations=1, lift_rounds=0)

    txt = exporters.to_txt(result.resume)
    assert b"PROFESSIONAL EXPERIENCE" in txt

    docx = exporters.to_docx(result.resume)
    with zipfile.ZipFile(io.BytesIO(docx)) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    assert "<w:tbl>" not in xml

    pdf = exporters.to_pdf(result.resume)
    assert pdf[:5] == b"%PDF-"
    assert PdfReader(io.BytesIO(pdf)).pages


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["DB_PATH"] = str(tmp_path_factory.mktemp("db") / "test.db")
    from fastapi.testclient import TestClient

    from backend import db, main

    db.init()
    return TestClient(main.app)


def test_health_endpoint(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "mock"


def test_analyze_rejects_a_too_short_resume(client):
    response = client.post(
        "/api/analyze", data={"resume_text": "hi", "jd_text": "x" * 200}
    )
    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower()


def test_full_http_flow(client, texts):
    resume_text, jd_text = texts

    analysis = client.post(
        "/api/analyze",
        data={"resume_text": resume_text, "jd_text": jd_text, "target_market": "USA"},
    )
    assert analysis.status_code == 200, analysis.text
    a = analysis.json()
    assert a["matrix"] and a["trace"]

    generation = client.post(
        "/api/generate",
        json={"analysis_id": a["analysis_id"], "max_repair_iterations": 1, "lift_rounds": 0},
    )
    assert generation.status_code == 200, generation.text
    g = generation.json()

    versions = client.get("/api/versions").json()
    assert any(v["id"] == g["version_id"] for v in versions)

    for fmt, magic in (("txt", b""), ("docx", b"PK"), ("pdf", b"%PDF-")):
        export = client.get(f"/api/export/{g['version_id']}.{fmt}")
        assert export.status_code == 200
        assert export.content.startswith(magic)
        assert "attachment" in export.headers["content-disposition"]

    assert client.get(f"/api/export/{g['version_id']}.rtf").status_code == 400
    assert client.get("/api/export/nope.pdf").status_code == 404


def test_generate_with_unknown_analysis_is_404(client):
    assert client.post("/api/generate", json={"analysis_id": "an_missing"}).status_code == 404


def test_tracker_and_learning_loop(client):
    created = client.post(
        "/api/applications",
        json={
            "company": "Helios", "job_title": "Staff Backend Engineer",
            "positioning": "Staff Backend Engineer", "status": "interview",
            "ats_score": 94.0, "jd_match_score": 88.0,
        },
    ).json()
    assert created["saved"]

    analytics = client.get("/api/analytics/positioning").json()
    row = next(r for r in analytics["rows"] if r["positioning"] == "Staff Backend Engineer")
    assert row["applications"] == 1
    assert row["interviews"] == 1
    # One application is not a signal, and the API must say so.
    assert row["significant"] is False

    assert client.delete(f"/api/applications/{created['id']}").json()["deleted"]


def test_purge_removes_everything(client):
    assert client.delete("/api/data").status_code == 200
    assert client.get("/api/versions").json() == []
    assert client.get("/api/applications").json() == []
