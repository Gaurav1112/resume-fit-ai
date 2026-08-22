"""FastAPI application.

The browser never sees an API key. The flow is always:

    frontend  →  this backend  →  LLM provider

Analyses are held in an in-memory session cache for speed and persisted to
SQLite for durability; if the process restarts, `/generate` rehydrates the
pipeline context from the stored analysis without re-running any LLM stage.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import db
from .config import ROOT, settings
from .graph import Context, GraphError
from .llm import LLMError, LLMRefusal, get_provider
from .models.schemas import AnalysisResult, GenerationResult
from .services import exporters, matching, pipeline, text_extract
from .services.text_extract import ExtractionError

@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init()
    yield


app = FastAPI(
    title="ResumeFit AI",
    description="JD-to-Resume Optimization Engine",
    version="1.0.0",
    lifespan=lifespan,
)

FRONTEND = ROOT / "frontend"

# analysis_id -> pipeline Context. Bounded so a long session can't grow forever.
_SESSIONS: dict[str, Context] = {}
_SESSION_LIMIT = 24


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _remember(analysis_id: str, ctx: Context) -> None:
    _SESSIONS[analysis_id] = ctx
    while len(_SESSIONS) > _SESSION_LIMIT:
        _SESSIONS.pop(next(iter(_SESSIONS)))


# --------------------------------------------------------------------------- #
# Error handling — surface actionable messages, never stack traces
# --------------------------------------------------------------------------- #
@app.exception_handler(LLMRefusal)
def _refusal_handler(_request, exc: LLMRefusal):
    return JSONResponse(
        status_code=422,
        content={
            "error": "model_refusal",
            "message": str(exc),
            "hint": "This is unusual for resume content. Check the JD text for anything "
                    "that could read as a prohibited topic, or retry.",
        },
    )


@app.exception_handler(LLMError)
def _llm_handler(_request, exc: LLMError):
    return JSONResponse(
        status_code=502, content={"error": "llm_error", "message": str(exc)}
    )


@app.exception_handler(GraphError)
def _graph_handler(_request, exc: GraphError):
    return JSONResponse(
        status_code=502,
        content={
            "error": "pipeline_error",
            "message": str(exc),
            "stage": exc.node,
        },
    )


@app.exception_handler(ExtractionError)
def _extract_handler(_request, exc: ExtractionError):
    return JSONResponse(
        status_code=400, content={"error": "extraction_error", "message": str(exc)}
    )


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": settings.provider,
        "model": settings.model,
        "effort": settings.effort,
        "configured": settings.configured,
        "supported_uploads": sorted(text_extract.SUPPORTED),
    }


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    from .services.scoring import DEFAULT_WEIGHTS, LABELS

    return {
        "provider": settings.provider,
        "model": settings.model,
        "weights": DEFAULT_WEIGHTS,
        "weight_labels": LABELS,
        "markets": ["India", "USA", "Canada", "Europe", "UK", "Middle East", "Global Remote"],
    }


# --------------------------------------------------------------------------- #
# Text intake
# --------------------------------------------------------------------------- #
async def _read_upload(upload: UploadFile | None) -> str:
    if upload is None or not upload.filename:
        return ""
    data = await upload.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"'{upload.filename}' exceeds the "
                   f"{settings.max_upload_bytes // (1024 * 1024)} MB limit.",
        )
    return text_extract.extract(upload.filename, data)


@app.post("/api/extract")
async def extract_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extract text from an upload so the UI can preview it before analysing."""
    text = await _read_upload(file)
    return {"filename": file.filename, "characters": len(text), "text": text}


# --------------------------------------------------------------------------- #
# Analyse
# --------------------------------------------------------------------------- #
@app.post("/api/analyze")
async def analyze(
    resume_file: UploadFile | None = File(None),
    resume_text: str = Form(""),
    jd_file: UploadFile | None = File(None),
    jd_text: str = Form(""),
    target_market: str = Form("Global Remote"),
    weights: str = Form(""),
) -> dict[str, Any]:
    resume_content = (await _read_upload(resume_file)) or resume_text.strip()
    jd_content = (await _read_upload(jd_file)) or jd_text.strip()

    if len(resume_content) < 200:
        raise HTTPException(
            status_code=400,
            detail="The master resume is empty or too short to analyse. Upload a "
                   "text-based PDF/DOCX or paste the full text.",
        )
    if len(jd_content) < 80:
        raise HTTPException(
            status_code=400,
            detail="The job description is empty or too short to analyse.",
        )

    parsed_weights: dict[str, float] = {}
    if weights.strip():
        import json

        try:
            parsed_weights = {k: float(v) for k, v in json.loads(weights).items()}
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid weights: {exc}") from exc

    result, ctx = pipeline.analyse(
        resume_content, jd_content, market=target_market, weights=parsed_weights
    )
    _remember(result.analysis_id, ctx)

    db.save_analysis(
        analysis_id=result.analysis_id,
        created_at=result.created_at,
        market=target_market,
        job_title=result.jd.job_title,
        company=result.jd.company,
        resume_text=resume_content,
        jd_text=jd_content,
        payload=result.model_dump(),
    )

    provider = ctx["provider"]
    return {
        **result.model_dump(),
        "jd_match_score": _jd_match(result),
        "trace": ctx.trace_dicts(),
        "usage": provider.usage.to_dict(),
    }


def _jd_match(result: AnalysisResult) -> float:
    from .services.scoring import jd_match_score

    return jd_match_score(result.matrix)


# --------------------------------------------------------------------------- #
# Generate
# --------------------------------------------------------------------------- #
def _rehydrate(analysis_id: str) -> Context:
    """Rebuild a pipeline context from persisted state — no LLM calls."""
    cached = _SESSIONS.get(analysis_id)
    if cached is not None:
        return cached

    record = db.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis '{analysis_id}' not found. Run the analysis again.",
        )

    result = AnalysisResult.model_validate(record["payload"])
    ctx = Context()
    ctx.set("resume_text", record["resume_text"])
    ctx.set("jd_text", record["jd_text"])
    ctx.set("market", record["market"])
    ctx.set("weights", {})
    ctx.set("provider", get_provider())
    ctx.set("analysis_id", analysis_id)
    ctx.set("profile", result.profile)
    ctx.set("jd", result.jd)
    ctx.set("matrix", result.matrix)
    ctx.set("matrix_final", result.matrix)
    ctx.set("gaps", result.gaps)
    ctx.set("positioning", result.positioning)
    ctx.set("baseline_scores", result.baseline_scores)
    ctx.set(
        "evidence_index",
        matching.build_evidence_index(result.profile, record["resume_text"]),
    )
    _remember(analysis_id, ctx)
    return ctx


@app.post("/api/generate")
async def generate(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    analysis_id = payload.get("analysis_id", "")
    if not analysis_id:
        raise HTTPException(status_code=400, detail="analysis_id is required.")

    ctx = _rehydrate(analysis_id)
    # Force a fresh document even when regenerating within the same session.
    for key in ("resume", "truth_audit", "recruiter"):
        ctx.values.pop(key, None)

    result = pipeline.generate(
        ctx,
        max_repair_iterations=int(payload.get("max_repair_iterations", 3)),
        lift_rounds=int(payload.get("lift_rounds", 1)),
    )

    db.save_version(
        version_id=result.version_id,
        analysis_id=analysis_id,
        created_at=result.created_at,
        name=result.version_name,
        job_title=ctx["jd"].job_title,
        company=ctx["jd"].company,
        positioning=ctx["positioning"].target_title,
        ats_score=result.ats_report.score(),
        jd_match_score=_component(result, "semantic_alignment"),
        recruiter_score=result.recruiter.score,
        status=result.status,
        payload=result.model_dump(),
    )

    return {
        **result.model_dump(),
        "trace": ctx.trace_dicts(),
        "warnings": ctx.warnings,
        "usage": ctx["provider"].usage.to_dict(),
    }


def _component(result: GenerationResult, key: str) -> float:
    for component in result.scores.components:
        if component.key == key:
            return component.raw
    return 0.0


# --------------------------------------------------------------------------- #
# Versions + export
# --------------------------------------------------------------------------- #
@app.get("/api/versions")
def versions() -> list[dict[str, Any]]:
    return db.list_versions()


@app.get("/api/versions/{version_id}")
def version_detail(version_id: str) -> dict[str, Any]:
    record = db.get_version(version_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Version not found.")
    return record["payload"]


@app.delete("/api/versions/{version_id}")
def version_delete(version_id: str) -> dict[str, bool]:
    return {"deleted": db.delete_version(version_id)}


@app.get("/api/export/{version_id}.{fmt}")
def export(version_id: str, fmt: str) -> Response:
    fmt = fmt.lower()
    if fmt not in exporters.EXPORTERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Use one of: "
                   f"{', '.join(exporters.EXPORTERS)}",
        )
    record = db.get_version(version_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Version not found.")

    result = GenerationResult.model_validate(record["payload"])
    render, media_type, extension = exporters.EXPORTERS[fmt]
    data = render(result.resume)

    name = result.resume.contact.name or "resume"
    filename = exporters.safe_filename(f"{name}_{result.version_name}") + f".{extension}"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------------- #
# Application tracker + learning loop
# --------------------------------------------------------------------------- #
@app.get("/api/applications")
def applications() -> list[dict[str, Any]]:
    return db.list_applications()


@app.post("/api/applications")
def application_create(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    app_id = payload.get("id") or f"app_{uuid.uuid4().hex[:10]}"
    db.save_application(app_id, _now(), payload)
    return {"id": app_id, "saved": True}


@app.delete("/api/applications/{app_id}")
def application_delete(app_id: str) -> dict[str, bool]:
    return {"deleted": db.delete_application(app_id)}


@app.get("/api/analytics/positioning")
def positioning_analytics() -> dict[str, Any]:
    rows = db.positioning_performance()
    significant = [r for r in rows if r["significant"]]
    best = significant[0]["positioning"] if significant else None
    return {
        "rows": rows,
        "best_positioning": best,
        "note": (
            "Positionings with fewer than 8 applications are shown but flagged as "
            "not yet significant — a 100% rate from two applications is noise."
        ),
    }


# --------------------------------------------------------------------------- #
# Data control
# --------------------------------------------------------------------------- #
@app.delete("/api/data")
def purge() -> dict[str, str]:
    db.purge_all()
    _SESSIONS.clear()
    return {"status": "all stored resumes, analyses and applications deleted"}


# --------------------------------------------------------------------------- #
# Static frontend (mounted last so /api/* wins)
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
