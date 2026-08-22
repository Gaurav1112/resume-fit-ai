"""SQLite persistence: analyses, resume versions, and the application tracker.

Local-first by design. Resume text is sensitive, so it lives in a file on the
user's own machine and nowhere else, and every row is deletable through the API.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    market        TEXT,
    job_title     TEXT,
    company       TEXT,
    resume_text   TEXT NOT NULL,
    jd_text       TEXT NOT NULL,
    payload       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
    id              TEXT PRIMARY KEY,
    analysis_id     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    name            TEXT NOT NULL,
    job_title       TEXT,
    company         TEXT,
    positioning     TEXT,
    ats_score       REAL,
    jd_match_score  REAL,
    recruiter_score REAL,
    status          TEXT,
    payload         TEXT NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS applications (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    company         TEXT,
    job_title       TEXT,
    jd_excerpt      TEXT,
    version_id      TEXT,
    version_name    TEXT,
    positioning     TEXT,
    applied_on      TEXT,
    ats_score       REAL,
    jd_match_score  REAL,
    url             TEXT,
    status          TEXT,
    recruiter       TEXT,
    interview_stage TEXT,
    notes           TEXT,
    result          TEXT
);

CREATE INDEX IF NOT EXISTS idx_versions_analysis ON versions(analysis_id);
CREATE INDEX IF NOT EXISTS idx_apps_positioning ON applications(positioning);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def save_analysis(
    analysis_id: str,
    created_at: str,
    market: str,
    job_title: str,
    company: str,
    resume_text: str,
    jd_text: str,
    payload: dict[str, Any],
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analyses "
            "(id, created_at, market, job_title, company, resume_text, jd_text, payload) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                analysis_id, created_at, market, job_title, company,
                resume_text, jd_text, json.dumps(payload),
            ),
        )


def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["payload"] = json.loads(record["payload"])
    return record


def delete_analysis(analysis_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #
def save_version(
    version_id: str,
    analysis_id: str,
    created_at: str,
    name: str,
    job_title: str,
    company: str,
    positioning: str,
    ats_score: float,
    jd_match_score: float,
    recruiter_score: float,
    status: str,
    payload: dict[str, Any],
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO versions "
            "(id, analysis_id, created_at, name, job_title, company, positioning, "
            " ats_score, jd_match_score, recruiter_score, status, payload) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                version_id, analysis_id, created_at, name, job_title, company,
                positioning, ats_score, jd_match_score, recruiter_score, status,
                json.dumps(payload),
            ),
        )


def list_versions(limit: int = 200) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, analysis_id, created_at, name, job_title, company, positioning, "
            "ats_score, jd_match_score, recruiter_score, status "
            "FROM versions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_version(version_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM versions WHERE id = ?", (version_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    record["payload"] = json.loads(record["payload"])
    return record


def delete_version(version_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM versions WHERE id = ?", (version_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Applications
# --------------------------------------------------------------------------- #
APP_FIELDS = [
    "company", "job_title", "jd_excerpt", "version_id", "version_name", "positioning",
    "applied_on", "ats_score", "jd_match_score", "url", "status", "recruiter",
    "interview_stage", "notes", "result",
]


def save_application(app_id: str, created_at: str, data: dict[str, Any]) -> None:
    values = [data.get(f) for f in APP_FIELDS]
    placeholders = ",".join("?" * (len(APP_FIELDS) + 2))
    with connect() as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO applications (id, created_at, {','.join(APP_FIELDS)}) "
            f"VALUES ({placeholders})",
            [app_id, created_at, *values],
        )


def list_applications() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM applications ORDER BY COALESCE(applied_on, created_at) DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_application(app_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM applications WHERE id = ?", (app_id,))
    return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Learning loop
# --------------------------------------------------------------------------- #
INTERVIEW_STATES = {"interview", "screen", "phone screen", "onsite", "final", "offer"}
OFFER_STATES = {"offer", "accepted", "hired"}


def positioning_performance() -> list[dict[str, Any]]:
    """Aggregate outcomes by positioning so you can see what is actually working.

    Deliberately reports raw counts alongside rates: a 100% interview rate from
    two applications is not a signal, and the UI should be able to say so.
    """
    rows = list_applications()
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (row.get("positioning") or "Unspecified").strip() or "Unspecified"
        bucket = buckets.setdefault(
            key,
            {
                "positioning": key, "applications": 0, "interviews": 0, "offers": 0,
                "rejected": 0, "avg_ats": 0.0, "avg_match": 0.0, "_ats": [], "_match": [],
            },
        )
        bucket["applications"] += 1
        status = (row.get("status") or "").lower()
        stage = (row.get("interview_stage") or "").lower()
        result = (row.get("result") or "").lower()
        if status in INTERVIEW_STATES or stage or status in OFFER_STATES:
            bucket["interviews"] += 1
        if status in OFFER_STATES or result in OFFER_STATES:
            bucket["offers"] += 1
        if "reject" in status or "reject" in result:
            bucket["rejected"] += 1
        if row.get("ats_score"):
            bucket["_ats"].append(row["ats_score"])
        if row.get("jd_match_score"):
            bucket["_match"].append(row["jd_match_score"])

    out: list[dict[str, Any]] = []
    for bucket in buckets.values():
        apps = bucket["applications"]
        ats = bucket.pop("_ats")
        match = bucket.pop("_match")
        bucket["avg_ats"] = round(sum(ats) / len(ats), 1) if ats else 0.0
        bucket["avg_match"] = round(sum(match) / len(match), 1) if match else 0.0
        bucket["interview_rate"] = round(100.0 * bucket["interviews"] / apps, 1) if apps else 0.0
        bucket["offer_rate"] = round(100.0 * bucket["offers"] / apps, 1) if apps else 0.0
        # Fewer than 8 applications is noise, not evidence.
        bucket["significant"] = apps >= 8
        out.append(bucket)

    out.sort(key=lambda b: (b["significant"], b["interview_rate"], b["applications"]), reverse=True)
    return out


def purge_all() -> None:
    with connect() as conn:
        conn.executescript(
            "DELETE FROM versions; DELETE FROM analyses; DELETE FROM applications;"
        )
