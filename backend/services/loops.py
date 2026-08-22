"""Convergence loops: generate → validate → repair → re-validate.

A single-shot generation cannot *guarantee* the truth and ATS gates pass. The
writer might surface a metric that isn't in the master resume, or repeat a
keyword often enough to look like stuffing. Prompting harder does not fix this
reliably; a feedback loop does.

Two loops live here:

  RepairLoop  — runs the candidate through deterministic validators, and on a
                critical failure feeds the *exact offending strings* back to the
                producer and regenerates. Stops on: all-critical-pass, score
                plateau, or iteration cap — and reports which one fired.

  lift_loop   — an optional keyword-lift pass: while there are requirements that
                are supported by evidence but absent from the document, ask for
                one more targeted revision. Stops when a round surfaces nothing
                new ("loop until dry"), so it can't spin forever.

Both loops are pure orchestration — the validators they call are deterministic,
so the loop's stopping condition is checkable rather than vibes-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from ..models.schemas import ValidationCheck, ValidationReport

T = TypeVar("T")

Producer = Callable[[list[str], int], T]          # (feedback, iteration) -> candidate
Validator = Callable[[T], ValidationReport]
Scorer = Callable[[T], float]


@dataclass
class LoopAttempt:
    iteration: int
    score: float
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    feedback_sent: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "score": round(self.score, 1),
            "critical_failures": self.critical_failures,
            "warnings": self.warnings[:5],
            "feedback_sent": self.feedback_sent,
        }


@dataclass
class LoopResult(Generic[T]):
    value: T
    report: ValidationReport
    score: float
    converged: bool
    stop_reason: str
    attempts: list[LoopAttempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "stop_reason": self.stop_reason,
            "iterations": len(self.attempts),
            "attempts": [a.to_dict() for a in self.attempts],
        }


def _feedback_from(report: ValidationReport, limit: int = 12) -> list[str]:
    """Turn failed checks into instructions the writer can act on."""
    lines: list[str] = []
    for check in report.checks:
        if check.passed:
            continue
        prefix = "MUST FIX" if check.severity == "critical" else "SHOULD FIX"
        detail = check.detail or check.label
        line = f"[{prefix}] {check.label}: {detail}"
        if check.offenders:
            line += " | offending items: " + "; ".join(check.offenders[:6])
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


class RepairLoop(Generic[T]):
    """Iterate a producer against a validator until the critical gates pass."""

    def __init__(
        self,
        produce: Producer,
        validate: Validator,
        score: Scorer | None = None,
        *,
        max_iterations: int = 3,
        min_gain: float = 1.0,
    ) -> None:
        self.produce = produce
        self.validate = validate
        self.score = score or (lambda _c: 0.0)
        self.max_iterations = max(1, max_iterations)
        self.min_gain = min_gain

    def run(self) -> LoopResult[T]:
        attempts: list[LoopAttempt] = []
        feedback: list[str] = []

        best_value: T | None = None
        best_report: ValidationReport | None = None
        best_score = float("-inf")
        stop_reason = "iteration_cap"
        converged = False

        for i in range(1, self.max_iterations + 1):
            candidate = self.produce(feedback, i)
            report = self.validate(candidate)
            score = self.score(candidate)

            criticals = [c.label for c in report.checks if not c.passed and c.severity == "critical"]
            warns = [c.label for c in report.checks if not c.passed and c.severity == "warning"]
            attempts.append(
                LoopAttempt(
                    iteration=i,
                    score=score,
                    critical_failures=criticals,
                    warnings=warns,
                    feedback_sent=list(feedback),
                )
            )

            # Prefer a candidate that clears the critical gate; among those (or
            # among all failures) prefer the higher score.
            candidate_rank = (report.passed, score)
            best_rank = (best_report.passed if best_report else False, best_score)
            if best_value is None or candidate_rank > best_rank:
                best_value, best_report, best_score = candidate, report, score

            if report.passed:
                stop_reason = "all_critical_checks_passed"
                converged = True
                break

            if i >= 2 and score - attempts[-2].score < self.min_gain:
                stop_reason = "score_plateau"
                break

            feedback = _feedback_from(report)
            if not feedback:
                stop_reason = "no_actionable_feedback"
                break

        assert best_value is not None and best_report is not None
        return LoopResult(
            value=best_value,
            report=best_report,
            score=best_score if best_score != float("-inf") else 0.0,
            converged=converged,
            stop_reason=stop_reason,
            attempts=attempts,
        )


def lift_loop(
    produce: Callable[[list[str], int], T],
    find_missing: Callable[[T], list[str]],
    *,
    max_rounds: int = 2,
) -> tuple[T, list[dict]]:
    """Loop-until-dry: keep surfacing supported-but-absent keywords.

    Stops as soon as a round finds nothing new, so it converges rather than
    running a fixed number of expensive rounds.
    """
    rounds: list[dict] = []
    seen: set[str] = set()
    candidate = produce([], 0)

    for r in range(1, max_rounds + 1):
        missing = [m for m in find_missing(candidate) if m not in seen]
        if not missing:
            rounds.append({"round": r, "missing": [], "action": "dry — stopped"})
            break
        seen.update(missing)
        instruction = [
            "These requirements ARE supported by the candidate's evidence but do not "
            "appear anywhere in the generated resume. Surface each one naturally in the "
            "most appropriate section — only where the evidence genuinely supports it. "
            "Do not add a keyword to a bullet it does not belong in: "
            + ", ".join(missing[:12])
        ]
        rounds.append({"round": r, "missing": missing[:12], "action": "revision requested"})
        candidate = produce(instruction, r)

    return candidate, rounds


def merge_reports(*reports: ValidationReport) -> ValidationReport:
    checks: list[ValidationCheck] = []
    for report in reports:
        checks.extend(report.checks)
    return ValidationReport(checks=checks)
