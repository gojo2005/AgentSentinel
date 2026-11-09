"""Unit tests for deterministic rule-based run diagnostics."""

from typing import Literal
from uuid import UUID, uuid4

import pytest

from app.diagnostics.diagnostic_engine import RuleBasedDiagnosticEngine
from app.diagnostics.schemas import DiagnosticFinding
from app.evaluation.schemas import EvaluationResult
from app.monitoring.schemas import ExecutionEvent


def make_event(
    run_id: UUID,
    *,
    event_type: Literal[
        "run_started",
        "llm_call",
        "tool_call",
        "run_completed",
        "run_failed",
    ] = "run_started",
    status: Literal["started", "success", "failure"] = "success",
    duration_ms: float | None = None,
    **overrides: object,
) -> ExecutionEvent:
    """Create a valid execution event, allowing individual fields to vary."""
    values: dict[str, object] = {
        "run_id": run_id,
        "event_type": event_type,
        "agent_name": "test-agent",
        "status": status,
        "duration_ms": duration_ms,
    }
    values.update(overrides)
    return ExecutionEvent(**values)


def make_evaluation(
    run_id: UUID,
    **overrides: object,
) -> EvaluationResult:
    """Create a valid evaluation result, allowing scores to be overridden."""
    values: dict[str, object] = {
        "run_id": run_id,
        "evaluator": "rule_based",
    }
    values.update(overrides)
    return EvaluationResult(**values)


def find_finding(
    findings: list[DiagnosticFinding],
    category: str,
) -> DiagnosticFinding:
    """Return a finding by category, failing clearly when it is absent."""
    return next(finding for finding in findings if finding.category == category)


def test_empty_events_without_evaluation_report_insufficient_data() -> None:
    report = RuleBasedDiagnosticEngine().diagnose(uuid4(), [])

    finding = find_finding(report.findings, "insufficient_data")
    assert finding.severity == "medium"
    assert report.overall_status == "warning"


def test_empty_events_with_evaluation_do_not_report_insufficient_data() -> None:
    run_id = uuid4()

    report = RuleBasedDiagnosticEngine().diagnose(
        run_id,
        [],
        make_evaluation(run_id),
    )

    assert all(
        finding.category != "insufficient_data" for finding in report.findings
    )
    assert report.overall_status == "healthy"


def test_failed_run_produces_critical_finding_and_report() -> None:
    run_id = uuid4()
    events = [
        make_event(
            run_id,
            event_type="run_failed",
            status="failure",
            error="execution stopped",
        )
    ]

    report = RuleBasedDiagnosticEngine().diagnose(run_id, events)

    assert find_finding(report.findings, "run_failure").severity == "critical"
    assert report.overall_status == "critical"


def test_failed_tool_call_produces_tool_failure_finding() -> None:
    run_id = uuid4()
    event = make_event(
        run_id,
        event_type="tool_call",
        status="failure",
        tool_name="search",
        error="index unavailable",
    )

    report = RuleBasedDiagnosticEngine().diagnose(run_id, [event])

    finding = find_finding(report.findings, "tool_failure")
    assert finding.severity == "high"
    assert any("tool_name=search" in evidence for evidence in finding.evidence)
    assert any("index unavailable" in evidence for evidence in finding.evidence)


def test_failed_llm_call_produces_llm_failure_finding() -> None:
    run_id = uuid4()
    event = make_event(
        run_id,
        event_type="llm_call",
        status="failure",
        provider="test-provider",
        error="service unavailable",
    )

    report = RuleBasedDiagnosticEngine().diagnose(run_id, [event])

    finding = find_finding(report.findings, "llm_failure")
    assert finding.severity == "high"
    assert any("service unavailable" in evidence for evidence in finding.evidence)


def test_high_hallucination_score_produces_high_finding() -> None:
    run_id = uuid4()
    evaluation = make_evaluation(run_id, hallucination_score=0.70)

    report = RuleBasedDiagnosticEngine().diagnose(run_id, [], evaluation)

    assert find_finding(report.findings, "high_hallucination").severity == "high"


def test_low_relevance_score_produces_high_finding() -> None:
    run_id = uuid4()
    evaluation = make_evaluation(run_id, relevance_score=0.39)

    report = RuleBasedDiagnosticEngine().diagnose(run_id, [], evaluation)

    assert find_finding(report.findings, "low_relevance").severity == "high"


def test_overall_score_below_50_produces_high_finding() -> None:
    run_id = uuid4()
    evaluation = make_evaluation(run_id, overall_score=49.99)

    report = RuleBasedDiagnosticEngine().diagnose(run_id, [], evaluation)

    assert find_finding(report.findings, "low_overall_score").severity == "high"


@pytest.mark.parametrize("overall_score", [50.0, 69.99])
def test_overall_score_from_50_to_69_99_produces_medium_finding(
    overall_score: float,
) -> None:
    run_id = uuid4()
    evaluation = make_evaluation(run_id, overall_score=overall_score)

    report = RuleBasedDiagnosticEngine().diagnose(run_id, [], evaluation)

    assert find_finding(report.findings, "low_overall_score").severity == "medium"


def test_slow_terminal_event_produces_medium_duration_finding() -> None:
    run_id = uuid4()
    event = make_event(
        run_id,
        event_type="run_completed",
        status="success",
        duration_ms=10_001,
    )

    report = RuleBasedDiagnosticEngine().diagnose(run_id, [event])

    assert (
        find_finding(report.findings, "high_execution_duration").severity
        == "medium"
    )


def test_healthy_run_with_good_evaluation_has_no_findings() -> None:
    run_id = uuid4()
    events = [
        make_event(run_id, event_type="run_started", status="success"),
        make_event(
            run_id,
            event_type="run_completed",
            status="success",
            duration_ms=5_000,
        ),
    ]
    evaluation = make_evaluation(
        run_id,
        correctness_score=0.95,
        relevance_score=0.90,
        context_relevance_score=0.90,
        hallucination_score=0.05,
        overall_score=90,
        passed=True,
    )

    report = RuleBasedDiagnosticEngine().diagnose(run_id, events, evaluation)

    assert report.overall_status == "healthy"
    assert report.findings == []


def test_event_with_different_run_id_is_rejected() -> None:
    run_id = uuid4()
    event = make_event(uuid4())

    with pytest.raises(ValueError, match="events must match"):
        RuleBasedDiagnosticEngine().diagnose(run_id, [event])


def test_evaluation_with_different_run_id_is_rejected() -> None:
    run_id = uuid4()
    evaluation = make_evaluation(uuid4())

    with pytest.raises(ValueError, match="evaluation run_id must match"):
        RuleBasedDiagnosticEngine().diagnose(run_id, [], evaluation)


def test_terminal_duration_exactly_10000_does_not_trigger_finding() -> None:
    run_id = uuid4()
    event = make_event(
        run_id,
        event_type="run_completed",
        status="success",
        duration_ms=10_000,
    )

    report = RuleBasedDiagnosticEngine().diagnose(run_id, [event])

    assert all(
        finding.category != "high_execution_duration"
        for finding in report.findings
    )


def test_duration_diagnostic_uses_failure_detector_selected_duration() -> None:
    run_id = uuid4()
    events = [
        make_event(
            run_id,
            event_type="run_failed",
            status="failure",
            duration_ms=15_000,
        ),
        make_event(
            run_id,
            event_type="run_completed",
            status="success",
            duration_ms=5_000,
        ),
    ]

    report = RuleBasedDiagnosticEngine().diagnose(run_id, events)

    assert all(
        finding.category != "high_execution_duration"
        for finding in report.findings
    )
    assert report.metadata["total_duration_ms"] == 5_000
