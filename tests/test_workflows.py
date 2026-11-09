"""Tests for persisted evaluation and diagnostic application workflows."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.sanitization import redact_sensitive_data
from app.core.workflow_errors import (
    DiagnosticFailedError,
    EvaluationFailedError,
    ExecutionNotFoundError,
)
from app.diagnostics.service import DiagnosticService
from app.detection.failure_detector import FailureDetector
from app.evaluation.schemas import EvaluationResult
from app.evaluation.service import EvaluationService
from app.monitoring.schemas import ExecutionEvent
from app.storage.diagnostic_repository import DiagnosticReportRepository
from app.storage.event_repository import ExecutionEventRepository
from app.storage.evaluation_repository import EvaluationResultRepository
from app.storage.models import Base


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Create an isolated database for application workflow tests."""
    database_path = tmp_path / "workflow_tests.db"
    test_engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Provide one session bound to the workflow test database."""
    test_session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    with test_session_factory() as test_session:
        yield test_session


def persist_run(
    session: Session,
    run_id: UUID,
    *,
    failed: bool = False,
    error: str = "simulated failure",
) -> None:
    """Persist the minimum terminal event sequence for an execution."""
    started_at = datetime(2026, 8, 1, tzinfo=timezone.utc)
    terminal_type = "run_failed" if failed else "run_completed"
    terminal_status = "failure" if failed else "success"
    repository = ExecutionEventRepository(session)
    for event_type, status, timestamp in (
        ("run_started", "started", started_at),
        (terminal_type, terminal_status, started_at + timedelta(seconds=2)),
    ):
        repository.save(
            ExecutionEvent(
                run_id=run_id,
                event_type=event_type,
                timestamp=timestamp,
                agent_name="workflow-test-agent",
                status=status,
                duration_ms=2_000 if event_type != "run_started" else None,
                error=error if failed and status == "failure" else None,
            )
        )


def make_evaluation(
    run_id: UUID,
    *,
    timestamp: datetime | None = None,
    correctness_score: float | None = 0.9,
    relevance_score: float | None = 0.9,
    context_relevance_score: float | None = 0.9,
    hallucination_score: float | None = 0.1,
    overall_score: float | None = 90,
    passed: bool | None = True,
) -> EvaluationResult:
    """Create a test evaluation associated with a run."""
    return EvaluationResult(
        run_id=run_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        correctness_score=correctness_score,
        relevance_score=relevance_score,
        context_relevance_score=context_relevance_score,
        hallucination_score=hallucination_score,
        overall_score=overall_score,
        passed=passed,
        feedback="Test evaluation feedback.",
        evaluator="rule_based",
    )


class FakeJudge:
    """Return a fixed result and record judge inputs without network access."""

    def __init__(self) -> None:
        self.inputs: dict[str, Any] | None = None

    def evaluate(
        self,
        question: str,
        answer: str,
        run_id: UUID,
        reference_answer: str | None = None,
        context: str | None = None,
    ) -> EvaluationResult:
        self.inputs = {
            "question": question,
            "answer": answer,
            "run_id": run_id,
            "reference_answer": reference_answer,
            "context": context,
        }
        return EvaluationResult(
            run_id=run_id,
            correctness_score=0.8,
            relevance_score=0.75,
            context_relevance_score=None,
            hallucination_score=0.2,
            overall_score=78.33,
            passed=True,
            feedback="Mocked judge feedback.",
            evaluator="llm_judge",
        )


def test_evaluation_workflow_persists_rule_result_and_optional_inputs(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)
    service = EvaluationService(session)

    result = service.evaluate_run(
        run_id,
        question="What color is the sky?",
        answer="The sky is blue.",
        reference_answer="The sky is blue.",
        context="The sky appears blue during the day.",
    )

    saved = EvaluationResultRepository(session).get_by_id(result.evaluation_id)
    assert result.evaluator == "rule_based"
    assert result.run_id == run_id
    assert result.correctness_score == 1
    assert result.context_relevance_score is not None
    assert saved == result


def test_evaluation_workflow_allows_omitted_reference_and_context(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)

    result = EvaluationService(session).evaluate_run(
        run_id,
        question="Name a planet.",
        answer="Mars.",
    )

    assert result.correctness_score is None
    assert result.context_relevance_score is None
    assert result.hallucination_score is None


def test_evaluation_workflow_uses_injected_llm_judge_without_network(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)
    judge = FakeJudge()

    result = EvaluationService(
        session,
        llm_judge_factory=lambda: judge,
    ).evaluate_run(
        run_id,
        question="Question",
        answer="Answer",
        evaluator="llm_judge",
    )

    assert result.evaluator == "llm_judge"
    assert judge.inputs is not None
    assert judge.inputs["run_id"] == run_id
    assert judge.inputs["reference_answer"] is None
    assert EvaluationResultRepository(session).get_by_id(result.evaluation_id) == result


def test_evaluation_workflow_rejects_unknown_run(session: Session) -> None:
    with pytest.raises(ExecutionNotFoundError):
        EvaluationService(session).evaluate_run(
            uuid4(),
            question="Question",
            answer="Answer",
        )


def test_evaluation_workflow_rejects_result_for_another_run(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)

    class MismatchedEvaluator:
        def evaluate(
            self,
            question: str,
            answer: str,
            run_id: UUID,
            reference_answer: str | None = None,
            context: str | None = None,
        ) -> EvaluationResult:
            return make_evaluation(uuid4())

    with pytest.raises(EvaluationFailedError):
        EvaluationService(
            session,
            rule_evaluator=MismatchedEvaluator(),
        ).evaluate_run(
            run_id,
            question="Question",
            answer="Answer",
        )


@pytest.mark.parametrize(
    ("question", "answer", "reference_answer", "context"),
    [
        (" ", "Answer", None, None),
        ("Question", "\t", None, None),
        ("Question", "Answer", " ", None),
        ("Question", "Answer", None, " "),
    ],
)
def test_evaluation_workflow_rejects_invalid_inputs(
    session: Session,
    question: str,
    answer: str,
    reference_answer: str | None,
    context: str | None,
) -> None:
    with pytest.raises(ValueError):
        EvaluationService(session).evaluate_run(
            uuid4(),
            question=question,
            answer=answer,
            reference_answer=reference_answer,
            context=context,
        )


def test_evaluation_workflow_preserves_repeated_results(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)
    service = EvaluationService(session)
    inputs = {
        "question": "Name a planet.",
        "answer": "Mars",
    }

    first = service.evaluate_run(run_id, **inputs)
    second = service.evaluate_run(run_id, **inputs)

    assert first.evaluation_id != second.evaluation_id
    assert len(EvaluationResultRepository(session).get_by_run_id(run_id)) == 2


def test_diagnostic_workflow_persists_failed_run_and_findings(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id, failed=True)

    report = DiagnosticService(session).diagnose_run(run_id)

    stored = DiagnosticReportRepository(session).get_by_id(report.report_id)
    assert stored == report
    assert report.overall_status == "critical"
    assert any(finding.category == "run_failure" for finding in report.findings)
    assert report.findings


def test_diagnostic_workflow_marks_healthy_run_without_findings(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)
    EvaluationResultRepository(session).save(make_evaluation(run_id))

    report = DiagnosticService(session).diagnose_run(run_id)

    assert report.overall_status == "healthy"
    assert report.findings == []


def test_diagnostic_workflow_uses_latest_evaluation_for_quality_findings(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)
    results = EvaluationResultRepository(session)
    base_time = datetime(2026, 8, 1, tzinfo=timezone.utc)
    results.save(make_evaluation(run_id, timestamp=base_time))
    results.save(
        make_evaluation(
            run_id,
            timestamp=base_time + timedelta(minutes=1),
            correctness_score=0,
            relevance_score=0.2,
            context_relevance_score=None,
            hallucination_score=0.9,
            overall_score=30,
            passed=False,
        )
    )

    report = DiagnosticService(session).diagnose_run(run_id)

    categories = {finding.category for finding in report.findings}
    assert report.overall_status == "warning"
    assert {"high_hallucination", "low_relevance", "low_overall_score"} <= categories


def test_diagnostic_workflow_does_not_require_an_evaluation(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id, failed=True)

    report = DiagnosticService(session).diagnose_run(run_id)

    assert report.overall_status == "critical"
    assert any(finding.category == "run_failure" for finding in report.findings)


def test_diagnostic_workflow_rejects_unknown_run(session: Session) -> None:
    with pytest.raises(ExecutionNotFoundError):
        DiagnosticService(session).diagnose_run(uuid4())


def test_diagnostic_workflow_rejects_mismatched_evaluation_run_id(
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)
    mismatched_result = make_evaluation(uuid4())
    service = DiagnosticService(session)
    monkeypatch.setattr(
        service._evaluations,
        "get_by_run_id",
        lambda requested_run_id: [mismatched_result],
    )

    with pytest.raises(DiagnosticFailedError):
        service.diagnose_run(run_id)


def test_diagnostic_workflow_preserves_repeated_reports_and_findings(
    session: Session,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id, failed=True)
    service = DiagnosticService(session)

    first = service.diagnose_run(run_id)
    second = service.diagnose_run(run_id)
    reports = DiagnosticReportRepository(session).get_by_run_id(run_id)

    assert first.report_id != second.report_id
    assert len(reports) == 2
    assert all(report.findings for report in reports)


def test_unexpected_evaluator_error_is_sanitized(
    session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id = uuid4()
    persist_run(session, run_id)

    class FailingEvaluator:
        def evaluate(
            self,
            question: str,
            answer: str,
            run_id: UUID,
            reference_answer: str | None = None,
            context: str | None = None,
        ) -> EvaluationResult:
            raise RuntimeError(
                "provider failed with api_key=sk-12345678901234567890 "
                "and password=hunter2"
            )

    with pytest.raises(EvaluationFailedError) as error_info:
        EvaluationService(
            session,
            rule_evaluator=FailingEvaluator(),
        ).evaluate_run(
            run_id,
            question="Question",
            answer="Answer",
        )

    assert "sensitive" not in str(error_info.value)
    assert error_info.value.correlation_id
    assert "RuntimeError" in caplog.text
    assert str(error_info.value.correlation_id) in caplog.text
    assert "sk-12345678901234567890" not in caplog.text
    assert "hunter2" not in caplog.text
    assert "provider failed" not in caplog.text


@pytest.mark.parametrize(
    "secret",
    [
        "api_key=sk-12345678901234567890",
        "Bearer eyJabcdefgh.ijklmnop.qrstuvwx",
        "password=hunter2",
        "token=ordinary-secret-value",
    ],
)
def test_diagnostic_findings_and_failure_summary_redact_event_errors(
    session: Session,
    secret: str,
) -> None:
    run_id = uuid4()
    persist_run(
        session,
        run_id,
        failed=True,
        error=f"provider failure: {secret}",
    )

    events = ExecutionEventRepository(session).get_by_run_id(run_id)
    failure_summary = FailureDetector().analyze_run(events)
    report = DiagnosticService(session).diagnose_run(run_id)
    report_text = str(report.model_dump())

    assert secret not in str(failure_summary)
    assert secret not in report_text
    assert "provider failure" in report_text


def test_nested_error_data_is_recursively_redacted() -> None:
    payload = {
        "message": "request failed",
        "details": [
            {"Authorization": "Bearer confidential-token"},
            {"message": "password: 'super-secret'"},
        ],
    }

    redacted = redact_sensitive_data(payload)

    assert redacted["details"][0]["Authorization"] == "[REDACTED]"
    assert "confidential-token" not in str(redacted)
    assert "super-secret" not in str(redacted)
    assert "request failed" in str(redacted)
