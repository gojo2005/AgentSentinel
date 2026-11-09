"""Integration tests for the AgentSentinel FastAPI endpoints."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.main import app
from app.api.routes import agent as agent_routes
from app.diagnostics import service as diagnostic_service
from app.evaluation import service as evaluation_service
from app.diagnostics.schemas import DiagnosticFinding, DiagnosticReport
from app.evaluation.schemas import EvaluationResult
from app.monitoring.event_logger import EventLogger
from app.monitoring.schemas import ExecutionEvent
from app.storage.database import get_db
from app.storage.diagnostic_repository import DiagnosticReportRepository
from app.storage.evaluation_repository import EvaluationResultRepository
from app.storage.event_repository import ExecutionEventRepository
from app.storage.models import Base


@pytest.fixture
def api_engine(tmp_path: Path) -> Iterator[Engine]:
    """Create and dispose of an isolated temporary SQLite database."""
    database_path = tmp_path / "api_tests.db"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def api_session_factory(api_engine: Engine) -> sessionmaker[Session]:
    """Create sessions bound to the temporary API test database."""
    return sessionmaker(
        bind=api_engine,
        autoflush=False,
        autocommit=False,
    )


@pytest.fixture
def db_session(api_session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a session for seeding records in the temporary database."""
    with api_session_factory() as session:
        yield session


@pytest.fixture
def client(
    api_session_factory: sessionmaker[Session],
) -> Iterator[TestClient]:
    """Create a TestClient whose database dependency uses only the test DB."""
    def override_get_db() -> Iterator[Session]:
        with api_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)


def make_event(
    run_id: UUID,
    timestamp: datetime,
    *,
    event_type: Literal[
        "run_started",
        "llm_call",
        "tool_call",
        "run_completed",
        "run_failed",
    ],
    status: Literal["started", "success", "failure"],
) -> ExecutionEvent:
    """Build a valid execution event with deterministic fields."""
    return ExecutionEvent(
        run_id=run_id,
        event_type=event_type,
        timestamp=timestamp,
        agent_name="api-test-agent",
        status=status,
        tool_name="test-tool" if event_type == "tool_call" else None,
        input={"source": "api-test"},
        output={"result": "ok"} if status == "success" else None,
        duration_ms=2_000 if event_type in {"run_completed", "run_failed"} else None,
        error="simulated failure" if event_type == "run_failed" else None,
        metadata={"test": True},
    )


def persist_execution(
    session: Session,
    run_id: UUID,
    started_at: datetime,
    *,
    terminal: bool = True,
    failed: bool = False,
) -> list[ExecutionEvent]:
    """Persist a run's start, tool call, and optional terminal event."""
    events = [
        make_event(
            run_id,
            started_at,
            event_type="run_started",
            status="started",
        ),
        make_event(
            run_id,
            started_at + timedelta(seconds=1),
            event_type="tool_call",
            status="success",
        ),
    ]
    if terminal:
        events.append(
            make_event(
                run_id,
                started_at + timedelta(seconds=2),
                event_type="run_failed" if failed else "run_completed",
                status="failure" if failed else "success",
            )
        )

    repository = ExecutionEventRepository(session)
    for event in reversed(events):
        repository.save(event)
    return events


def make_evaluation(run_id: UUID, timestamp: datetime) -> EvaluationResult:
    """Build a valid evaluation result for a run."""
    return EvaluationResult(
        run_id=run_id,
        timestamp=timestamp,
        correctness_score=0.9,
        relevance_score=0.8,
        context_relevance_score=0.7,
        hallucination_score=0.1,
        overall_score=85,
        passed=True,
        feedback="The result is supported.",
        evaluator="rule_based",
        metadata={"test": True},
    )


def make_report(
    run_id: UUID,
    timestamp: datetime,
    *,
    title: str = "Test diagnostic finding",
) -> DiagnosticReport:
    """Build a valid diagnostic report containing one finding."""
    finding = DiagnosticFinding(
        category="reliability",
        severity="medium",
        title=title,
        description="A diagnostic condition was detected.",
        evidence=["test evidence"],
        recommendation="Review the condition.",
        metadata={"test": True},
    )
    return DiagnosticReport(
        run_id=run_id,
        timestamp=timestamp,
        overall_status="warning",
        findings=[finding],
        summary="A test diagnostic report.",
        metadata={"test": True},
    )


def test_health_endpoint_returns_healthy_service(client: TestClient) -> None:
    """The health endpoint identifies AgentSentinel as healthy."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "application": "AgentSentinel",
    }


def test_agent_endpoint_returns_expected_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mocked agent run returns its answer and shared run identifier."""
    run_id = uuid4()

    def create_test_logger() -> EventLogger:
        return EventLogger()

    def run_test_agent(question: str, *, logger: EventLogger) -> str:
        logger.log_event(
            run_id=run_id,
            event_type="run_started",
            agent_name="api-test-agent",
            status="started",
            input={"question": question},
        )
        logger.log_event(
            run_id=run_id,
            event_type="run_completed",
            agent_name="api-test-agent",
            status="success",
            output={"answer": "A test answer."},
        )
        return "A test answer."

    monkeypatch.setattr(
        agent_routes,
        "create_production_event_logger",
        create_test_logger,
    )
    monkeypatch.setattr(agent_routes, "run_agent", run_test_agent)

    response = client.post(
        "/api/v1/agent/run",
        json={"question": "What is the test answer?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": str(run_id),
        "question": "What is the test answer?",
        "answer": "A test answer.",
        "status": "completed",
    }


@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_agent_endpoint_rejects_empty_questions(
    client: TestClient,
    question: str,
) -> None:
    """Empty and whitespace-only questions fail request validation."""
    response = client.post("/api/v1/agent/run", json={"question": question})

    assert response.status_code == 422


def test_agent_endpoint_rejects_questions_over_maximum_length(
    client: TestClient,
) -> None:
    """Oversized questions are rejected before agent execution."""
    response = client.post(
        "/api/v1/agent/run",
        json={"question": "q" * 4001},
    )

    assert response.status_code == 422


def test_agent_endpoint_returns_safe_error_on_execution_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent errors return a generic message without exception details."""
    def fail_agent(question: str, *, logger: EventLogger) -> str:
        raise RuntimeError("sensitive provider token and stack details")

    monkeypatch.setattr(
        agent_routes,
        "create_production_event_logger",
        EventLogger,
    )
    monkeypatch.setattr(agent_routes, "run_agent", fail_agent)

    response = client.post(
        "/api/v1/agent/run",
        json={"question": "Run the mocked agent."},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Agent execution failed."}
    assert "sensitive" not in response.text
    assert "stack details" not in response.text


def test_agent_endpoint_returns_generic_error_on_timeout(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider timeouts do not expose exception details through the API."""
    def timeout_agent(question: str, *, logger: EventLogger) -> str:
        raise TimeoutError("provider timeout with sensitive-api-key-value")

    monkeypatch.setattr(
        agent_routes,
        "create_production_event_logger",
        EventLogger,
    )
    monkeypatch.setattr(agent_routes, "run_agent", timeout_agent)

    response = client.post(
        "/api/v1/agent/run",
        json={"question": "Run the mocked agent."},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Agent execution failed."}
    assert "sensitive-api-key-value" not in response.text


def test_execution_list_groups_paginates_and_sorts_newest_first(
    client: TestClient,
    db_session: Session,
) -> None:
    """Execution listing paginates distinct runs in newest-first order."""
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    oldest_run_id, middle_run_id, newest_run_id = uuid4(), uuid4(), uuid4()
    persist_execution(db_session, oldest_run_id, base_time)
    persist_execution(
        db_session,
        middle_run_id,
        base_time + timedelta(minutes=1),
        failed=True,
    )
    persist_execution(
        db_session,
        newest_run_id,
        base_time + timedelta(minutes=2),
        terminal=False,
    )

    first_page = client.get("/api/v1/executions", params={"limit": 2})

    assert first_page.status_code == 200
    payload = first_page.json()
    assert payload["total"] == 3
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert [item["run_id"] for item in payload["items"]] == [
        str(newest_run_id),
        str(middle_run_id),
    ]
    assert [item["status"] for item in payload["items"]] == ["running", "failed"]
    assert [item["total_events"] for item in payload["items"]] == [2, 3]

    second_page = client.get(
        "/api/v1/executions",
        params={"limit": 2, "offset": 2},
    )
    assert second_page.status_code == 200
    assert [item["run_id"] for item in second_page.json()["items"]] == [
        str(oldest_run_id)
    ]


def test_execution_detail_and_events_are_chronological(
    client: TestClient,
    db_session: Session,
) -> None:
    """Execution detail and events endpoints return events chronologically."""
    run_id = uuid4()
    started_at = datetime(2026, 2, 1, tzinfo=timezone.utc)
    expected_events = persist_execution(db_session, run_id, started_at)
    usage_event = ExecutionEvent(
        run_id=run_id,
        event_type="llm_call",
        timestamp=started_at + timedelta(milliseconds=500),
        agent_name="api-test-agent",
        provider="Groq",
        model="configured-model",
        status="success",
        duration_ms=120,
        metadata={
            "token_usage": {
                "available": True,
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            },
            "estimated_cost": {
                "available": True,
                "amount_usd": "0.000092000000",
                "currency": "USD",
                "pricing_version": "test-version",
            },
        },
    )
    ExecutionEventRepository(db_session).save(usage_event)
    expected_events.insert(1, usage_event)

    detail = client.get(f"/api/v1/executions/{run_id}")
    event_list = client.get(f"/api/v1/executions/{run_id}/events")

    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["summary"]["run_id"] == str(run_id)
    assert detail_payload["summary"]["status"] == "completed"
    assert detail_payload["summary"]["total_events"] == 4
    expected_ids = [str(event.event_id) for event in expected_events]
    assert [event["event_id"] for event in detail_payload["events"]] == expected_ids
    assert detail_payload["events"][1]["metadata"]["token_usage"] == {
        "available": True,
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert detail_payload["events"][1]["metadata"]["estimated_cost"][
        "pricing_version"
    ] == "test-version"

    assert event_list.status_code == 200
    assert event_list.json()["run_id"] == str(run_id)
    assert [event["event_id"] for event in event_list.json()["events"]] == expected_ids
    assert event_list.json()["events"][1]["metadata"]["token_usage"] == {
        "available": True,
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    assert event_list.json()["events"][1]["metadata"]["estimated_cost"][
        "amount_usd"
    ] == "0.000092000000"


def test_nonexistent_execution_returns_404(client: TestClient) -> None:
    """Unknown execution IDs return 404."""
    run_id = uuid4()

    response = client.get(f"/api/v1/executions/{run_id}")
    assert response.status_code == 404

    events_response = client.get(
        f"/api/v1/executions/{run_id}/events"
    )
    assert events_response.status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
    ],
)
def test_execution_list_rejects_invalid_pagination(
    client: TestClient,
    params: dict[str, int],
) -> None:
    """Execution pagination validates limit and offset bounds."""
    response = client.get("/api/v1/executions", params=params)

    assert response.status_code == 422


def test_evaluation_list_is_paginated_and_newest_first(
    client: TestClient,
    db_session: Session,
) -> None:
    """Evaluation listing returns newest results first with pagination."""
    base_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
    results = [
        make_evaluation(uuid4(), base_time + timedelta(minutes=index))
        for index in range(3)
    ]
    repository = EvaluationResultRepository(db_session)
    for result in results:
        repository.save(result)

    first_page = client.get("/api/v1/evaluations", params={"limit": 2})

    assert first_page.status_code == 200
    payload = first_page.json()
    assert payload["total"] == 3
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert [item["evaluation_id"] for item in payload["items"]] == [
        str(results[2].evaluation_id),
        str(results[1].evaluation_id),
    ]

    second_page = client.get(
        "/api/v1/evaluations",
        params={"limit": 2, "offset": 2},
    )
    assert [item["evaluation_id"] for item in second_page.json()["items"]] == [
        str(results[0].evaluation_id)
    ]


def test_create_evaluation_endpoint_persists_result_and_supports_history(
    client: TestClient,
    db_session: Session,
) -> None:
    run_id = uuid4()
    persist_execution(db_session, run_id, datetime(2026, 3, 10, tzinfo=timezone.utc))
    request = {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "reference_answer": "Paris",
        "context": "Paris is the capital of France.",
    }

    first = client.post(
        f"/api/v1/executions/{run_id}/evaluations",
        json=request,
    )
    second = client.post(
        f"/api/v1/executions/{run_id}/evaluations",
        json=request,
    )
    history = client.get(f"/api/v1/executions/{run_id}/evaluations")

    assert first.status_code == 201
    assert first.json()["evaluator"] == "rule_based"
    assert first.json()["run_id"] == str(run_id)
    assert first.json()["correctness_score"] == 1
    assert second.status_code == 201
    assert second.json()["evaluation_id"] != first.json()["evaluation_id"]
    assert len(history.json()) == 2


def test_create_evaluation_endpoint_rejects_unknown_run_and_invalid_input(
    client: TestClient,
) -> None:
    unknown_run = uuid4()
    missing_run_response = client.post(
        f"/api/v1/executions/{unknown_run}/evaluations",
        json={"question": "Question", "answer": "Answer"},
    )
    invalid_input_response = client.post(
        f"/api/v1/executions/{unknown_run}/evaluations",
        json={"question": " ", "answer": "Answer"},
    )

    assert missing_run_response.status_code == 404
    assert missing_run_response.json() == {"detail": "Execution not found."}
    assert invalid_input_response.status_code == 422


def test_create_evaluation_endpoint_supports_mocked_llm_judge(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = uuid4()
    persist_execution(db_session, run_id, datetime(2026, 3, 11, tzinfo=timezone.utc))
    judge_calls: list[UUID] = []

    class MockJudge:
        def evaluate(
            self,
            question: str,
            answer: str,
            run_id: UUID,
            reference_answer: str | None = None,
            context: str | None = None,
        ) -> EvaluationResult:
            judge_calls.append(run_id)
            return EvaluationResult(
                run_id=run_id,
                correctness_score=0.8,
                relevance_score=0.8,
                hallucination_score=0.1,
                overall_score=83,
                passed=True,
                evaluator="llm_judge",
                feedback="Mocked result.",
            )

    monkeypatch.setattr(evaluation_service, "LLMJudge", MockJudge)
    response = client.post(
        f"/api/v1/executions/{run_id}/evaluations",
        json={
            "question": "Question",
            "answer": "Answer",
            "evaluator": "llm_judge",
        },
    )

    assert response.status_code == 201
    assert response.json()["evaluator"] == "llm_judge"
    assert judge_calls == [run_id]


def test_create_evaluation_endpoint_hides_llm_configuration_details(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id = uuid4()
    persist_execution(db_session, run_id, datetime(2026, 3, 12, tzinfo=timezone.utc))

    def fail_configuration() -> None:
        raise ValueError("provider key must not be returned")

    monkeypatch.setattr(evaluation_service, "LLMJudge", fail_configuration)
    response = client.post(
        f"/api/v1/executions/{run_id}/evaluations",
        json={
            "question": "Question",
            "answer": "Answer",
            "evaluator": "llm_judge",
        },
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail.startswith("The configured LLM judge is unavailable.")
    assert "Correlation ID: " in detail
    assert "provider key" not in response.text
    assert "ValueError" in caplog.text
    assert "provider key must not be returned" not in caplog.text


def test_create_evaluation_endpoint_hides_llm_execution_errors(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id = uuid4()
    persist_execution(db_session, run_id, datetime(2026, 3, 13, tzinfo=timezone.utc))

    class FailingJudge:
        def evaluate(
            self,
            question: str,
            answer: str,
            run_id: UUID,
            reference_answer: str | None = None,
            context: str | None = None,
        ) -> EvaluationResult:
            raise RuntimeError("unredacted provider credential")

    monkeypatch.setattr(evaluation_service, "LLMJudge", FailingJudge)
    response = client.post(
        f"/api/v1/executions/{run_id}/evaluations",
        json={
            "question": "Question",
            "answer": "Answer",
            "evaluator": "llm_judge",
        },
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail.startswith("Evaluation could not be completed.")
    assert "Correlation ID: " in detail
    assert "credential" not in response.text
    assert "RuntimeError" in caplog.text
    assert "unredacted provider credential" not in caplog.text


def test_create_evaluation_endpoint_hides_persistence_errors(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id = uuid4()
    persist_execution(db_session, run_id, datetime(2026, 3, 14, tzinfo=timezone.utc))

    def fail_save(
        self: EvaluationResultRepository,
        result: EvaluationResult,
    ) -> EvaluationResult:
        raise RuntimeError("sensitive database credential")

    monkeypatch.setattr(
        evaluation_service.EvaluationResultRepository,
        "save",
        fail_save,
    )
    response = client.post(
        f"/api/v1/executions/{run_id}/evaluations",
        json={"question": "Question", "answer": "Answer"},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith("Evaluation result could not be saved.")
    assert "Correlation ID: " in detail
    assert "credential" not in response.text
    assert "RuntimeError" in caplog.text
    assert "sensitive database credential" not in caplog.text


def test_get_evaluation_by_id_and_unknown_id(
    client: TestClient,
    db_session: Session,
) -> None:
    """Evaluation detail returns a stored result and 404 for an unknown ID."""
    result = make_evaluation(uuid4(), datetime(2026, 3, 1, tzinfo=timezone.utc))
    EvaluationResultRepository(db_session).save(result)

    response = client.get(f"/api/v1/evaluations/{result.evaluation_id}")
    missing = client.get(f"/api/v1/evaluations/{uuid4()}")

    assert response.status_code == 200
    assert response.json()["evaluation_id"] == str(result.evaluation_id)
    assert response.json()["run_id"] == str(result.run_id)
    assert response.json()["evaluator"] == result.evaluator
    assert missing.status_code == 404


def test_execution_evaluations_and_execution_existence(
    client: TestClient,
    db_session: Session,
) -> None:
    """Run evaluations are returned, empty when absent, and 404 for unknown runs."""
    base_time = datetime(2026, 4, 1, tzinfo=timezone.utc)
    run_id = uuid4()
    no_evaluation_run_id = uuid4()
    persist_execution(db_session, run_id, base_time)
    persist_execution(
        db_session,
        no_evaluation_run_id,
        base_time + timedelta(minutes=1),
    )
    results = [
        make_evaluation(run_id, base_time + timedelta(minutes=2)),
        make_evaluation(run_id, base_time + timedelta(minutes=3)),
    ]
    repository = EvaluationResultRepository(db_session)
    for result in results:
        repository.save(result)

    response = client.get(f"/api/v1/executions/{run_id}/evaluations")
    empty = client.get(
        f"/api/v1/executions/{no_evaluation_run_id}/evaluations"
    )
    missing = client.get(f"/api/v1/executions/{uuid4()}/evaluations")

    assert response.status_code == 200
    assert [item["evaluation_id"] for item in response.json()] == [
        str(result.evaluation_id) for result in results
    ]
    assert empty.status_code == 200
    assert empty.json() == []
    assert missing.status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
    ],
)
def test_evaluation_list_rejects_invalid_pagination(
    client: TestClient,
    params: dict[str, int],
) -> None:
    """Evaluation pagination validates limit and offset bounds."""
    response = client.get("/api/v1/evaluations", params=params)

    assert response.status_code == 422


def test_diagnostic_list_is_paginated_and_newest_first(
    client: TestClient,
    db_session: Session,
) -> None:
    """Diagnostic listing returns newest reports first with pagination."""
    base_time = datetime(2026, 5, 1, tzinfo=timezone.utc)
    reports = [
        make_report(uuid4(), base_time + timedelta(minutes=index))
        for index in range(3)
    ]
    repository = DiagnosticReportRepository(db_session)
    for report in reports:
        repository.save(report)

    first_page = client.get("/api/v1/diagnostics", params={"limit": 2})

    assert first_page.status_code == 200
    payload = first_page.json()
    assert payload["total"] == 3
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert [item["report_id"] for item in payload["items"]] == [
        str(reports[2].report_id),
        str(reports[1].report_id),
    ]

    second_page = client.get(
        "/api/v1/diagnostics",
        params={"limit": 2, "offset": 2},
    )
    assert [item["report_id"] for item in second_page.json()["items"]] == [
        str(reports[0].report_id)
    ]


def test_get_diagnostic_report_includes_findings_and_unknown_id(
    client: TestClient,
    db_session: Session,
) -> None:
    """Report detail includes its nested findings and missing reports return 404."""
    report = make_report(
        uuid4(),
        datetime(2026, 5, 1, tzinfo=timezone.utc),
        title="Nested API finding",
    )
    DiagnosticReportRepository(db_session).save(report)

    response = client.get(f"/api/v1/diagnostics/{report.report_id}")
    missing = client.get(f"/api/v1/diagnostics/{uuid4()}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["report_id"] == str(report.report_id)
    assert payload["run_id"] == str(report.run_id)
    assert len(payload["findings"]) == 1
    assert payload["findings"][0]["finding_id"] == str(
        report.findings[0].finding_id
    )
    assert payload["findings"][0]["title"] == "Nested API finding"
    assert missing.status_code == 404


def test_execution_diagnostic_reports_and_execution_existence(
    client: TestClient,
    db_session: Session,
) -> None:
    """Run reports are returned, empty when absent, and 404 for unknown runs."""
    base_time = datetime(2026, 6, 1, tzinfo=timezone.utc)
    run_id = uuid4()
    no_report_run_id = uuid4()
    persist_execution(db_session, run_id, base_time)
    persist_execution(
        db_session,
        no_report_run_id,
        base_time + timedelta(minutes=1),
    )
    reports = [
        make_report(run_id, base_time + timedelta(minutes=2)),
        make_report(run_id, base_time + timedelta(minutes=3)),
    ]
    repository = DiagnosticReportRepository(db_session)
    for report in reports:
        repository.save(report)

    response = client.get(f"/api/v1/executions/{run_id}/diagnostics")
    empty = client.get(f"/api/v1/executions/{no_report_run_id}/diagnostics")
    missing = client.get(f"/api/v1/executions/{uuid4()}/diagnostics")

    assert response.status_code == 200
    assert [item["report_id"] for item in response.json()] == [
        str(report.report_id) for report in reports
    ]
    assert empty.status_code == 200
    assert empty.json() == []
    assert missing.status_code == 404


def test_create_diagnostic_endpoint_persists_report_without_evaluation(
    client: TestClient,
    db_session: Session,
) -> None:
    run_id = uuid4()
    persist_execution(
        db_session,
        run_id,
        datetime(2026, 6, 10, tzinfo=timezone.utc),
        failed=True,
    )

    first = client.post(f"/api/v1/executions/{run_id}/diagnostics")
    second = client.post(f"/api/v1/executions/{run_id}/diagnostics")
    history = client.get(f"/api/v1/executions/{run_id}/diagnostics")

    assert first.status_code == 201
    assert first.json()["overall_status"] == "critical"
    assert first.json()["run_id"] == str(run_id)
    assert any(
        finding["category"] == "run_failure"
        for finding in first.json()["findings"]
    )
    assert second.status_code == 201
    assert second.json()["report_id"] != first.json()["report_id"]
    assert len(history.json()) == 2


def test_create_diagnostic_endpoint_rejects_unknown_run(
    client: TestClient,
) -> None:
    response = client.post(f"/api/v1/executions/{uuid4()}/diagnostics")

    assert response.status_code == 404
    assert response.json() == {"detail": "Execution not found."}


def test_create_diagnostic_endpoint_hides_internal_error_and_returns_correlation_id(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run_id = uuid4()
    persist_execution(
        db_session,
        run_id,
        datetime(2026, 6, 11, tzinfo=timezone.utc),
        failed=True,
    )

    def fail_diagnosis(self: object, *args: object, **kwargs: object) -> None:
        raise RuntimeError("api_key=sk-12345678901234567890")

    monkeypatch.setattr(
        diagnostic_service.RuleBasedDiagnosticEngine,
        "diagnose",
        fail_diagnosis,
    )
    response = client.post(f"/api/v1/executions/{run_id}/diagnostics")

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith("Diagnostics could not be completed.")
    assert "Correlation ID: " in detail
    assert "sk-12345678901234567890" not in response.text
    assert "RuntimeError" in caplog.text
    assert "sk-12345678901234567890" not in caplog.text


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
    ],
)
def test_diagnostic_list_rejects_invalid_pagination(
    client: TestClient,
    params: dict[str, int],
) -> None:
    """Diagnostic pagination validates limit and offset bounds."""
    response = client.get("/api/v1/diagnostics", params=params)

    assert response.status_code == 422
