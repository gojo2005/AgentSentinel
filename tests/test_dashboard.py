"""Tests for dashboard rendering with a mocked AgentSentinel API client."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest

from app.dashboard import main as dashboard
from app.dashboard.api_client import APIClientError


class MockAPIClient:
    """Return configured API responses without making network requests."""

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        *,
        errors: dict[str, APIClientError] | None = None,
        post_responses: dict[str, Any] | None = None,
        post_errors: dict[str, APIClientError] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.errors = errors or {}
        self.post_responses = post_responses or {}
        self.post_errors = post_errors or {}
        self.requests: list[tuple[str, dict[str, Any] | None]] = []
        self.post_requests: list[tuple[str, dict[str, Any]]] = []

    def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self.requests.append((path, params))
        if path in self.errors:
            raise self.errors[path]
        return self.responses[path]

    def post(self, path: str, *, json: dict[str, Any]) -> Any:
        self.post_requests.append((path, json))
        if path in self.post_errors:
            raise self.post_errors[path]
        return self.post_responses[path]


class FakeDashboardUI:
    """Capture Streamlit output and provide deterministic widget values."""

    def __init__(
        self,
        *,
        selections: dict[str, Any] | None = None,
        buttons: set[str] | None = None,
        text_inputs: dict[str, str] | None = None,
    ) -> None:
        self.session_state: dict[str, Any] = {}
        self.selections = selections or {}
        self.buttons = buttons or set()
        self.text_inputs = text_inputs or {}
        self.metrics: list[tuple[str, Any]] = []
        self.dataframes: list[Any] = []
        self.dataframe_options: list[dict[str, Any]] = []
        self.charts: list[tuple[Any, dict[str, Any]]] = []
        self.messages: list[str] = []
        self.expanders: list[str] = []
        self.spinners: list[str] = []
        self.container_borders: list[bool] = []
        self.column_config = self

    def columns(
        self,
        spec: int | list[int],
        **_: Any,
    ) -> list[FakeDashboardUI]:
        count = spec if isinstance(spec, int) else len(spec)
        return [self] * count

    def __enter__(self) -> FakeDashboardUI:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def form(self, _: str) -> FakeDashboardUI:
        return self

    def container(self, *, border: bool = False) -> FakeDashboardUI:
        self.container_borders.append(border)
        return self

    def spinner(self, text: str) -> FakeDashboardUI:
        self.spinners.append(text)
        return self

    def TextColumn(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"args": args, **kwargs}

    def subheader(self, value: str) -> None:
        self.messages.append(value)

    def metric(self, label: str, value: Any, **_: Any) -> None:
        self.metrics.append((label, value))

    def selectbox(
        self,
        label: str,
        options: tuple[Any, ...] | list[Any],
        **_: Any,
    ) -> Any:
        return self.selections.get(label, options[0] if options else None)

    def button(self, label: str, **_: Any) -> bool:
        return label in self.buttons

    def form_submit_button(self, label: str, **kwargs: Any) -> bool:
        return self.button(label, **kwargs)

    def text_input(self, label: str, **kwargs: Any) -> str:
        key = kwargs.get("key", label)
        return self.text_inputs.get(key, "")

    def text_area(self, label: str, **kwargs: Any) -> str:
        key = kwargs.get("key", label)
        return self.text_inputs.get(key, "")

    def caption(self, value: str) -> None:
        self.messages.append(value)

    def dataframe(self, value: Any, **_: Any) -> None:
        self.dataframes.append(value)
        self.dataframe_options.append(_)

    def bar_chart(self, data: Any = None, **kwargs: Any) -> None:
        self.charts.append((data, kwargs))

    def line_chart(self, data: Any = None, **kwargs: Any) -> None:
        self.charts.append((data, kwargs))

    def info(self, value: str) -> None:
        self.messages.append(value)

    def error(self, value: str) -> None:
        self.messages.append(value)

    def warning(self, value: str) -> None:
        self.messages.append(value)

    def success(self, value: str) -> None:
        self.messages.append(value)

    def markdown(self, value: str) -> None:
        self.messages.append(value)

    def code(self, value: str) -> None:
        self.messages.append(value)

    def write(self, value: Any) -> None:
        self.messages.append(str(value))

    def expander(self, label: str) -> FakeDashboardUI:
        self.expanders.append(label)
        return self


def make_report(
    *,
    status: str = "warning",
    summary: str = "A diagnostic report.",
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a dictionary matching the actual DiagnosticReport response."""
    return {
        "report_id": str(uuid4()),
        "run_id": str(uuid4()),
        "timestamp": "2026-11-06T10:00:00+00:00",
        "overall_status": status,
        "findings": findings or [],
        "summary": summary,
        "metadata": {},
    }


def make_finding(**overrides: Any) -> dict[str, Any]:
    """Build a dictionary matching the actual DiagnosticFinding schema."""
    finding = {
        "finding_id": str(uuid4()),
        "category": "reliability",
        "severity": "high",
        "title": "Slow tool response",
        "description": "A tool call exceeded its expected duration.",
        "evidence": ["The call took 4 seconds."],
        "recommendation": "Review tool latency.",
        "metadata": {},
    }
    finding.update(overrides)
    return finding


def make_evaluation_result(
    run_id: str,
    *,
    overall_score: float | None = 82,
    passed: bool | None = True,
    evaluator: str = "rule_based",
    feedback: str = "The answer is supported.",
) -> dict[str, Any]:
    """Build a dictionary matching the persisted evaluation response schema."""
    return {
        "evaluation_id": str(uuid4()),
        "run_id": run_id,
        "timestamp": "2026-11-06T10:00:00+00:00",
        "correctness_score": 0.8,
        "relevance_score": 0.9,
        "context_relevance_score": 0.75,
        "hallucination_score": 0.1,
        "overall_score": overall_score,
        "passed": passed,
        "feedback": feedback,
        "evaluator": evaluator,
        "metadata": {},
    }


def execution_detail_responses(
    run_id: str,
    *,
    evaluations: list[dict[str, Any]] | None = None,
    reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build GET responses used by the execution detail workflows."""
    return {
        f"/api/v1/executions/{run_id}": {
            "summary": {
                "status": "completed",
                "started_at": "2026-11-06T10:00:00+00:00",
                "completed_at": "2026-11-06T10:00:02+00:00",
                "duration_ms": 2000,
            }
        },
        f"/api/v1/executions/{run_id}/events": {
            "events": [
                {
                    "event_type": "run_completed",
                    "timestamp": "2026-11-06T10:00:02+00:00",
                    "status": "success",
                    "duration_ms": 2000,
                }
            ]
        },
        f"/api/v1/executions/{run_id}/evaluations": evaluations or [],
        f"/api/v1/executions/{run_id}/diagnostics": reports or [],
    }


def install_fake_ui(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selections: dict[str, Any] | None = None,
    buttons: set[str] | None = None,
    text_inputs: dict[str, str] | None = None,
) -> FakeDashboardUI:
    """Replace Streamlit calls with a lightweight output-capturing fake."""
    ui = FakeDashboardUI(
        selections=selections,
        buttons=buttons,
        text_inputs=text_inputs,
    )
    monkeypatch.setattr(dashboard, "st", ui)
    return ui


def make_list_response(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the actual paginated diagnostic endpoint response shape."""
    return {
        "items": reports,
        "total": len(reports),
        "limit": 100,
        "offset": 0,
    }


def test_execution_detail_generates_evaluation_and_refreshes_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    old_result = make_evaluation_result(run_id, overall_score=64, passed=False)
    generated_result = make_evaluation_result(
        run_id,
        overall_score=91,
        evaluator="rule_based",
        feedback="A strong answer.",
    )
    responses = execution_detail_responses(run_id, evaluations=[old_result, generated_result])
    responses[f"/api/v1/evaluations/{old_result['evaluation_id']}"] = old_result
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Generate Evaluation"},
        text_inputs={
            f"evaluation_question_{run_id}": "  What is 2 + 2? ",
            f"evaluation_answer_{run_id}": "  4 ",
            f"evaluation_reference_answer_{run_id}": "Four",
            f"evaluation_context_{run_id}": "Basic arithmetic",
        },
    )
    client = MockAPIClient(
        responses,
        post_responses={
            f"/api/v1/executions/{run_id}/evaluations": generated_result,
        },
    )

    dashboard._render_execution_details(client, run_id)

    assert client.post_requests == [
        (
            f"/api/v1/executions/{run_id}/evaluations",
            {
                "question": "What is 2 + 2?",
                "answer": "4",
                "reference_answer": "Four",
                "context": "Basic arithmetic",
                "evaluator": "rule_based",
            },
        )
    ]
    assert (f"/api/v1/executions/{run_id}/evaluations", None) in client.requests
    assert any("Evaluation generated and saved." in message for message in ui.messages)
    assert ("Overall score", "91.00 / 100") in ui.metrics
    assert ("Correctness", "0.80 / 1") in ui.metrics
    assert ("Relevance", "0.90 / 1") in ui.metrics
    assert ("Context relevance", "0.75 / 1") in ui.metrics
    assert ("Hallucination", "0.10 / 1") in ui.metrics
    assert ("Result", "Passed") in ui.metrics
    assert ("Evaluator", "rule_based") in ui.metrics
    assert "A strong answer." in repr(ui.messages)
    assert any(
        {row["Evaluation ID"] for row in dataframe}
        == {old_result["evaluation_id"], generated_result["evaluation_id"]}
        for dataframe in ui.dataframes
        if dataframe and "Evaluation ID" in dataframe[0]
    )


def test_evaluation_form_omits_blank_optional_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    generated_result = make_evaluation_result(run_id)
    responses = execution_detail_responses(run_id)
    install_fake_ui(
        monkeypatch,
        buttons={"Generate Evaluation"},
        text_inputs={
            f"evaluation_question_{run_id}": "Question",
            f"evaluation_answer_{run_id}": "Answer",
        },
    )
    client = MockAPIClient(
        responses,
        post_responses={
            f"/api/v1/executions/{run_id}/evaluations": generated_result,
        },
    )

    dashboard._render_execution_details(client, run_id)

    assert client.post_requests[0][1] == {
        "question": "Question",
        "answer": "Answer",
        "evaluator": "rule_based",
    }


def test_evaluation_form_requires_question_and_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Generate Evaluation"},
        text_inputs={f"evaluation_question_{run_id}": "Question"},
    )
    client = MockAPIClient(execution_detail_responses(run_id))

    dashboard._render_execution_details(client, run_id)

    assert any("both a question and an answer" in msg for msg in ui.messages)
    assert not client.post_requests


def test_llm_judge_configuration_error_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    ui = install_fake_ui(
        monkeypatch,
        selections={"Evaluator": "LLM judge"},
        buttons={"Generate Evaluation"},
        text_inputs={
            f"evaluation_question_{run_id}": "Question",
            f"evaluation_answer_{run_id}": "Answer",
        },
    )
    client = MockAPIClient(
        execution_detail_responses(run_id),
        post_errors={
            f"/api/v1/executions/{run_id}/evaluations": APIClientError(
                "The backend returned an HTTP 503 response.",
                status_code=503,
            )
        },
    )

    dashboard._render_execution_details(client, run_id)

    assert any("LLM judge is unavailable" in message for message in ui.messages)
    assert "credential" not in repr(ui.messages).lower()
    assert client.post_requests[0][1]["evaluator"] == "llm_judge"


def test_evaluation_generation_handles_malformed_response_and_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Generate Evaluation"},
        text_inputs={
            f"evaluation_question_{run_id}": "Question",
            f"evaluation_answer_{run_id}": "Answer",
        },
    )
    client = MockAPIClient(
        execution_detail_responses(run_id),
        post_responses={f"/api/v1/executions/{run_id}/evaluations": {"feedback": "bad"}},
    )

    dashboard._render_execution_details(client, run_id)

    assert any("evaluation in an unexpected format" in msg for msg in ui.messages)
    assert any("This execution has no evaluations." in msg for msg in ui.messages)

    ui = install_fake_ui(
        monkeypatch,
        buttons={"Generate Evaluation"},
        text_inputs={
            f"evaluation_question_{run_id}": "Question",
            f"evaluation_answer_{run_id}": "Answer",
        },
    )
    client = MockAPIClient(
        execution_detail_responses(run_id),
        post_errors={
            f"/api/v1/executions/{run_id}/evaluations": APIClientError(
                "The backend returned an HTTP 502 response.",
                status_code=502,
            )
        },
    )

    dashboard._render_execution_details(client, run_id)

    assert any("HTTP 502" in msg for msg in ui.messages)
    assert "sensitive" not in repr(ui.messages).lower()


def test_execution_detail_generates_diagnostic_without_evaluation_and_refreshes_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    old_report = make_report(status="warning", findings=[make_finding()])
    old_report["run_id"] = run_id
    new_report = make_report(
        status="critical",
        summary="The run failed.",
        findings=[
            make_finding(
                severity="critical",
                evidence=["A call failed."],
                recommendation="Inspect the failing provider.",
            )
        ],
    )
    new_report["run_id"] = run_id
    responses = execution_detail_responses(run_id, reports=[old_report, new_report])
    responses[f"/api/v1/diagnostics/{old_report['report_id']}"] = old_report
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Generate Diagnostic Report"},
    )
    client = MockAPIClient(
        responses,
        post_responses={
            f"/api/v1/executions/{run_id}/diagnostics": new_report,
        },
    )

    dashboard._render_execution_details(client, run_id)

    assert client.post_requests == [
        (f"/api/v1/executions/{run_id}/diagnostics", {})
    ]
    assert any("Diagnostic report generated and saved." in message for message in ui.messages)
    assert ("Status", "Critical") in ui.metrics
    assert ("Findings", "1") in ui.metrics
    rendered = repr((ui.messages, ui.expanders, ui.dataframes))
    assert "A call failed." in rendered
    assert "Inspect the failing provider." in rendered
    assert any(
        {row["Report ID"] for row in dataframe}
        == {old_report["report_id"], new_report["report_id"]}
        for dataframe in ui.dataframes
        if dataframe and "Report ID" in dataframe[0]
    )
    assert not any(
        path.endswith("/evaluations")
        for path, _ in client.post_requests
    )


def test_diagnostic_generation_handles_empty_history_malformed_response_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    ui = install_fake_ui(monkeypatch, buttons={"Generate Diagnostic Report"})
    client = MockAPIClient(
        execution_detail_responses(run_id),
        post_responses={f"/api/v1/executions/{run_id}/diagnostics": {"summary": "bad"}},
    )

    dashboard._render_execution_details(client, run_id)

    assert any("diagnostic report in an unexpected format" in msg for msg in ui.messages)
    assert any("This execution has no diagnostic reports." in msg for msg in ui.messages)

    ui = install_fake_ui(monkeypatch, buttons={"Generate Diagnostic Report"})
    client = MockAPIClient(
        execution_detail_responses(run_id),
        post_errors={
            f"/api/v1/executions/{run_id}/diagnostics": APIClientError(
                "The backend returned an HTTP 500 response.",
                status_code=500,
            )
        },
    )

    dashboard._render_execution_details(client, run_id)

    assert any("HTTP 500" in msg for msg in ui.messages)
    assert "sensitive" not in repr(ui.messages).lower()


def test_agent_playground_runs_existing_endpoint_and_displays_answer_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    response = {
        "run_id": run_id,
        "question": "What is 2 + 2?",
        "answer": "The answer is 4.",
        "status": "completed",
    }
    execution = {
        "summary": {
            "run_id": run_id,
            "status": "completed",
            "duration_ms": 1842.4,
        },
        "events": [
            {
                "event_type": "tool_call",
                "tool_name": "calculator",
                "status": "success",
                "duration_ms": 12,
                "error": None,
            }
        ],
    }
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Run Agent"},
        text_inputs={dashboard._PLAYGROUND_QUESTION_KEY: "What is 2 + 2?"},
    )
    client = MockAPIClient(
        {f"/api/v1/executions/{run_id}": execution},
        post_responses={"/api/v1/agent/run": response},
    )

    dashboard._render_agent_playground(client)

    assert client.post_requests == [
        ("/api/v1/agent/run", {"question": "What is 2 + 2?"})
    ]
    assert (f"/api/v1/executions/{run_id}", None) in client.requests
    assert "The answer is 4." in ui.messages
    assert any(f"Run ID: {run_id}" in message for message in ui.messages)
    assert ("Execution status", "Completed") in ui.metrics
    assert ("Duration", "1842 ms") in ui.metrics
    assert any(
        row["Tool"] == "calculator"
        for dataframe in ui.dataframes
        for row in dataframe
    )
    assert ui.spinners == ["Running the agent..."]
    assert "Executions" in repr(ui.messages)
    assert ui.session_state[dashboard._PLAYGROUND_RESULT_KEY] == response
    assert ui.session_state[dashboard._PLAYGROUND_EXECUTION_KEY] == execution


def test_agent_playground_keeps_answer_when_execution_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    response = {
        "run_id": run_id,
        "question": "What is 2 + 2?",
        "answer": "The answer is 4.",
        "status": "completed",
    }
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Run Agent"},
        text_inputs={dashboard._PLAYGROUND_QUESTION_KEY: "What is 2 + 2?"},
    )
    client = MockAPIClient(
        errors={
            f"/api/v1/executions/{run_id}": APIClientError(
                "The backend is unavailable. Check that the API server is running."
            )
        },
        post_responses={"/api/v1/agent/run": response},
    )

    dashboard._render_agent_playground(client)

    assert "The answer is 4." in ui.messages
    assert ("Execution status", "Completed") in ui.metrics
    assert any("persisted execution details could not be loaded" in msg for msg in ui.messages)


def test_agent_playground_handles_api_failure_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Run Agent"},
        text_inputs={dashboard._PLAYGROUND_QUESTION_KEY: "Question"},
    )
    client = MockAPIClient(
        post_errors={
            "/api/v1/agent/run": APIClientError(
                "The backend returned an HTTP 500 response.",
                status_code=500,
            )
        }
    )

    dashboard._render_agent_playground(client)

    assert any("HTTP 500" in msg for msg in ui.messages)
    assert "credential" not in repr(ui.messages).lower()
    assert not client.requests


def test_agent_playground_retains_previous_result_after_failed_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_run_id = str(uuid4())
    old_result = {
        "run_id": old_run_id,
        "question": "Previous question",
        "answer": "Previous answer",
        "status": "completed",
    }
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Run Agent"},
        text_inputs={dashboard._PLAYGROUND_QUESTION_KEY: "New question"},
    )
    ui.session_state[dashboard._PLAYGROUND_RESULT_KEY] = old_result
    client = MockAPIClient(
        post_errors={
            "/api/v1/agent/run": APIClientError(
                "The backend returned an HTTP 500 response.",
                status_code=500,
            )
        }
    )

    dashboard._render_agent_playground(client)

    assert "Previous answer" in ui.messages
    assert any(f"Run ID: {old_run_id}" in message for message in ui.messages)


def test_agent_playground_requires_question_and_rejects_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Run Agent"},
        text_inputs={dashboard._PLAYGROUND_QUESTION_KEY: "   "},
    )
    client = MockAPIClient()

    dashboard._render_agent_playground(client)

    assert any("Enter a question" in msg for msg in ui.messages)
    assert not client.post_requests

    ui = install_fake_ui(
        monkeypatch,
        buttons={"Run Agent"},
        text_inputs={dashboard._PLAYGROUND_QUESTION_KEY: "Question"},
    )
    client = MockAPIClient(post_responses={"/api/v1/agent/run": {"answer": "bad"}})

    dashboard._render_agent_playground(client)

    assert any("unexpected agent response" in msg for msg in ui.messages)
    assert dashboard._PLAYGROUND_RESULT_KEY not in ui.session_state


@pytest.mark.parametrize(
    "value",
    [
        "2026-11-09T19:42:35+05:30",
        "2026-11-09T14:12:35Z",
        "2026-11-09T14:12:35",
    ],
)
def test_timestamp_format_is_complete_and_includes_timezone(value: str) -> None:
    expected_time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if expected_time.tzinfo is None:
        expected_time = expected_time.replace(tzinfo=timezone.utc)
    local_time = expected_time.astimezone()
    offset = local_time.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if offset else "+00:00"

    formatted = dashboard._format_local_datetime(value)

    assert formatted == (
        f"{local_time.strftime('%d %b %Y, %H:%M:%S')} UTC{offset}"
    )
    assert "2026-" not in formatted


def test_execution_detail_cards_show_full_timestamps_and_formatted_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    responses = execution_detail_responses(run_id)
    summary = responses[f"/api/v1/executions/{run_id}"]["summary"]
    summary["started_at"] = "2026-11-09T19:42:35+05:30"
    summary["completed_at"] = "2026-11-09T19:42:37+05:30"
    summary["duration_ms"] = 1967
    ui = install_fake_ui(monkeypatch)

    dashboard._render_execution_details(MockAPIClient(responses), run_id)

    started_display = dashboard._format_local_datetime(summary["started_at"])
    completed_display = dashboard._format_local_datetime(summary["completed_at"])
    assert "Last Execution Details" in ui.messages
    assert started_display in ui.messages
    assert completed_display in ui.messages
    assert "1.97 seconds" in ui.messages
    assert " UTC+" in started_display or " UTC-" in started_display
    assert " UTC+" in completed_display or " UTC-" in completed_display
    assert ui.container_borders[:4] == [True, True, True, True]


def test_execution_usage_aggregates_reported_calls_and_keeps_missing_usage_unknown() -> None:
    events = [
        {
            "event_type": "llm_call",
            "duration_ms": 120,
            "metadata": {
                "token_usage": {
                    "available": True,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                },
                "estimated_cost": {
                    "available": True,
                    "amount_usd": "0.000010000000",
                    "pricing_version": "test-version",
                },
            },
        },
        {
            "event_type": "llm_call",
            "duration_ms": 80,
            "metadata": {
                "token_usage": {
                    "available": True,
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                }
            },
        },
        {
            "event_type": "llm_call",
            "duration_ms": 100,
            "metadata": {
                "token_usage": {
                    "available": False,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                }
            },
        },
    ]

    summary = dashboard._execution_usage_summary(events)

    assert summary["input_tokens"] is None
    assert summary["output_tokens"] is None
    assert summary["total_tokens"] is None
    assert summary["llm_call_count"] == 3
    assert summary["reported_call_count"] == 2
    assert summary["unavailable_call_count"] == 1
    assert summary["llm_latency_ms"] == 300


def test_execution_usage_sums_each_complete_llm_call() -> None:
    events = [
        {
            "event_type": "llm_call",
            "metadata": {
                "token_usage": {
                    "available": True,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                },
                "estimated_cost": {
                    "available": True,
                    "amount_usd": "0.000010000000",
                    "pricing_version": "test-version",
                },
            },
        },
        {
            "event_type": "llm_call",
            "metadata": {
                "token_usage": {
                    "available": True,
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                }
            },
        },
    ]

    summary = dashboard._execution_usage_summary(events)

    assert summary["input_tokens"] == 15
    assert summary["output_tokens"] == 7
    assert summary["total_tokens"] == 22
    assert summary["reported_call_count"] == 2
    assert summary["unavailable_call_count"] == 0


def test_execution_cost_aggregates_only_complete_estimates() -> None:
    events = [
        {
            "event_type": "llm_call",
            "metadata": {
                "estimated_cost": {
                    "available": True,
                    "amount_usd": "0.001250000000",
                    "pricing_version": "pricing-v1",
                }
            },
        },
        {
            "event_type": "llm_call",
            "metadata": {
                "estimated_cost": {
                    "available": True,
                    "amount_usd": "0.000750000000",
                    "pricing_version": "pricing-v2",
                }
            },
        },
    ]

    estimate = dashboard._execution_cost_summary(events)

    assert estimate == {
        "available": True,
        "amount_usd": "0.002000000000",
        "pricing_versions": ["pricing-v1", "pricing-v2"],
    }
    assert dashboard._execution_cost_summary(events + [{"event_type": "llm_call"}])[
        "available"
    ] is False


def test_execution_usage_sums_complete_calls_and_renders_latency_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    responses = execution_detail_responses(run_id)
    responses[f"/api/v1/executions/{run_id}/events"]["events"] = [
        {
            "event_type": "llm_call",
            "timestamp": "2026-11-09T19:42:35+05:30",
            "status": "success",
            "provider": "Groq",
            "model": "configured-model",
            "duration_ms": 120,
            "metadata": {
                "token_usage": {
                    "available": True,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                },
                "estimated_cost": {
                    "available": True,
                    "amount_usd": "0.000010000000",
                    "pricing_version": "test-version",
                },
            },
        },
        {
            "event_type": "llm_call",
            "timestamp": "2026-11-09T19:42:36+05:30",
            "status": "success",
            "provider": "OpenAI",
            "model": "openai/model",
            "duration_ms": 80,
            "metadata": {
                "token_usage": {
                    "available": True,
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                }
            },
        },
        {
            "event_type": "run_completed",
            "timestamp": "2026-11-09T19:42:37+05:30",
            "status": "success",
            "duration_ms": 2000,
        },
    ]
    ui = install_fake_ui(monkeypatch)

    dashboard._render_execution_details(MockAPIClient(responses), run_id)

    llm_call_table = next(
        dataframe
        for dataframe in ui.dataframes
        if dataframe and "Call" in dataframe[0]
    )
    assert llm_call_table[0]["Input tokens"] == "10"
    assert llm_call_table[0]["Total tokens"] == "14"
    assert llm_call_table[0]["LLM latency"] == "0.12 seconds"
    assert llm_call_table[0]["Estimated cost"] == "$0.00001000 USD"
    assert llm_call_table[0]["Pricing version"] == "test-version"
    assert llm_call_table[1]["Provider / model"] == "OpenAI / openai/model"
    assert llm_call_table[1]["Estimated cost"] == "Unavailable"
    assert ("Estimated execution cost", "Unavailable") in ui.metrics
    assert "15" in ui.messages
    assert "7" in ui.messages
    assert "22" in ui.messages
    assert "2 (2 reported, 0 unavailable)" in ui.messages
    assert "Cumulative LLM latency: 0.20 seconds" in ui.messages
    assert "**Total execution latency**" in ui.messages
    assert "2.00 seconds" in ui.messages
    assert any("model provider" in message for message in ui.messages)


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [(1967.4, "1.97 seconds"), (None, "—")],
)
def test_duration_ms_formats_as_concise_seconds(
    duration_ms: float | None,
    expected: str,
) -> None:
    assert dashboard._format_duration_ms(duration_ms) == expected


def test_execution_event_table_displays_full_timestamp_in_wide_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = str(uuid4())
    responses = execution_detail_responses(run_id)
    responses[f"/api/v1/executions/{run_id}/events"]["events"][0]["timestamp"] = (
        "2026-11-09T19:42:35+05:30"
    )
    ui = install_fake_ui(monkeypatch)

    dashboard._render_execution_details(
        MockAPIClient(responses),
        run_id,
    )

    event_table = next(
        dataframe
        for dataframe in ui.dataframes
        if dataframe and "Timestamp" in dataframe[0]
    )
    timestamp = event_table[0]["Timestamp"]
    expected_local_time = datetime.fromisoformat(
        "2026-11-09T19:42:35+05:30"
    ).astimezone()
    assert timestamp.startswith(
        expected_local_time.strftime("%d %b %Y, %H:%M:%S")
    )
    assert "UTC" in timestamp
    event_table_options = next(
        options
        for dataframe, options in zip(ui.dataframes, ui.dataframe_options)
        if dataframe is event_table
    )
    assert (
        event_table_options["column_config"]["Timestamp"]["width"] == "large"
    )


def test_dashboard_navigation_has_descriptions_for_six_pages() -> None:
    assert dashboard._NAVIGATION_ITEMS == (
        "Overview",
        "Executions",
        "Failures",
        "Evaluations",
        "Diagnostics",
        "Agent Playground",
    )
    assert set(dashboard._PAGE_DESCRIPTIONS) == set(dashboard._NAVIGATION_ITEMS)
    assert all(
        description.strip()
        for description in dashboard._PAGE_DESCRIPTIONS.values()
    )


def test_diagnostics_page_handles_empty_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(monkeypatch)
    client = MockAPIClient(
        {"/api/v1/diagnostics": make_list_response([])}
    )

    dashboard._render_diagnostics(client)

    assert ("Total diagnostic reports", "0") in ui.metrics
    assert any("No diagnostic reports have been recorded" in msg for msg in ui.messages)
    assert not ui.dataframes


def test_diagnostics_page_loads_reports_and_global_status_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = make_report(status="healthy")
    second = make_report(status="warning", findings=[make_finding()])
    third = make_report(status="critical")
    reports = [first, second, third]
    ui = install_fake_ui(monkeypatch)
    client = MockAPIClient(
        {
            "/api/v1/diagnostics": make_list_response(reports),
            f"/api/v1/diagnostics/{first['report_id']}": first,
        }
    )

    dashboard._render_diagnostics(client)

    assert ("Total diagnostic reports", "3") in ui.metrics
    assert ("Healthy reports", "1") in ui.metrics
    assert ("Warning reports", "1") in ui.metrics
    assert ("Critical reports", "1") in ui.metrics
    assert ui.dataframes[0][0]["Report ID"] == first["report_id"]
    assert (
        f"/api/v1/diagnostics/{first['report_id']}",
        None,
    ) in client.requests


def test_diagnostics_status_filter_and_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = [
        make_report(status="healthy") for _ in range(12)
    ]
    reports.extend(make_report(status="warning") for _ in range(2))
    ui = install_fake_ui(
        monkeypatch,
        selections={"Report status": "Healthy", "Reports per page": 10},
    )
    ui.session_state[dashboard._DIAGNOSTICS_PAGE_KEY] = 1
    client = MockAPIClient(
        {
            "/api/v1/diagnostics": make_list_response(reports),
            f"/api/v1/diagnostics/{reports[10]['report_id']}": reports[10],
        }
    )

    dashboard._render_diagnostics(client)

    displayed_rows = ui.dataframes[0]
    assert len(displayed_rows) == 2
    assert all(row["Status"] == "healthy" for row in displayed_rows)
    assert ui.session_state[dashboard._DIAGNOSTICS_PAGE_KEY] == 1
    assert any("12 matching" in msg for msg in ui.messages)


def test_diagnostics_report_detail_renders_findings_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "gsk_1234567890abcdefghijklmnop"
    report = make_report(
        summary=f"Investigated api_key={secret}",
        findings=[
            make_finding(
                title="Finding with secret",
                description=f"Observed token={secret}",
                evidence=[f"authorization: Bearer {secret}"],
                recommendation=f"Rotate secret {secret}",
                metadata={"api_key": secret},
            )
        ],
    )
    ui = install_fake_ui(monkeypatch)
    client = MockAPIClient(
        {
            "/api/v1/diagnostics": make_list_response([report]),
            f"/api/v1/diagnostics/{report['report_id']}": report,
        }
    )

    dashboard._render_diagnostics(client)

    rendered = repr((ui.messages, ui.expanders, ui.dataframes))
    assert "Finding 1:" in rendered
    assert "Recommendation" in rendered
    assert secret not in rendered
    assert "[REDACTED]" in rendered


def test_diagnostics_page_displays_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(monkeypatch)
    client = MockAPIClient(
        errors={
            "/api/v1/diagnostics": APIClientError(
                "The backend is unavailable. Check that the API server is running."
            )
        }
    )

    dashboard._render_diagnostics(client)

    assert any("Diagnostic reports are unavailable" in msg for msg in ui.messages)
    assert not ui.dataframes


@pytest.mark.parametrize(
    "response",
    [
        {"items": [], "total": "invalid", "limit": 100, "offset": 0},
        {"items": [{"report_id": "not-a-report"}], "total": 1, "limit": 100, "offset": 0},
    ],
)
def test_diagnostics_page_handles_malformed_responses(
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
) -> None:
    ui = install_fake_ui(monkeypatch)
    client = MockAPIClient({"/api/v1/diagnostics": response})

    dashboard._render_diagnostics(client)

    assert any("unexpected format" in msg for msg in ui.messages)
    assert not ui.dataframes


def test_execution_diagnostics_lookup_handles_reports_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = make_report()
    run_id = report["run_id"]
    ui = install_fake_ui(
        monkeypatch,
        buttons={"Load execution diagnostic reports"},
        text_inputs={"diagnostic_run_id_input": run_id},
    )
    client = MockAPIClient(
        {
            "/api/v1/diagnostics": make_list_response([]),
            f"/api/v1/executions/{run_id}/diagnostics": [report],
            f"/api/v1/diagnostics/{report['report_id']}": report,
        }
    )

    dashboard._render_diagnostics(client)

    assert any(
        path == f"/api/v1/executions/{run_id}/diagnostics"
        for path, _ in client.requests
    )
    assert any("Report ID" in row for row in ui.dataframes[0])


def test_diagnostics_report_detail_handles_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = make_report()
    ui = install_fake_ui(monkeypatch)
    client = MockAPIClient(
        {f"/api/v1/diagnostics/{report['report_id']}": {"summary": "incomplete"}}
    )

    dashboard._render_diagnostic_report_details(client, report["report_id"])

    assert any("details in an unexpected format" in msg for msg in ui.messages)


@pytest.mark.parametrize(
    "key",
    ["token", "refresh_token", "accessToken"],
)
def test_redacts_generic_and_variant_token_dictionary_keys(key: str) -> None:
    secret = "sensitive-value-that-must-not-be-displayed"

    redacted = dashboard._redact_payload({key: secret})

    assert redacted == {key: "[REDACTED]"}


def test_redacts_nested_credential_values() -> None:
    payload = {
        "request": {
            "credentials": {
                "username": "agent",
                "password": "nested-password",
            },
            "headers": {"Authorization": "Bearer nested-bearer-secret"},
        }
    }

    redacted = dashboard._redact_payload(payload)

    assert redacted["request"]["credentials"] == "[REDACTED]"
    assert redacted["request"]["headers"]["Authorization"] == "[REDACTED]"


def test_redacts_groq_api_keys_and_common_text_credentials() -> None:
    groq_key = "gsk_1234567890abcdefghijklmnop"
    api_key = "generic-api-key-value"
    payload = {
        "GROQ_API_KEY": groq_key,
        "details": (
            f"api_key={api_key}; Authorization: Bearer {groq_key}; "
            f"refresh_token: '{api_key}'; accessToken={api_key}"
        ),
    }

    rendered = dashboard._format_event_payload(payload)

    assert groq_key not in rendered
    assert api_key not in rendered
    assert rendered.count("[REDACTED]") >= 4


def make_overview_responses(
    executions: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build collection responses used by the Overview dashboard."""
    normalized_executions = []
    responses: dict[str, dict[str, Any]] = {}
    for index, execution in enumerate(executions):
        run_id = execution.get("run_id") or str(uuid4())
        normalized_execution = {
            "run_id": run_id,
            "agent_name": "overview-test-agent",
            "started_at": f"2026-11-{index + 1:02d}T10:00:00+00:00",
            "duration_ms": 1000 if execution.get("status") != "running" else None,
            **execution,
        }
        normalized_executions.append(normalized_execution)
        responses[f"/api/v1/executions/{run_id}/events"] = {
            "run_id": run_id,
            "events": [],
        }
    responses.update({
        "/api/v1/executions": {
            "items": normalized_executions,
            "total": len(normalized_executions),
            "limit": 100,
            "offset": 0,
        },
        "/api/v1/evaluations": {
            "items": evaluations,
            "total": len(evaluations),
            "limit": 100,
            "offset": 0,
        },
        "/api/v1/diagnostics": make_list_response(reports),
    })
    return responses


def test_overview_charts_show_execution_and_evaluation_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(monkeypatch)
    executions = [
        {"status": "completed"},
        {"status": "completed"},
        {"status": "failed"},
        {"status": "running"},
        {"status": "unknown"},
    ]
    evaluations = [
        {
            "overall_score": 80,
            "correctness_score": 0.8,
            "relevance_score": None,
            "context_relevance_score": 0.6,
            "hallucination_score": 0.2,
        },
        {
            "overall_score": 100,
            "correctness_score": 0.6,
            "relevance_score": 0.8,
            "context_relevance_score": True,
            "hallucination_score": "invalid",
        },
        {
            "overall_score": False,
            "correctness_score": "invalid",
            "relevance_score": None,
            "context_relevance_score": None,
            "hallucination_score": None,
        },
    ]
    reports = [
        make_report(status="healthy"),
        make_report(status="warning"),
        make_report(status="warning"),
        make_report(status="critical"),
    ]
    client = MockAPIClient(
        make_overview_responses(executions, evaluations, reports)
    )

    dashboard._render_overview(client)

    assert ui.charts[:3] == [
        (
            [
                {"Status": "Completed", "Executions": 2},
                {"Status": "Failed", "Executions": 1},
                {"Status": "Running", "Executions": 1},
            ],
            {
                "x": "Status",
                "y": "Executions",
                "height": 260,
                "alt": "Counts of completed, failed, and running executions",
            },
        ),
        (
            [
                {"Metric": "Overall score (0–100)", "Average score": 90},
                {"Metric": "Correctness (0–1)", "Average score": 0.7},
                {"Metric": "Relevance (0–1)", "Average score": 0.8},
                {"Metric": "Context relevance (0–1)", "Average score": 0.6},
                {"Metric": "Hallucination (0–1)", "Average score": 0.2},
            ],
            {
                "x": "Metric",
                "y": "Average score",
                "height": 260,
                "alt": "Average overall and component evaluation scores",
            },
        ),
        (
            [
                {"Status": "Healthy", "Reports": 1},
                {"Status": "Warning", "Reports": 2},
                {"Status": "Critical", "Reports": 1},
            ],
            {
                "x": "Status",
                "y": "Reports",
                "height": 260,
                "alt": "Counts of healthy, warning, and critical diagnostic reports",
            },
        ),
    ]
    diagnostic_requests = [
        request
        for request in client.requests
        if request[0] == "/api/v1/diagnostics"
    ]
    assert diagnostic_requests == [
        ("/api/v1/diagnostics", {"limit": 100, "offset": 0})
    ]
    assert len(
        [request for request in client.requests if request[0].endswith("/events")]
    ) == len(executions)


def test_overview_renders_persisted_usage_cost_and_model_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(monkeypatch)
    execution = {
        "run_id": str(uuid4()),
        "agent_name": "usage-test-agent",
        "status": "completed",
        "started_at": "2026-11-09T10:00:00+00:00",
        "duration_ms": 1500,
    }
    responses = make_overview_responses(
        [execution],
        [],
        [],
    )
    run_id = execution["run_id"]
    responses[f"/api/v1/executions/{run_id}/events"]["events"] = [
        {
            "event_type": "llm_call",
            "provider": "Anthropic",
            "model": "anthropic/claude-sonnet-5",
            "duration_ms": 500,
            "metadata": {
                "token_usage": {
                    "available": True,
                    "input_tokens": 1_000,
                    "output_tokens": 500,
                    "total_tokens": 1_500,
                },
                "estimated_cost": {
                    "available": True,
                    "amount_usd": "0.007000000000",
                    "currency": "USD",
                    "pricing_version": "2026-11-09-v1",
                },
            },
        }
    ]

    dashboard._render_overview(MockAPIClient(responses))

    assert ("Total input tokens", "1,000") in ui.metrics
    assert ("Total output tokens", "500") in ui.metrics
    assert ("Total tokens", "1,500") in ui.metrics
    assert ("Estimated total cost", "$0.00700000 USD") in ui.metrics
    assert any("Token consumption trend" in message for message in ui.messages)
    assert any("Usage by provider and model" in message for message in ui.messages)
    model_usage = next(
        dataframe
        for dataframe in ui.dataframes
        if dataframe and "Provider / model" in dataframe[0]
    )
    assert model_usage[0]["Provider / model"] == (
        "Anthropic / anthropic/claude-sonnet-5"
    )
    assert model_usage[0]["Estimated cost"] == "$0.00700000 USD"


def test_usage_analytics_loads_events_for_every_execution_page() -> None:
    executions = [
        {
            "run_id": str(uuid4()),
            "status": "completed",
            "started_at": "2026-11-09T10:00:00+00:00",
            "duration_ms": 1000,
        }
        for _ in range(101)
    ]

    class PagedExecutionClient:
        def __init__(self) -> None:
            self.execution_offsets: list[int] = []
            self.event_requests: list[str] = []

        def get(
            self,
            path: str,
            *,
            params: dict[str, Any] | None = None,
        ) -> Any:
            if path == "/api/v1/executions":
                assert params is not None
                offset = params["offset"]
                self.execution_offsets.append(offset)
                return {
                    "items": executions[offset : offset + params["limit"]],
                    "total": len(executions),
                }
            run_id = path.removeprefix("/api/v1/executions/").removesuffix("/events")
            self.event_requests.append(run_id)
            return {"run_id": run_id, "events": []}

    client = PagedExecutionClient()
    all_executions, total = dashboard._fetch_all_items(
        client,
        "/api/v1/executions",
    )
    analytics = dashboard._fetch_usage_analytics(client, all_executions)

    assert total == 101
    assert len(all_executions) == 101
    assert client.execution_offsets == [0, 100]
    assert len(client.event_requests) == 101
    assert analytics["execution_count"] == 101


def test_usage_analytics_computes_average_and_p95_with_twenty_durations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(monkeypatch)
    executions = [
        {
            "run_id": str(uuid4()),
            "status": "completed",
            "started_at": f"2026-11-{index:02d}T10:00:00+00:00",
            "duration_ms": index * 100,
        }
        for index in range(1, 21)
    ]

    analytics = dashboard._aggregate_usage_analytics(executions, {})

    assert analytics["average_latency_ms"] == 1050
    assert analytics["p95_latency_ms"] == 1900
    assert analytics["latency_sample_count"] == 20
    dashboard._render_usage_analytics(analytics)
    assert ("P95 execution latency", "1.90 seconds") in ui.metrics


def test_usage_analytics_identifies_relative_token_and_latency_outliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(monkeypatch)
    executions = []
    events_by_run: dict[str, list[dict[str, Any]]] = {}
    for index, (tokens, duration) in enumerate(
        ((10, 100), (20, 200), (100, 1000)),
        start=1,
    ):
        run_id = str(uuid4())
        executions.append(
            {
                "run_id": run_id,
                "status": "completed",
                "started_at": f"2026-11-0{index}T10:00:00+00:00",
                "duration_ms": duration,
            }
        )
        events_by_run[run_id] = [
            {
                "event_type": "llm_call",
                "provider": "Anthropic",
                "model": "anthropic/claude-sonnet-5",
                "metadata": {
                    "token_usage": {
                        "available": True,
                        "input_tokens": tokens // 2,
                        "output_tokens": tokens - tokens // 2,
                        "total_tokens": tokens,
                    },
                    "estimated_cost": {
                        "available": False,
                        "amount_usd": None,
                    },
                },
            }
        ]

    analytics = dashboard._aggregate_usage_analytics(executions, events_by_run)

    dashboard._render_usage_analytics(analytics)

    outliers = next(
        dataframe
        for dataframe in ui.dataframes
        if dataframe and "Observation" in dataframe[0]
    )
    assert {row["Observation"] for row in outliers} == {
        "High token consumption",
        "High execution latency",
    }
    assert any(row["Run ID"] == executions[2]["run_id"] for row in outliers)
    assert any("do not measure optimization savings" in message for message in ui.messages)


def test_overview_charts_handle_empty_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(monkeypatch)
    client = MockAPIClient(make_overview_responses([], [], []))

    dashboard._render_overview(client)

    assert not ui.charts
    assert any("No execution data" in message for message in ui.messages)
    assert any("No evaluation data" in message for message in ui.messages)
    assert any("No diagnostic data" in message for message in ui.messages)


def test_overview_omits_charts_when_api_data_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui = install_fake_ui(monkeypatch)
    unavailable = APIClientError("Backend unavailable.")
    client = MockAPIClient(
        errors={
            "/api/v1/executions": unavailable,
            "/api/v1/evaluations": unavailable,
            "/api/v1/diagnostics": unavailable,
        }
    )

    dashboard._render_overview(client)

    assert not ui.charts
    for section in ("Execution", "Evaluation", "Diagnostic"):
        assert any(
            f"{section} data is unavailable" in message
            for message in ui.messages
        )
