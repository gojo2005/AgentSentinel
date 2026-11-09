"""Streamlit entry point for the AgentSentinel monitoring dashboard."""

import json
import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from statistics import median
from typing import Any
from uuid import UUID

import streamlit as st

from app.dashboard.api_client import APIClientError, AgentSentinelAPIClient
from app.core.sanitization import redact_sensitive_data

_NAVIGATION_ITEMS = (
    "Agent Playground",
    "Overview",
    "Executions",
    "Failures",
    "Evaluations",
    "Diagnostics",
)
_PAGE_DESCRIPTIONS = {
    "Overview": (
        "A quick view of agent runs, quality scores, and diagnostic health. "
        "Scores summarize persisted evaluations; reports summarize identified issues."
    ),
    "Executions": (
        "Review recorded agent runs and their events. A run is one complete "
        "attempt to answer a question."
    ),
    "Failures": (
        "Focus on runs that did not complete successfully. Failure counts are "
        "limited to the execution page currently loaded."
    ),
    "Evaluations": (
        "Review quality scores and evaluator feedback. Overall scores range "
        "from 0–100; component scores range from 0–1."
    ),
    "Diagnostics": (
        "Review reports that summarize run health, evidence, and suggested next steps."
    ),
    "Agent Playground": (
        "Ask the existing AgentSentinel agent a question, view its answer, "
        "and inspect the recorded run."
    ),
}
_MAX_PAYLOAD_LENGTH = 4_000
_PLAYGROUND_QUESTION_KEY = "playground_question"
_PLAYGROUND_RESULT_KEY = "playground_last_result"
_PLAYGROUND_EXECUTION_KEY = "playground_last_execution"
_PLAYGROUND_DETAIL_ERROR_KEY = "playground_detail_error"
_NAVIGATION_KEY = "dashboard_navigation"
_EXECUTIONS_PAGE_KEY = "executions_page"
_EXECUTIONS_STATUS_KEY = "executions_status_filter"
_EXECUTIONS_PAGE_SIZE_KEY = "executions_page_size"
_FAILURES_PAGE_KEY = "failures_page"
_FAILURES_PAGE_SIZE_KEY = "failures_page_size"
_EVALUATIONS_PAGE_KEY = "evaluations_page"
_EVALUATIONS_PAGE_SIZE_KEY = "evaluations_page_size"
_EVALUATIONS_STATUS_KEY = "evaluations_status_filter"
_EVALUATIONS_EVALUATOR_KEY = "evaluations_evaluator_filter"
_DIAGNOSTICS_PAGE_KEY = "diagnostics_page"
_DIAGNOSTICS_PAGE_SIZE_KEY = "diagnostics_page_size"
_DIAGNOSTICS_STATUS_KEY = "diagnostics_status_filter"
_DIAGNOSTIC_STATUSES = ("healthy", "warning", "critical")


def _read_collection_page(
    client: AgentSentinelAPIClient,
    path: str,
    *,
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch and validate one paginated collection response."""
    payload = client.get(path, params={"limit": limit, "offset": offset})
    if not isinstance(payload, dict):
        raise APIClientError("The backend returned an unexpected format.")

    items = payload.get("items")
    total = payload.get("total")
    if (
        not isinstance(items, list)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or not all(isinstance(item, dict) for item in items)
    ):
        raise APIClientError("The backend returned an unexpected format.")
    return items, total


def _fetch_all_items(
    client: AgentSentinelAPIClient,
    path: str,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch every page so dashboard aggregates use the full dataset."""
    page_size = 100
    items: list[dict[str, Any]] = []
    total: int | None = None

    while total is None or len(items) < total:
        page, page_total = _read_collection_page(
            client,
            path,
            limit=page_size,
            offset=len(items),
        )
        if total is None:
            total = page_total
        elif page_total != total:
            raise APIClientError(
                "The backend data changed while it was being loaded. Refresh and try again."
            )

        if not page and len(items) < total:
            raise APIClientError(
                "The backend returned incomplete paginated data."
            )
        items.extend(page)

    if total is None:
        raise APIClientError("The backend did not return collection metadata.")
    return items, total


def _format_number(value: int) -> str:
    """Format an integer metric with thousands separators."""
    return f"{value:,}"


def _format_duration_ms(value: Any) -> str:
    """Format a duration in milliseconds as concise elapsed seconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{value / 1000:.2f} seconds"


def _format_token_count(value: Any) -> str:
    """Format a known token count, leaving unavailable values explicit."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return "Unavailable"
    return f"{value:,}"


def _format_estimated_cost(cost: Any) -> str:
    """Render a recorded USD estimate without implying unavailable data is zero."""
    if not isinstance(cost, dict) or cost.get("available") is not True:
        return "Unavailable"
    amount = cost.get("amount_usd")
    try:
        decimal_amount = Decimal(str(amount))
    except (InvalidOperation, ValueError):
        return "Unavailable"
    if not decimal_amount.is_finite() or decimal_amount < 0:
        return "Unavailable"
    return f"${decimal_amount:.8f} USD"


def _execution_usage_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only token fields that are known for every LLM call."""
    llm_events = [
        event for event in events if event.get("event_type") == "llm_call"
    ]
    usage_records = []
    for event in llm_events:
        metadata = event.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        usage = metadata.get("token_usage")
        usage_records.append(usage if isinstance(usage, dict) else {})

    totals: dict[str, int | None] = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        counts = [usage.get(field) for usage in usage_records]
        if not counts:
            totals[field] = 0
        elif all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for count in counts
        ):
            totals[field] = sum(counts)
        else:
            totals[field] = None

    durations = [event.get("duration_ms") for event in llm_events]
    llm_duration_ms = (
        sum(durations)
        if durations
        and all(
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and duration >= 0
            for duration in durations
        )
        else None
    )
    reported_calls = sum(usage.get("available") is True for usage in usage_records)

    return {
        **totals,
        "llm_call_count": len(llm_events),
        "reported_call_count": reported_calls,
        "unavailable_call_count": len(llm_events) - reported_calls,
        "llm_latency_ms": llm_duration_ms,
        "llm_events": llm_events,
    }


def _execution_cost_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum persisted call estimates only when every LLM call has an estimate."""
    llm_events = [
        event for event in events if event.get("event_type") == "llm_call"
    ]
    if not llm_events:
        return {"available": True, "amount_usd": "0", "pricing_versions": []}

    total = Decimal(0)
    versions: set[str] = set()
    for event in llm_events:
        metadata = event.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        estimate = metadata.get("estimated_cost")
        if not isinstance(estimate, dict) or estimate.get("available") is not True:
            return {"available": False, "amount_usd": None, "pricing_versions": []}
        try:
            amount = Decimal(str(estimate.get("amount_usd")))
        except (InvalidOperation, ValueError):
            return {"available": False, "amount_usd": None, "pricing_versions": []}
        if not amount.is_finite() or amount < 0:
            return {"available": False, "amount_usd": None, "pricing_versions": []}
        version = estimate.get("pricing_version")
        if not isinstance(version, str) or not version:
            return {"available": False, "amount_usd": None, "pricing_versions": []}
        total += amount
        versions.add(version)

    return {
        "available": True,
        "amount_usd": format(total, "f"),
        "pricing_versions": sorted(versions),
    }


def _execution_date(value: Any) -> str | None:
    """Normalize an execution timestamp into its UTC calendar date."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()


def _aggregate_usage_analytics(
    executions: list[dict[str, Any]],
    events_by_run: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate a complete execution set and its complete event lists."""
    rows: list[dict[str, Any]] = []
    daily_tokens: dict[str, int] = {}
    daily_latency: dict[str, list[float]] = {}
    model_usage: dict[str, dict[str, Any]] = {}
    durations: list[float] = []
    input_total = output_total = total_tokens = 0
    token_totals_available = True
    total_cost = Decimal(0)
    total_cost_available = True
    unavailable_usage_calls = 0
    known_token_runs: list[int] = []

    for execution in executions:
        run_id = execution.get("run_id")
        run_events = events_by_run.get(str(run_id), [])
        usage = _execution_usage_summary(run_events)
        cost = _execution_cost_summary(run_events)
        row = {
            "run_id": run_id,
            "agent_name": execution.get("agent_name"),
            "started_at": execution.get("started_at"),
            "status": execution.get("status"),
            "duration_ms": execution.get("duration_ms"),
            "usage": usage,
            "cost": cost,
        }
        rows.append(row)
        unavailable_usage_calls += usage["unavailable_call_count"]
        counts = (
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
        )
        if all(isinstance(value, int) for value in counts):
            input_total += counts[0]
            output_total += counts[1]
            total_tokens += counts[2]
            known_token_runs.append(counts[2])
            day = _execution_date(execution.get("started_at"))
            if day is not None:
                daily_tokens[day] = daily_tokens.get(day, 0) + counts[2]
        else:
            token_totals_available = False

        duration = execution.get("duration_ms")
        if (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and duration >= 0
            and execution.get("status") in {"completed", "failed"}
        ):
            durations.append(float(duration))
            day = _execution_date(execution.get("started_at"))
            if day is not None:
                daily_latency.setdefault(day, []).append(float(duration))

        if cost["available"]:
            total_cost += Decimal(cost["amount_usd"])
        else:
            total_cost_available = False

        for event in usage["llm_events"]:
            provider_model = " / ".join(
                str(event[field])
                for field in ("provider", "model")
                if event.get(field)
            ) or "Unknown model"
            bucket = model_usage.setdefault(
                provider_model,
                {
                    "Provider / model": provider_model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                    "input_unavailable": False,
                    "output_unavailable": False,
                    "total_unavailable": False,
                    "estimated_cost": Decimal(0),
                    "unavailable_cost_calls": 0,
                },
            )
            bucket["calls"] += 1
            metadata = event.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            token_usage = metadata.get("token_usage")
            token_usage = token_usage if isinstance(token_usage, dict) else {}
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                field_valid = (
                    isinstance(token_usage.get(field), int)
                    and not isinstance(token_usage.get(field), bool)
                    and token_usage[field] >= 0
                )
                if field_valid:
                    bucket[field] += token_usage[field]
                else:
                    bucket[field.replace("_tokens", "_unavailable")] = True
            estimate = metadata.get("estimated_cost")
            if isinstance(estimate, dict) and estimate.get("available") is True:
                try:
                    bucket["estimated_cost"] += Decimal(
                        str(estimate.get("amount_usd"))
                    )
                except (InvalidOperation, ValueError):
                    bucket["unavailable_cost_calls"] += 1
            else:
                bucket["unavailable_cost_calls"] += 1

    formatted_model_usage = []
    for model_row in sorted(
        model_usage.values(),
        key=lambda row: (row["total_unavailable"], -row["total_tokens"]),
    ):
        formatted_model_usage.append(
            {
                "Provider / model": model_row["Provider / model"],
                "Calls": model_row["calls"],
                "Input tokens": (
                    _format_number(model_row["input_tokens"])
                    if not model_row["input_unavailable"]
                    else "Unavailable"
                ),
                "Output tokens": (
                    _format_number(model_row["output_tokens"])
                    if not model_row["output_unavailable"]
                    else "Unavailable"
                ),
                "Total tokens": (
                    _format_number(model_row["total_tokens"])
                    if not model_row["total_unavailable"]
                    else "Unavailable"
                ),
                "Estimated cost": _format_estimated_cost(
                    {
                        "available": model_row["unavailable_cost_calls"] == 0,
                        "amount_usd": format(
                            model_row["estimated_cost"],
                            "f",
                        ),
                    }
                ),
            }
        )

    known_durations = sorted(durations)
    p95_duration = (
        known_durations[math.ceil(0.95 * len(known_durations)) - 1]
        if len(known_durations) >= 20
        else None
    )
    return {
        "executions": rows,
        "execution_count": len(executions),
        "input_tokens": input_total if token_totals_available else None,
        "output_tokens": output_total if token_totals_available else None,
        "total_tokens": total_tokens if token_totals_available else None,
        "total_cost_usd": format(total_cost, "f") if total_cost_available else None,
        "average_latency_ms": (
            sum(known_durations) / len(known_durations) if known_durations else None
        ),
        "p95_latency_ms": p95_duration,
        "latency_sample_count": len(known_durations),
        "latency_sample_total": len(executions),
        "daily_tokens": [
            {"Date": day, "Tokens": count}
            for day, count in sorted(daily_tokens.items())
        ],
        "daily_latency": [
            {
                "Date": day,
                "Average latency (ms)": sum(day_durations) / len(day_durations),
            }
            for day, day_durations in sorted(daily_latency.items())
        ],
        "model_usage": formatted_model_usage,
        "unavailable_usage_calls": unavailable_usage_calls,
        "llm_call_count": sum(row["usage"]["llm_call_count"] for row in rows),
        "reported_usage_calls": sum(
            row["usage"]["reported_call_count"] for row in rows
        ),
        "known_token_runs": known_token_runs,
        "token_sample_count": len(known_token_runs),
    }


def _fetch_usage_analytics(
    client: AgentSentinelAPIClient,
    executions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load each execution's complete persisted events for usage analysis."""
    events_by_run: dict[str, list[dict[str, Any]]] = {}
    for execution in executions:
        run_id = execution.get("run_id")
        if not isinstance(run_id, str) or not _is_uuid_value(run_id):
            raise APIClientError("The backend returned an invalid execution ID.")
        event_response = client.get(f"/api/v1/executions/{run_id}/events")
        if (
            not isinstance(event_response, dict)
            or not isinstance(event_response.get("events"), list)
            or not all(isinstance(event, dict) for event in event_response["events"])
        ):
            raise APIClientError("The backend returned incomplete execution events.")
        if event_response.get("run_id") != run_id:
            raise APIClientError("The backend returned events for a different execution.")
        events_by_run[run_id] = event_response["events"]
    return _aggregate_usage_analytics(executions, events_by_run)


def _format_local_datetime(value: Any) -> str:
    """Convert an ISO timestamp into the dashboard server's local time."""
    if not isinstance(value, str) or not value:
        return "—"
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    local_timestamp = timestamp.astimezone()
    offset = local_timestamp.strftime("%z")
    formatted_offset = f"{offset[:3]}:{offset[3:]}" if offset else "+00:00"
    return (
        f"{local_timestamp.strftime('%d %b %Y, %H:%M:%S')} "
        f"UTC{formatted_offset}"
    )


def _redact_payload(value: Any, *, key: str = "") -> Any:
    """Mask secret-like values in event payloads before rendering them."""
    return redact_sensitive_data(value, key=key)


def _format_event_payload(value: Any) -> str:
    """Serialize a redacted payload and cap its displayed size."""
    try:
        payload = json.dumps(
            _redact_payload(value),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        payload = "[Payload could not be displayed safely]"
    if len(payload) > _MAX_PAYLOAD_LENGTH:
        return f"{payload[:_MAX_PAYLOAD_LENGTH]}\n… [truncated]"
    return payload


def _is_agent_run_response(value: Any) -> bool:
    """Validate the existing agent-run response before retaining it in session state."""
    return (
        isinstance(value, dict)
        and _is_uuid_value(value.get("run_id"))
        and isinstance(value.get("question"), str)
        and isinstance(value.get("answer"), str)
        and value.get("status") == "completed"
    )


def _get_playground_execution_details(
    client: AgentSentinelAPIClient,
    run_id: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load persisted status, duration, and events without losing the answer."""
    try:
        detail = client.get(f"/api/v1/executions/{run_id}")
    except APIClientError as error:
        return None, str(error)

    if (
        not isinstance(detail, dict)
        or not isinstance(detail.get("summary"), dict)
        or not isinstance(detail.get("events"), list)
        or not all(isinstance(event, dict) for event in detail["events"])
    ):
        return None, "Execution details were returned in an unexpected format."
    summary = detail["summary"]
    if summary.get("run_id") != run_id:
        return None, "Execution details did not match the returned run ID."
    return detail, None


def _render_playground_result(
    result: dict[str, Any],
    execution: dict[str, Any] | None,
    detail_error: str | None,
) -> None:
    """Display a completed answer and its available persisted run metadata."""
    run_id = result["run_id"]
    st.subheader("Agent response")
    st.write(_redact_payload(result["answer"]))
    st.caption(f"Run ID: {run_id}")
    st.caption(
        "To inspect this run, open Executions in the sidebar and select its run ID."
    )

    summary = execution["summary"] if execution is not None else {}
    status = summary.get("status") or result.get("status") or "Unknown"
    duration = summary.get("duration_ms")
    status_column, duration_column = st.columns(2)
    status_column.metric("Execution status", str(status).title(), border=True)
    duration_column.metric(
        "Duration",
        "—"
        if not isinstance(duration, (int, float)) or isinstance(duration, bool)
        else f"{duration:.0f} ms",
        border=True,
    )

    if detail_error:
        st.info(
            "The answer was returned, but persisted execution details could not "
            f"be loaded: {detail_error}"
        )
        return

    events = execution["events"] if execution is not None else []
    tool_calls = [
        event
        for event in events
        if event.get("event_type") == "tool_call"
    ]
    st.subheader("Tool calls")
    if not tool_calls:
        st.info("No tool calls were recorded for this run.")
        return
    st.dataframe(
        [
            {
                "Tool": event.get("tool_name") or "Unknown",
                "Status": event.get("status") or "Unknown",
                "Duration (ms)": event.get("duration_ms"),
                "Error": _format_event_payload(event["error"])
                if event.get("error")
                else "—",
            }
            for event in tool_calls
        ],
        hide_index=True,
        alt="Tools called during this agent run",
    )


def _render_agent_playground(client: AgentSentinelAPIClient) -> None:
    """Run the existing agent API endpoint and retain its response per session."""
    st.subheader("Ask the AgentSentinel agent")
    st.caption(
        "Each submission starts a new run and records it for later inspection."
    )
    with st.form("agent_playground_form"):
        question = st.text_area(
            "Question",
            key=_PLAYGROUND_QUESTION_KEY,
            placeholder="Ask a question for the agent to answer",
            height=130,
        )
        submitted = st.form_submit_button(
            "Run Agent",
            type="primary",
            icon=":material/play_arrow:",
        )

    if submitted:
        if not question.strip():
            st.warning("Enter a question before running the agent.")
        else:
            try:
                with st.spinner("Running the agent..."):
                    result = client.post(
                        "/api/v1/agent/run",
                        json={"question": question.strip()},
                    )
                    if not _is_agent_run_response(result):
                        raise APIClientError(
                            "The backend returned an unexpected agent response."
                        )
                    execution, detail_error = _get_playground_execution_details(
                        client,
                        result["run_id"],
                    )
                st.session_state[_PLAYGROUND_RESULT_KEY] = result
                st.session_state[_PLAYGROUND_EXECUTION_KEY] = execution
                st.session_state[_PLAYGROUND_DETAIL_ERROR_KEY] = detail_error
                st.success("Agent run completed.")
            except APIClientError as error:
                st.error(f"Agent run could not be completed: {error}")

    result = st.session_state.get(_PLAYGROUND_RESULT_KEY)
    if not _is_agent_run_response(result):
        st.info(
            "No agent run in this session yet. Enter a question above to start one."
        )
        return
    _render_playground_result(
        result,
        st.session_state.get(_PLAYGROUND_EXECUTION_KEY),
        st.session_state.get(_PLAYGROUND_DETAIL_ERROR_KEY),
    )


def _render_execution_details(
    client: AgentSentinelAPIClient,
    run_id: str,
    *,
    highlight_failures: bool = False,
) -> None:
    """Fetch and render one execution summary and its events."""
    try:
        detail = client.get(f"/api/v1/executions/{run_id}")
        events_response = client.get(f"/api/v1/executions/{run_id}/events")
    except APIClientError as error:
        st.error(f"Execution details are unavailable: {error}")
        return

    if (
        not isinstance(detail, dict)
        or not isinstance(detail.get("summary"), dict)
        or not isinstance(events_response, dict)
        or not isinstance(events_response.get("events"), list)
        or not all(isinstance(event, dict) for event in events_response["events"])
    ):
        st.error("The backend returned execution details in an unexpected format.")
        return

    summary = detail["summary"]
    st.subheader("Last Execution Details")
    summary_columns = st.columns(4)
    execution_metrics = (
        ("Status", str(summary.get("status", "Unknown"))),
        ("Started", _format_local_datetime(summary.get("started_at"))),
        ("Completed", _format_local_datetime(summary.get("completed_at"))),
        (
            "Total execution latency",
            _format_duration_ms(summary.get("duration_ms")),
        ),
    )
    for column, (label, value) in zip(summary_columns, execution_metrics):
        with column.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(value)

    events = events_response["events"]
    usage = _execution_usage_summary(events)
    execution_cost = _execution_cost_summary(events)
    st.metric(
        "Estimated execution cost",
        _format_estimated_cost(
            {
                "available": execution_cost["available"],
                "amount_usd": execution_cost["amount_usd"],
            }
        ),
        border=True,
    )
    if execution_cost["pricing_versions"]:
        st.caption(
            "Pricing version(s): "
            + ", ".join(execution_cost["pricing_versions"])
            + ". Estimates use recorded per-call rates."
        )
    else:
        st.caption(
            "Estimated costs are available only for calls with known token "
            "usage and a configured model rate."
        )
    st.caption(
        "Token counts are reported by the model provider; unavailable values "
        "are not estimated."
    )
    usage_columns = st.columns(4)
    usage_metrics = (
        ("Input tokens", _format_token_count(usage["input_tokens"])),
        ("Output tokens", _format_token_count(usage["output_tokens"])),
        ("Total tokens", _format_token_count(usage["total_tokens"])),
        (
            "LLM calls",
            f"{usage['llm_call_count']} "
            f"({usage['reported_call_count']} reported, "
            f"{usage['unavailable_call_count']} unavailable)",
        ),
    )
    for column, (label, value) in zip(usage_columns, usage_metrics):
        with column.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(value)

    st.markdown("**LLM call usage and latency**")
    st.caption(
        "LLM latency is the cumulative duration of model calls; execution "
        "latency above includes the complete run."
    )
    st.caption(
        "Cumulative LLM latency: "
        f"{_format_duration_ms(usage['llm_latency_ms'])}"
    )
    if usage["llm_events"]:
        llm_call_rows = []
        for index, event in enumerate(usage["llm_events"], start=1):
            metadata = event.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            token_usage = metadata.get("token_usage")
            token_usage = token_usage if isinstance(token_usage, dict) else {}
            llm_call_rows.append(
                {
                    "Call": index,
                    "Provider / model": " / ".join(
                        str(event[field])
                        for field in ("provider", "model")
                        if event.get(field)
                    )
                    or "—",
                    "Input tokens": _format_token_count(
                        token_usage.get("input_tokens")
                    ),
                    "Output tokens": _format_token_count(
                        token_usage.get("output_tokens")
                    ),
                    "Total tokens": _format_token_count(
                        token_usage.get("total_tokens")
                    ),
                    "Usage": (
                        "Reported"
                        if token_usage.get("available") is True
                        else "Unavailable"
                    ),
                    "LLM latency": _format_duration_ms(event.get("duration_ms")),
                    "Estimated cost": _format_estimated_cost(
                        metadata.get("estimated_cost")
                    ),
                    "Pricing version": (
                        metadata["estimated_cost"].get("pricing_version", "—")
                        if isinstance(metadata.get("estimated_cost"), dict)
                        else "—"
                    ),
                }
            )
        st.dataframe(
            llm_call_rows,
            hide_index=True,
            alt="Token usage and latency for each LLM call",
        )
    else:
        st.info("This execution has no recorded LLM calls.")

    st.caption(f"{len(events)} event(s), ordered chronologically")
    if not events:
        st.info("This execution has no recorded events.")
    else:
        event_rows = []
        for event in events:
            event_status = event.get("status", "Unknown")
            if highlight_failures and event_status == "failure":
                event_status = "❗ failure"
            event_rows.append(
                {
                    "Event type": event.get("event_type", "Unknown"),
                    "Timestamp": _format_local_datetime(event.get("timestamp")),
                    "Status": event_status,
                    "Provider / model": " / ".join(
                        str(event[field])
                        for field in ("provider", "model")
                        if event.get(field)
                    )
                    or "—",
                    "Tool": event.get("tool_name") or "—",
                    "Duration (ms)": event.get("duration_ms"),
                    "Error": _format_event_payload(event["error"])
                    if event.get("error")
                    else "—",
                }
            )
        st.dataframe(
            event_rows,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn(width="large")
            },
            alt="Events recorded during this execution",
        )

        for index, event in enumerate(events, start=1):
            event_type = event.get("event_type", "Unknown")
            event_timestamp = _format_local_datetime(event.get("timestamp"))
            is_failure = event.get("status") == "failure"
            failure_marker = "❗ " if highlight_failures and is_failure else ""
            with st.expander(
                f"{failure_marker}{index}. {event_type} · {event_timestamp}"
            ):
                if highlight_failures and is_failure:
                    st.error("This event ended in failure.")
                input_column, output_column = st.columns(2)
                with input_column:
                    st.markdown("**Input**")
                    st.code(_format_event_payload(event.get("input")))
                with output_column:
                    st.markdown("**Output**")
                    st.code(_format_event_payload(event.get("output")))

    _render_evaluation_workflow(client, run_id)
    _render_run_evaluations(client, run_id)
    _render_diagnostic_workflow(client, run_id)
    _render_execution_diagnostic_reports(client, run_id)


def _reset_execution_page() -> None:
    """Return execution history to its first page after filter changes."""
    st.session_state[_EXECUTIONS_PAGE_KEY] = 0


def _move_execution_page(offset: int) -> None:
    """Move the execution history page within its available lower bound."""
    current_page = st.session_state.get(_EXECUTIONS_PAGE_KEY, 0)
    if not isinstance(current_page, int):
        current_page = 0
    st.session_state[_EXECUTIONS_PAGE_KEY] = max(0, current_page + offset)


def _reset_failure_page() -> None:
    """Return failed execution history to its first page."""
    st.session_state[_FAILURES_PAGE_KEY] = 0


def _move_failure_page(offset: int) -> None:
    """Move the failed execution history page without going below zero."""
    current_page = st.session_state.get(_FAILURES_PAGE_KEY, 0)
    if not isinstance(current_page, int):
        current_page = 0
    st.session_state[_FAILURES_PAGE_KEY] = max(0, current_page + offset)


def _render_executions(client: AgentSentinelAPIClient) -> None:
    """Render a filterable, paginated execution history and run details."""
    st.subheader("Execution history")
    st.caption(
        "Each row is one agent run. Select a run to review its events, "
        "create an evaluation, or generate a diagnostic report."
    )
    filter_column, page_size_column = st.columns([2, 1])
    selected_status = filter_column.selectbox(
        "Execution status",
        ("All", "Completed", "Failed", "Running"),
        key=_EXECUTIONS_STATUS_KEY,
        on_change=_reset_execution_page,
    )
    page_size = page_size_column.selectbox(
        "Rows per page",
        (10, 20, 50, 100),
        index=1,
        key=_EXECUTIONS_PAGE_SIZE_KEY,
        on_change=_reset_execution_page,
    )

    current_page = st.session_state.get(_EXECUTIONS_PAGE_KEY, 0)
    if not isinstance(current_page, int) or current_page < 0:
        current_page = 0

    while True:
        try:
            executions, total = _read_collection_page(
                client,
                "/api/v1/executions",
                limit=page_size,
                offset=current_page * page_size,
            )
        except APIClientError as error:
            st.error(f"Execution history is unavailable: {error}")
            return

        page_count = (total + page_size - 1) // page_size
        if page_count == 0:
            st.session_state[_EXECUTIONS_PAGE_KEY] = 0
            st.info(
                "No executions have been recorded yet. Run the agent from "
                "Agent Playground to create an execution."
            )
            return
        if current_page < page_count:
            break
        current_page = page_count - 1
        st.session_state[_EXECUTIONS_PAGE_KEY] = current_page

    st.session_state[_EXECUTIONS_PAGE_KEY] = current_page
    status_by_filter = {
        "Completed": "completed",
        "Failed": "failed",
        "Running": "running",
    }
    filtered_executions = (
        executions
        if selected_status == "All"
        else [
            execution
            for execution in executions
            if execution.get("status") == status_by_filter[selected_status]
        ]
    )
    page_executions = filtered_executions
    st.caption(
        f"Showing {len(filtered_executions):,} matching execution(s) on "
        f"page {current_page + 1} of {page_count} ({total:,} total executions). "
        "The status filter applies to this page only."
    )

    previous_column, page_column, next_column = st.columns([1, 2, 1])
    previous_column.button(
        "Previous",
        disabled=current_page == 0,
        on_click=_move_execution_page,
        args=(-1,),
    )
    page_column.caption(f"Page {current_page + 1} of {page_count}")
    next_column.button(
        "Next",
        disabled=current_page >= page_count - 1,
        on_click=_move_execution_page,
        args=(1,),
    )

    if not filtered_executions:
        st.info("No executions on this page match the selected status.")
        return

    table_rows = [
        {
            "Run ID": execution.get("run_id", "Unknown"),
            "Agent name": execution.get("agent_name", "Unknown"),
            "Status": execution.get("status", "Unknown"),
            "Start time": _format_local_datetime(execution.get("started_at")),
            "Completion time": _format_local_datetime(
                execution.get("completed_at")
            ),
            "Duration (ms)": execution.get("duration_ms"),
            "Total events": execution.get("total_events", 0),
        }
        for execution in page_executions
    ]
    st.dataframe(
        table_rows,
        hide_index=True,
        column_config={
            "Start time": st.column_config.TextColumn(width="large"),
            "Completion time": st.column_config.TextColumn(width="large"),
        },
        alt="Agent execution history",
    )

    run_ids = [
        execution["run_id"]
        for execution in page_executions
        if isinstance(execution.get("run_id"), str)
    ]
    if not run_ids:
        st.info("The current page has no selectable execution IDs.")
        return

    selected_run_id = st.selectbox(
        "Inspect an execution",
        run_ids,
        format_func=lambda run_id: next(
            (
                f"{execution.get('agent_name', 'Unknown')} · "
                f"{execution.get('status', 'Unknown')} · {run_id}"
                for execution in page_executions
                if execution.get("run_id") == run_id
            ),
            run_id,
        ),
    )
    _render_execution_details(client, selected_run_id)


def _render_failures(client: AgentSentinelAPIClient) -> None:
    """Render failed runs from one backend-paginated execution page."""
    st.subheader("Failure monitoring")
    st.caption(
        "A failed run is an execution with a recorded failure status. "
        "The count below covers only the loaded page."
    )
    page_size_column, _ = st.columns([1, 2])
    page_size = page_size_column.selectbox(
        "Executions per page",
        (10, 20, 50, 100),
        index=1,
        key=_FAILURES_PAGE_SIZE_KEY,
        on_change=_reset_failure_page,
    )

    current_page = st.session_state.get(_FAILURES_PAGE_KEY, 0)
    if not isinstance(current_page, int) or current_page < 0:
        current_page = 0

    while True:
        try:
            executions, total = _read_collection_page(
                client,
                "/api/v1/executions",
                limit=page_size,
                offset=current_page * page_size,
            )
        except APIClientError as error:
            st.error(f"Failure data is unavailable: {error}")
            return

        page_count = (total + page_size - 1) // page_size
        if page_count == 0:
            st.session_state[_FAILURES_PAGE_KEY] = 0
            st.info(
                "No executions have been recorded yet. Run the agent from "
                "Agent Playground to create one."
            )
            return
        if current_page < page_count:
            break
        current_page = page_count - 1
        st.session_state[_FAILURES_PAGE_KEY] = current_page

    st.session_state[_FAILURES_PAGE_KEY] = current_page
    failed_executions = [
        execution
        for execution in executions
        if execution.get("status") == "failed"
    ]

    metric_column, _ = st.columns([1, 2])
    metric_column.metric(
        "Failed executions in the currently loaded page",
        _format_number(len(failed_executions)),
        border=True,
    )
    st.caption(
        "The executions API does not provide status-filtered totals. "
        "Failure counts therefore apply only to the currently loaded page; "
        "a dataset-wide failure rate is not available."
    )

    previous_column, page_column, next_column = st.columns([1, 2, 1])
    previous_column.button(
        "Previous",
        disabled=current_page == 0,
        on_click=_move_failure_page,
        args=(-1,),
    )
    page_column.caption(f"Page {current_page + 1} of {page_count}")
    next_column.button(
        "Next",
        disabled=current_page >= page_count - 1,
        on_click=_move_failure_page,
        args=(1,),
    )

    if not failed_executions:
        st.info(
            "No failed executions were found on this page. Runs created in "
            "Agent Playground will appear here if they fail."
        )
        return

    table_rows = [
        {
            "Run ID": execution.get("run_id", "Unknown"),
            "Agent name": execution.get("agent_name", "Unknown"),
            "Start time": _format_local_datetime(execution.get("started_at")),
            "Completion time": _format_local_datetime(
                execution.get("completed_at")
            ),
            "Duration (ms)": execution.get("duration_ms"),
            "Total events": execution.get("total_events", 0),
        }
        for execution in failed_executions
    ]
    st.dataframe(
        table_rows,
        hide_index=True,
        column_config={
            "Start time": st.column_config.TextColumn(width="large"),
            "Completion time": st.column_config.TextColumn(width="large"),
        },
        alt="Failed agent executions",
    )

    run_ids = [
        execution["run_id"]
        for execution in failed_executions
        if isinstance(execution.get("run_id"), str)
    ]
    if not run_ids:
        st.info("No failed execution IDs are available to inspect.")
        return

    selected_run_id = st.selectbox(
        "Inspect a failed execution",
        run_ids,
        format_func=lambda run_id: next(
            (
                f"{execution.get('agent_name', 'Unknown')} · {run_id}"
                for execution in failed_executions
                if execution.get("run_id") == run_id
            ),
            run_id,
        ),
    )
    _render_execution_details(
        client,
        selected_run_id,
        highlight_failures=True,
    )


def _reset_evaluation_page() -> None:
    """Return evaluation results to the first page after changing filters."""
    st.session_state[_EVALUATIONS_PAGE_KEY] = 0


def _move_evaluation_page(offset: int) -> None:
    """Move the evaluation page without allowing a negative index."""
    current_page = st.session_state.get(_EVALUATIONS_PAGE_KEY, 0)
    if not isinstance(current_page, int):
        current_page = 0
    st.session_state[_EVALUATIONS_PAGE_KEY] = max(0, current_page + offset)


def _average_metric(
    evaluations: list[dict[str, Any]],
    field: str,
) -> float | None:
    """Average available numeric values for one evaluation score field."""
    values = [
        evaluation[field]
        for evaluation in evaluations
        if isinstance(evaluation.get(field), (int, float))
        and not isinstance(evaluation.get(field), bool)
    ]
    return sum(values) / len(values) if values else None


def _render_evaluation_details(
    client: AgentSentinelAPIClient,
    evaluation_id: str,
) -> None:
    """Fetch and display one evaluation's full scores and feedback."""
    try:
        result = client.get(f"/api/v1/evaluations/{evaluation_id}")
    except APIClientError as error:
        st.error(f"Evaluation details are unavailable: {error}")
        return

    if not _is_evaluation_result(result):
        st.error("The backend returned evaluation details in an unexpected format.")
        return

    _render_evaluation_result(result)


def _render_evaluation_result(result: dict[str, Any]) -> None:
    """Render a validated evaluation response without another API request."""
    evaluation_id = result["evaluation_id"]
    st.subheader("Evaluation details")
    columns = st.columns(4)
    score_fields = (
        ("Overall score", "overall_score", 100),
        ("Correctness", "correctness_score", 1),
        ("Relevance", "relevance_score", 1),
        ("Context relevance", "context_relevance_score", 1),
    )
    for column, (label, field, maximum) in zip(columns, score_fields):
        value = result.get(field)
        column.metric(
            label,
            "—"
            if not isinstance(value, (int, float)) or isinstance(value, bool)
            else f"{value:.2f} / {maximum}",
            border=True,
        )

    extra_columns = st.columns(3)
    hallucination = result.get("hallucination_score")
    extra_columns[0].metric(
        "Hallucination",
        "—"
        if not isinstance(hallucination, (int, float))
        or isinstance(hallucination, bool)
        else f"{hallucination:.2f} / 1",
        border=True,
    )
    passed = result.get("passed")
    extra_columns[1].metric(
        "Result",
        "Passed"
        if passed is True
        else "Failed"
        if passed is False
        else "Unknown",
        border=True,
    )
    extra_columns[2].metric(
        "Evaluator",
        str(result.get("evaluator") or "Unknown"),
        border=True,
    )

    st.caption(f"Evaluation ID: {evaluation_id}")
    run_id = result.get("run_id")
    if isinstance(run_id, str):
        st.caption(f"Run ID: {run_id}")
    st.caption(f"Timestamp: {_format_local_datetime(result.get('timestamp'))}")

    feedback = result.get("feedback")
    st.markdown("**Feedback**")
    st.write(
        _format_event_payload(feedback)
        if isinstance(feedback, str) and feedback
        else "No feedback provided."
    )


def _is_evaluation_result(value: Any) -> bool:
    """Check that an API evaluation has the fields required by its schema."""
    score_fields = (
        "correctness_score",
        "relevance_score",
        "context_relevance_score",
        "hallucination_score",
        "overall_score",
    )
    return (
        isinstance(value, dict)
        and _is_uuid_value(value.get("evaluation_id"))
        and _is_uuid_value(value.get("run_id"))
        and _is_timestamp_value(value.get("timestamp"))
        and isinstance(value.get("evaluator"), str)
        and all(
            value.get(field) is None
            or (
                isinstance(value.get(field), (int, float))
                and not isinstance(value.get(field), bool)
            )
            for field in score_fields
        )
        and (value.get("passed") is None or isinstance(value.get("passed"), bool))
        and (value.get("feedback") is None or isinstance(value.get("feedback"), str))
    )


def _render_evaluation_workflow(
    client: AgentSentinelAPIClient,
    run_id: str,
) -> None:
    """Collect evaluation input and create a persisted evaluation on submit."""
    st.subheader("Generate evaluation")
    with st.form(f"evaluation_form_{run_id}"):
        question = st.text_area(
            "Question",
            key=f"evaluation_question_{run_id}",
        )
        answer = st.text_area(
            "Answer",
            key=f"evaluation_answer_{run_id}",
        )
        reference_answer = st.text_area(
            "Reference answer (optional)",
            key=f"evaluation_reference_answer_{run_id}",
        )
        context = st.text_area(
            "Context (optional)",
            key=f"evaluation_context_{run_id}",
        )
        evaluator = st.selectbox(
            "Evaluator",
            ("Rule-based", "LLM judge"),
            key=f"evaluation_method_{run_id}",
        )
        submitted = st.form_submit_button("Generate Evaluation")

    if not submitted:
        return
    if not question.strip() or not answer.strip():
        st.warning("Enter both a question and an answer to generate an evaluation.")
        return

    payload: dict[str, Any] = {
        "question": question.strip(),
        "answer": answer.strip(),
        "evaluator": "llm_judge" if evaluator == "LLM judge" else "rule_based",
    }
    if reference_answer.strip():
        payload["reference_answer"] = reference_answer.strip()
    if context.strip():
        payload["context"] = context.strip()

    try:
        result = client.post(
            f"/api/v1/executions/{run_id}/evaluations",
            json=payload,
        )
    except APIClientError as error:
        if error.status_code == 503 and payload["evaluator"] == "llm_judge":
            st.error(
                "The LLM judge is unavailable. Check the backend's provider "
                "configuration and try again."
            )
        else:
            st.error(f"Evaluation could not be generated: {error}")
        return

    if not _is_evaluation_result(result) or result["run_id"] != run_id:
        st.error("The backend returned an evaluation in an unexpected format.")
        _render_run_evaluations(client, run_id)
        return

    st.success("Evaluation generated and saved.")
    _render_evaluation_result(result)


def _render_run_evaluations(
    client: AgentSentinelAPIClient,
    run_id: str,
) -> None:
    """Fetch and display evaluations associated with one execution."""
    try:
        results = client.get(f"/api/v1/executions/{run_id}/evaluations")
    except APIClientError as error:
        st.error(f"Run evaluations are unavailable: {error}")
        return

    if not isinstance(results, list) or not all(
        isinstance(result, dict) for result in results
    ):
        st.error("The backend returned run evaluations in an unexpected format.")
        return
    if not results:
        st.info(
            "This execution has no evaluations. Generate one from the "
            "execution details above."
        )
        return
    if not all(
        _is_evaluation_result(result) and result["run_id"] == run_id
        for result in results
    ):
        st.error("The backend returned run evaluations in an unexpected format.")
        return

    st.dataframe(
        [
            {
                "Evaluation ID": result.get("evaluation_id", "Unknown"),
                "Evaluator": result.get("evaluator", "Unknown"),
                "Overall score": result.get("overall_score"),
                "Passed": (
                    "Passed"
                    if result.get("passed") is True
                    else "Failed"
                    if result.get("passed") is False
                    else "Unknown"
                ),
                "Timestamp": _format_local_datetime(result.get("timestamp")),
            }
            for result in results
        ],
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.TextColumn(width="large")
        },
        alt="Evaluation history for this execution",
    )
    evaluation_ids = [result["evaluation_id"] for result in results]
    selected_evaluation_id = st.selectbox(
        "Inspect a saved evaluation",
        evaluation_ids,
        key=f"saved_evaluation_{run_id}",
    )
    _render_evaluation_details(client, selected_evaluation_id)


def _render_diagnostic_workflow(
    client: AgentSentinelAPIClient,
    run_id: str,
) -> None:
    """Create and display a persisted diagnostic report for an execution."""
    st.subheader("Generate diagnostic report")
    with st.form(f"diagnostic_form_{run_id}"):
        submitted = st.form_submit_button("Generate Diagnostic Report")
    if not submitted:
        return

    try:
        report = client.post(f"/api/v1/executions/{run_id}/diagnostics", json={})
    except APIClientError as error:
        st.error(f"Diagnostic report could not be generated: {error}")
        return

    if not _is_diagnostic_report(report) or report["run_id"] != run_id:
        st.error("The backend returned a diagnostic report in an unexpected format.")
        _render_execution_diagnostic_reports(client, run_id)
        return

    st.success("Diagnostic report generated and saved.")
    _render_diagnostic_report(report)


def _render_evaluations(client: AgentSentinelAPIClient) -> None:
    """Render global evaluation metrics, filtered results, and details."""
    st.subheader("Evaluation results")
    st.caption(
        "Overall scores use a 0–100 scale; individual quality dimensions "
        "use 0–1. A pass rate includes only evaluations with a known result."
    )
    try:
        evaluations, total = _fetch_all_items(
            client,
            "/api/v1/evaluations",
        )
    except APIClientError as error:
        st.error(f"Evaluation data is unavailable: {error}")
        return

    metric_fields = (
        ("Average overall score", "overall_score", 100),
        ("Average correctness", "correctness_score", 1),
        ("Average relevance", "relevance_score", 1),
        ("Average context relevance", "context_relevance_score", 1),
        ("Average hallucination", "hallucination_score", 1),
    )
    score_metrics = {
        field: _average_metric(evaluations, field)
        for _, field, _ in metric_fields
    }
    known_pass_results = [
        result["passed"]
        for result in evaluations
        if isinstance(result.get("passed"), bool)
    ]
    pass_rate = (
        sum(known_pass_results) / len(known_pass_results) * 100
        if known_pass_results
        else None
    )

    def format_average(field: str, maximum: int) -> str:
        value = score_metrics[field]
        return "—" if value is None else f"{value:.2f} / {maximum}"

    metric_row = st.columns(4)
    metric_row[0].metric("Total evaluations", _format_number(total), border=True)
    metric_row[1].metric(
        metric_fields[0][0],
        format_average(metric_fields[0][1], metric_fields[0][2]),
        border=True,
    )
    metric_row[2].metric(
        "Pass rate",
        "—" if pass_rate is None else f"{pass_rate:.1f}%",
        border=True,
    )
    metric_row[3].metric(
        metric_fields[1][0],
        format_average(metric_fields[1][1], metric_fields[1][2]),
        border=True,
    )

    score_row = st.columns(3)
    for column, (label, field, maximum) in zip(score_row, metric_fields[2:]):
        value = score_metrics[field]
        column.metric(
            label,
            "—" if value is None else f"{value:.2f} / {maximum}",
            border=True,
        )
    st.caption(
        "Pass rate uses only evaluations with a known pass/fail result. "
        "Score averages exclude missing values."
    )

    if not evaluations:
        st.info(
            "No evaluation results have been recorded yet. Open an execution "
            "and generate an evaluation to start measuring answer quality."
        )

    evaluators = sorted(
        {
            result["evaluator"]
            for result in evaluations
            if isinstance(result.get("evaluator"), str)
        }
    )
    filter_columns = st.columns(3)
    status_filter = filter_columns[0].selectbox(
        "Pass/fail status",
        ("All", "Passed", "Failed", "Unknown"),
        key=_EVALUATIONS_STATUS_KEY,
        on_change=_reset_evaluation_page,
    )
    evaluator_filter = filter_columns[1].selectbox(
        "Evaluator",
        ("All", *evaluators),
        key=_EVALUATIONS_EVALUATOR_KEY,
        on_change=_reset_evaluation_page,
    )
    page_size = filter_columns[2].selectbox(
        "Rows per page",
        (10, 20, 50, 100),
        index=1,
        key=_EVALUATIONS_PAGE_SIZE_KEY,
        on_change=_reset_evaluation_page,
    )

    filtered_results = evaluations
    if status_filter != "All":
        wanted_pass_value = {
            "Passed": True,
            "Failed": False,
            "Unknown": None,
        }[status_filter]
        filtered_results = [
            result
            for result in filtered_results
            if result.get("passed") is wanted_pass_value
        ]
    if evaluator_filter != "All":
        filtered_results = [
            result
            for result in filtered_results
            if result.get("evaluator") == evaluator_filter
        ]

    st.caption(
        f"Showing {len(filtered_results):,} matching result(s) "
        f"from {total:,} evaluations."
    )
    if not filtered_results:
        st.info("No evaluations match the selected filters.")
    else:
        page_count = (len(filtered_results) + page_size - 1) // page_size
        page_key = _EVALUATIONS_PAGE_KEY
        current_page = st.session_state.get(page_key, 0)
        if not isinstance(current_page, int) or current_page < 0:
            current_page = 0
        current_page = min(current_page, page_count - 1)
        st.session_state[page_key] = current_page

        previous_column, page_column, next_column = st.columns([1, 2, 1])
        previous_column.button(
            "Previous",
            disabled=current_page == 0,
            on_click=_move_evaluation_page,
            args=(-1,),
        )
        page_column.caption(f"Page {current_page + 1} of {page_count}")
        next_column.button(
            "Next",
            disabled=current_page >= page_count - 1,
            on_click=_move_evaluation_page,
            args=(1,),
        )

        offset = current_page * page_size
        page_results = filtered_results[offset : offset + page_size]
        st.dataframe(
            [
                {
                    "Evaluation ID": result.get("evaluation_id", "Unknown"),
                    "Run ID": result.get("run_id", "Unknown"),
                    "Evaluator": result.get("evaluator", "Unknown"),
                    "Overall score": result.get("overall_score"),
                    "Status": (
                        "Passed"
                        if result.get("passed") is True
                        else "Failed"
                        if result.get("passed") is False
                        else "Unknown"
                    ),
                    "Timestamp": _format_local_datetime(result.get("timestamp")),
                }
                for result in page_results
            ],
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn(width="large")
            },
            alt="Persisted evaluation results",
        )

        evaluation_ids = [
            result["evaluation_id"]
            for result in page_results
            if isinstance(result.get("evaluation_id"), str)
        ]
        if evaluation_ids:
            selected_evaluation_id = st.selectbox(
                "Inspect an evaluation",
                evaluation_ids,
                key="selected_evaluation_id",
            )
            _render_evaluation_details(client, selected_evaluation_id)
        else:
            st.info("The current page has no selectable evaluation IDs.")

    st.subheader("Evaluations by execution")
    run_ids = sorted(
        {
            result["run_id"]
            for result in evaluations
            if isinstance(result.get("run_id"), str)
        }
    )
    default_run_id = run_ids[0] if run_ids else ""
    run_id_input = st.text_input(
        "Execution run ID",
        value=default_run_id,
        key="evaluation_run_id_input",
    )
    if st.button(
        "Load execution evaluations",
        key="load_run_evaluations",
    ):
        if run_id_input.strip():
            _render_run_evaluations(client, run_id_input.strip())
        else:
            st.warning("Enter an execution run ID to look up its evaluations.")


def _reset_diagnostics_page() -> None:
    """Return diagnostic reports to the first page after filter changes."""
    st.session_state[_DIAGNOSTICS_PAGE_KEY] = 0


def _move_diagnostics_page(offset: int) -> None:
    """Move the diagnostic report page without allowing a negative index."""
    current_page = st.session_state.get(_DIAGNOSTICS_PAGE_KEY, 0)
    if not isinstance(current_page, int):
        current_page = 0
    st.session_state[_DIAGNOSTICS_PAGE_KEY] = max(0, current_page + offset)


def _is_diagnostic_finding(value: Any) -> bool:
    """Check that an API finding contains the fields required by its schema."""
    return (
        isinstance(value, dict)
        and _is_uuid_value(value.get("finding_id"))
        and value.get("severity") in ("low", "medium", "high", "critical")
        and all(
            isinstance(value.get(field), str)
            for field in ("category", "title", "description", "recommendation")
        )
        and isinstance(value.get("evidence"), list)
        and all(isinstance(item, str) for item in value["evidence"])
        and isinstance(value.get("metadata"), dict)
    )


def _is_diagnostic_report(value: Any) -> bool:
    """Check that an API report and its findings match the diagnostic schema."""
    return (
        isinstance(value, dict)
        and _is_uuid_value(value.get("report_id"))
        and _is_uuid_value(value.get("run_id"))
        and value.get("overall_status") in _DIAGNOSTIC_STATUSES
        and _is_timestamp_value(value.get("timestamp"))
        and isinstance(value.get("summary"), str)
        and isinstance(value.get("findings"), list)
        and all(_is_diagnostic_finding(finding) for finding in value["findings"])
        and isinstance(value.get("metadata"), dict)
    )


def _is_uuid_value(value: Any) -> bool:
    """Check that a value is a UUID string as returned by the diagnostic API."""
    if not isinstance(value, str):
        return False
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _is_timestamp_value(value: Any) -> bool:
    """Check that an API timestamp is an ISO-formatted datetime string."""
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _diagnostic_report_rows(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build safe, readable table rows from diagnostic reports."""
    return [
        {
            "Report ID": report["report_id"],
            "Run ID": report["run_id"],
            "Status": report["overall_status"],
            "Timestamp": _format_local_datetime(report["timestamp"]),
            "Findings": len(report["findings"]),
        }
        for report in reports
    ]


def _render_diagnostic_report_details(
    client: AgentSentinelAPIClient,
    report_id: str,
) -> None:
    """Fetch and safely display a diagnostic report and all its findings."""
    try:
        report = client.get(f"/api/v1/diagnostics/{report_id}")
    except APIClientError as error:
        st.error(f"Diagnostic report details are unavailable: {error}")
        return

    if not _is_diagnostic_report(report):
        st.error("The backend returned diagnostic details in an unexpected format.")
        return

    _render_diagnostic_report(report)


def _render_diagnostic_report(report: dict[str, Any]) -> None:
    """Render a validated report response and its findings safely."""
    st.subheader("Diagnostic report details")
    status_column, timestamp_column, finding_column = st.columns(3)
    status_column.metric("Status", report["overall_status"].title(), border=True)
    timestamp_column.metric(
        "Timestamp",
        _format_local_datetime(report["timestamp"]),
        border=True,
    )
    finding_column.metric(
        "Findings",
        _format_number(len(report["findings"])),
        border=True,
    )
    st.caption(f"Report ID: {report['report_id']}")
    st.caption(f"Associated execution ID: {report['run_id']}")
    st.markdown("**Summary**")
    st.code(_format_event_payload(report["summary"]))

    if not report["findings"]:
        st.info("This report has no diagnostic findings.")
    for index, finding in enumerate(report["findings"], start=1):
        heading = _format_event_payload(
            f"{finding['severity'].upper()} · {finding['category']} · "
            f"{finding['title']}"
        )
        with st.expander(f"Finding {index}: {heading}"):
            for label, field in (
                ("Severity", "severity"),
                ("Category", "category"),
                ("Description", "description"),
                ("Recommendation", "recommendation"),
                ("Evidence", "evidence"),
                ("Additional details", "metadata"),
            ):
                st.markdown(f"**{label}**")
                st.code(_format_event_payload(finding[field]))

    if report["metadata"]:
        with st.expander("Report metadata"):
            st.code(_format_event_payload(report["metadata"]))


def _render_execution_diagnostic_reports(
    client: AgentSentinelAPIClient,
    run_id: str,
) -> None:
    """Fetch and display diagnostic reports associated with one execution."""
    try:
        reports = client.get(f"/api/v1/executions/{run_id}/diagnostics")
    except APIClientError as error:
        st.error(f"Execution diagnostic reports are unavailable: {error}")
        return

    if not isinstance(reports, list) or not all(
        _is_diagnostic_report(report) for report in reports
    ):
        st.error("The backend returned execution reports in an unexpected format.")
        return
    if not reports:
        st.info(
            "This execution has no diagnostic reports. Generate a report from "
            "the execution details above."
        )
        return
    st.dataframe(
        _diagnostic_report_rows(reports),
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.TextColumn(width="large")
        },
        alt="Diagnostic reports for this execution",
    )
    report_ids = [report["report_id"] for report in reports]
    selected_report_id = st.selectbox(
        "Inspect a saved execution report",
        report_ids,
        key=f"saved_diagnostic_report_{run_id}",
    )
    _render_diagnostic_report_details(client, selected_report_id)


def _render_diagnostics(client: AgentSentinelAPIClient) -> None:
    """Render global diagnostic metrics, filterable reports, and report details."""
    st.subheader("Diagnostic reports")
    st.caption(
        "Reports group detected issues by severity and include evidence and "
        "recommendations. A healthy report has no detected findings."
    )
    try:
        reports, total = _fetch_all_items(
            client,
            "/api/v1/diagnostics",
        )
    except APIClientError as error:
        st.error(f"Diagnostic reports are unavailable: {error}")
        return

    if not all(_is_diagnostic_report(report) for report in reports):
        st.error("The backend returned diagnostic reports in an unexpected format.")
        return

    status_counts = {
        status: sum(report["overall_status"] == status for report in reports)
        for status in _DIAGNOSTIC_STATUSES
    }
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Total diagnostic reports",
        _format_number(total),
        border=True,
    )
    for column, status in zip(metric_columns[1:], _DIAGNOSTIC_STATUSES):
        column.metric(
            f"{status.title()} reports",
            _format_number(status_counts[status]),
            border=True,
        )

    filter_columns = st.columns(2)
    selected_status = filter_columns[0].selectbox(
        "Report status",
        ("All", "Healthy", "Warning", "Critical"),
        key=_DIAGNOSTICS_STATUS_KEY,
        on_change=_reset_diagnostics_page,
    )
    page_size = filter_columns[1].selectbox(
        "Reports per page",
        (10, 20, 50, 100),
        index=1,
        key=_DIAGNOSTICS_PAGE_SIZE_KEY,
        on_change=_reset_diagnostics_page,
    )
    filtered_reports = (
        reports
        if selected_status == "All"
        else [
            report
            for report in reports
            if report["overall_status"] == selected_status.lower()
        ]
    )

    if not reports:
        st.info(
            "No diagnostic reports have been recorded yet. Generate a report "
            "from any execution's details."
        )
    elif not filtered_reports:
        st.info("No diagnostic reports match the selected status.")
    else:
        page_count = (len(filtered_reports) + page_size - 1) // page_size
        current_page = st.session_state.get(_DIAGNOSTICS_PAGE_KEY, 0)
        if not isinstance(current_page, int) or current_page < 0:
            current_page = 0
        current_page = min(current_page, page_count - 1)
        st.session_state[_DIAGNOSTICS_PAGE_KEY] = current_page

        previous_column, page_column, next_column = st.columns([1, 2, 1])
        previous_column.button(
            "Previous",
            disabled=current_page == 0,
            on_click=_move_diagnostics_page,
            args=(-1,),
        )
        page_column.caption(f"Page {current_page + 1} of {page_count}")
        next_column.button(
            "Next",
            disabled=current_page >= page_count - 1,
            on_click=_move_diagnostics_page,
            args=(1,),
        )

        offset = current_page * page_size
        page_reports = filtered_reports[offset : offset + page_size]
        st.caption(
            f"Showing {len(page_reports):,} report(s) on page "
            f"{current_page + 1} of {page_count} "
            f"({len(filtered_reports):,} matching, {total:,} total)."
        )
        st.dataframe(
            _diagnostic_report_rows(page_reports),
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn(width="large")
            },
            alt="Persisted diagnostic reports",
        )

        report_ids = [report["report_id"] for report in page_reports]
        selected_report_id = st.selectbox(
            "Inspect a diagnostic report",
            report_ids,
            key="selected_diagnostic_report_id",
        )
        _render_diagnostic_report_details(client, selected_report_id)

    st.subheader("Diagnostic reports by execution")
    run_id = st.text_input(
        "Execution run ID",
        key="diagnostic_run_id_input",
    )
    if st.button("Load execution diagnostic reports"):
        if run_id.strip():
            if not _is_uuid_value(run_id.strip()):
                st.warning("Enter a valid execution UUID.")
            else:
                _render_execution_diagnostic_reports(client, run_id.strip())
        else:
            st.warning("Enter an execution run ID to look up its diagnostic reports.")


def _render_usage_analytics(analytics: dict[str, Any]) -> None:
    """Render compact platform-wide token, cost, latency, and efficiency views."""
    st.subheader("Usage and efficiency analytics")
    st.caption(
        "Metrics use every execution page and each execution's full persisted "
        "event list. Costs are estimates from the pricing version recorded per call."
    )
    if not analytics["executions"]:
        st.info("No execution data is available for usage analytics.")
        return

    p95 = analytics["p95_latency_ms"]
    p95_value = (
        _format_duration_ms(p95)
        if p95 is not None
        else "Unavailable (requires 20+ executions)"
    )
    metrics = st.columns(5)
    metrics[0].metric(
        "Total input tokens",
        _format_token_count(analytics["input_tokens"]),
        border=True,
    )
    metrics[1].metric(
        "Total output tokens",
        _format_token_count(analytics["output_tokens"]),
        border=True,
    )
    metrics[2].metric(
        "Total tokens",
        _format_token_count(analytics["total_tokens"]),
        border=True,
    )
    metrics[3].metric(
        "Estimated total cost",
        _format_estimated_cost(
            {
                "available": analytics["total_cost_usd"] is not None,
                "amount_usd": analytics["total_cost_usd"],
            }
        ),
        border=True,
    )
    metrics[4].metric(
        "Average execution latency",
        _format_duration_ms(analytics["average_latency_ms"]),
        border=True,
    )
    st.metric("P95 execution latency", p95_value, border=True)
    st.caption(
        f"Latency uses {analytics['latency_sample_count']} of "
        f"{analytics['latency_sample_total']} executions with a known terminal "
        "duration. P95 is shown only with at least 20 known durations."
    )
    if analytics["unavailable_usage_calls"]:
        st.caption(
            f"{analytics['reported_usage_calls']} of "
            f"{analytics['llm_call_count']} LLM call(s) reported token usage; "
            f"{analytics['unavailable_usage_calls']} are unavailable. "
            "Complete token totals are withheld."
        )

    trend_columns = st.columns(2)
    with trend_columns[0]:
        with st.container(border=True):
            st.markdown("**Token consumption trend**")
            st.caption("Reported total tokens by execution start date.")
            if analytics["daily_tokens"]:
                st.line_chart(
                    analytics["daily_tokens"],
                    x="Date",
                    y="Tokens",
                    height=220,
                    alt="Reported token consumption by execution date",
                )
            else:
                st.info("No executions with complete token usage are available.")
    with trend_columns[1]:
        with st.container(border=True):
            st.markdown("**Execution latency trend**")
            st.caption("Average complete execution duration by start date.")
            if analytics["daily_latency"]:
                st.line_chart(
                    analytics["daily_latency"],
                    x="Date",
                    y="Average latency (ms)",
                    height=220,
                    alt="Average execution latency by execution date",
                )
            else:
                st.info("No completed executions with a recorded duration are available.")

    distribution_columns = st.columns(2)
    with distribution_columns[0]:
        with st.container(border=True):
            st.markdown("**Input and output token distribution**")
            if (
                analytics["input_tokens"] is not None
                and analytics["output_tokens"] is not None
            ):
                st.bar_chart(
                    [
                        {"Token type": "Input", "Tokens": analytics["input_tokens"]},
                        {"Token type": "Output", "Tokens": analytics["output_tokens"]},
                    ],
                    x="Token type",
                    y="Tokens",
                    height=220,
                    alt="Reported input and output token totals",
                )
            else:
                st.info(
                    "Distribution is unavailable because at least one call's "
                    "token usage was not reported."
                )
    with distribution_columns[1]:
        with st.container(border=True):
            st.markdown("**Usage by provider and model**")
            if analytics["model_usage"]:
                st.dataframe(
                    analytics["model_usage"],
                    hide_index=True,
                    alt="Token usage and estimated cost by provider and model",
                )
            else:
                st.info("No LLM calls have been recorded.")

    known_token_rows = [
        row
        for row in analytics["executions"]
        if row["usage"]["total_tokens"] is not None
    ]
    top_executions = sorted(
        known_token_rows,
        key=lambda row: row["usage"]["total_tokens"],
        reverse=True,
    )[:10]
    with st.container(border=True):
        st.markdown("**Highest token-consuming executions**")
        if top_executions:
            st.dataframe(
                [
                    {
                        "Run ID": row["run_id"],
                        "Started": _format_local_datetime(row["started_at"]),
                        "Input tokens": _format_token_count(
                            row["usage"]["input_tokens"]
                        ),
                        "Output tokens": _format_token_count(
                            row["usage"]["output_tokens"]
                        ),
                        "Total tokens": _format_token_count(
                            row["usage"]["total_tokens"]
                        ),
                    }
                    for row in top_executions
                ],
                hide_index=True,
                column_config={
                    "Started": st.column_config.TextColumn(width="large")
                },
                alt="Executions ranked by reported token consumption",
            )
        else:
            st.info("No executions have complete token usage to rank.")

    token_samples = analytics["known_token_runs"]
    latency_samples = [
        row["duration_ms"]
        for row in analytics["executions"]
        if row["duration_ms"] is not None
        and row["status"] in {"completed", "failed"}
    ]
    insights: list[str] = []
    outlier_rows: list[dict[str, Any]] = []
    if len(token_samples) >= 3:
        token_median = median(token_samples)
        high_token_runs = [
            row
            for row in known_token_rows
            if token_median > 0
            and row["usage"]["total_tokens"] >= token_median * 2
            and row["usage"]["total_tokens"] > token_median
        ]
        if high_token_runs:
            insights.append(
                f"{len(high_token_runs)} execution(s) used at least twice the "
                f"median token count ({token_median:,.0f})."
            )
            outlier_rows.extend(
                {
                    "Observation": "High token consumption",
                    "Run ID": row["run_id"],
                    "Observed": _format_token_count(row["usage"]["total_tokens"]),
                    "Comparison": f"Median {token_median:,.0f} tokens",
                }
                for row in high_token_runs
            )
    if len(latency_samples) >= 3:
        latency_median = median(latency_samples)
        high_latency_runs = [
            row
            for row in analytics["executions"]
            if row["duration_ms"] is not None
            and latency_median > 0
            and row["duration_ms"] >= latency_median * 2
            and row["duration_ms"] > latency_median
        ]
        if high_latency_runs:
            insights.append(
                f"{len(high_latency_runs)} execution(s) took at least twice "
                f"the median latency ({_format_duration_ms(latency_median)})."
            )
            outlier_rows.extend(
                {
                    "Observation": "High execution latency",
                    "Run ID": row["run_id"],
                    "Observed": _format_duration_ms(row["duration_ms"]),
                    "Comparison": (
                        f"Median {_format_duration_ms(latency_median)}"
                    ),
                }
                for row in high_latency_runs
            )
    st.markdown("**Relative usage observations**")
    st.caption("These comparisons do not measure optimization savings.")
    if insights:
        for insight in insights:
            st.info(insight)
        st.dataframe(
            outlier_rows,
            hide_index=True,
            alt="Executions identified as relative token or latency outliers",
        )
    else:
        st.caption(
            "No outlier observation is shown; at least three comparable "
            "executions are required. These comparisons do not imply savings."
        )


def _render_overview(client: AgentSentinelAPIClient) -> None:
    """Fetch and display overview metrics and recent executions."""
    st.subheader("Platform health at a glance")
    st.caption(
        "Success rate excludes runs still in progress. Evaluation scores and "
        "diagnostic statuses summarize persisted records."
    )
    try:
        executions, execution_total = _fetch_all_items(
            client,
            "/api/v1/executions",
        )
    except APIClientError as error:
        executions = []
        execution_total = None
        st.warning(f"Execution data is unavailable: {error}")

    usage_analytics: dict[str, Any] | None = None
    if execution_total is not None:
        try:
            usage_analytics = _fetch_usage_analytics(client, executions)
        except APIClientError as error:
            st.warning(f"Usage analytics are unavailable: {error}")

    try:
        evaluations, evaluation_total = _fetch_all_items(
            client,
            "/api/v1/evaluations",
        )
    except APIClientError as error:
        evaluations = []
        evaluation_total = None
        st.warning(f"Evaluation data is unavailable: {error}")

    try:
        diagnostic_reports, diagnostic_total = _fetch_all_items(
            client,
            "/api/v1/diagnostics",
        )
    except APIClientError as error:
        diagnostic_reports = []
        diagnostic_total = None
        st.warning(f"Diagnostic data is unavailable: {error}")

    if execution_total is None:
        total_executions = successful_executions = failed_executions = None
        running_executions = None
        success_rate = None
    else:
        total_executions = execution_total
        successful_executions = sum(
            execution.get("status") == "completed" for execution in executions
        )
        failed_executions = sum(
            execution.get("status") == "failed" for execution in executions
        )
        running_executions = sum(
            execution.get("status") == "running" for execution in executions
        )
        terminal_executions = successful_executions + failed_executions
        success_rate = (
            successful_executions / terminal_executions * 100
            if terminal_executions
            else None
        )

    if evaluation_total is None:
        total_evaluations = None
        average_score = None
        evaluation_averages: dict[str, float | None] = {}
    else:
        total_evaluations = evaluation_total
        evaluation_fields = (
            "overall_score",
            "correctness_score",
            "relevance_score",
            "context_relevance_score",
            "hallucination_score",
        )
        evaluation_averages = {
            field: _average_metric(evaluations, field)
            for field in evaluation_fields
        }
        average_score = evaluation_averages["overall_score"]

    first_row = st.columns(4)
    first_row[0].metric(
        "Total executions",
        "—" if total_executions is None else _format_number(total_executions),
        border=True,
    )
    first_row[1].metric(
        "Successful executions",
        "—"
        if successful_executions is None
        else _format_number(successful_executions),
        border=True,
    )
    first_row[2].metric(
        "Failed executions",
        "—" if failed_executions is None else _format_number(failed_executions),
        border=True,
    )
    first_row[3].metric(
        "Execution success rate",
        "—" if success_rate is None else f"{success_rate:.1f}%",
        border=True,
    )

    second_row = st.columns(3)
    second_row[0].metric(
        "Total evaluations",
        "—" if total_evaluations is None else _format_number(total_evaluations),
        border=True,
    )
    second_row[1].metric(
        "Average evaluation score",
        "—" if average_score is None else f"{average_score:.1f} / 100",
        border=True,
    )
    second_row[2].metric(
        "Total diagnostic reports",
        "—" if diagnostic_total is None else _format_number(diagnostic_total),
        border=True,
    )

    charts = st.columns(2)
    with charts[0]:
        with st.container(border=True):
            st.subheader("Execution status")
            st.caption("Completed, failed, and in-progress runs in the data set.")
            if execution_total is None:
                st.info(
                    "Execution status is unavailable because execution data "
                    "could not be loaded."
                )
            elif not executions:
                st.info(
                    "No execution data is available to chart. Run an agent "
                    "from Agent Playground to create execution data."
                )
            else:
                st.bar_chart(
                    [
                        {"Status": "Completed", "Executions": successful_executions},
                        {"Status": "Failed", "Executions": failed_executions},
                        {"Status": "Running", "Executions": running_executions},
                    ],
                    x="Status",
                    y="Executions",
                    height=260,
                    alt="Counts of completed, failed, and running executions",
                )

    with charts[1]:
        with st.container(border=True):
            st.subheader("Evaluation scores")
            st.caption("Average score by quality dimension; higher is better.")
            if evaluation_total is None:
                st.info(
                    "Evaluation scores are unavailable because evaluation "
                    "data could not be loaded."
                )
            elif not evaluations:
                st.info(
                    "No evaluation data is available to chart. Generate an "
                    "evaluation from an execution's details."
                )
            else:
                evaluation_labels = (
                    ("overall_score", "Overall score (0–100)"),
                    ("correctness_score", "Correctness (0–1)"),
                    ("relevance_score", "Relevance (0–1)"),
                    ("context_relevance_score", "Context relevance (0–1)"),
                    ("hallucination_score", "Hallucination (0–1)"),
                )
                evaluation_chart_data = [
                    {"Metric": label, "Average score": evaluation_averages[field]}
                    for field, label in evaluation_labels
                    if evaluation_averages[field] is not None
                ]
                if evaluation_chart_data:
                    st.caption(
                        "Component scores use 0–1; overall scores use 0–100."
                    )
                    st.bar_chart(
                        evaluation_chart_data,
                        x="Metric",
                        y="Average score",
                        height=260,
                        alt="Average overall and component evaluation scores",
                    )
                else:
                    st.info("No numeric evaluation scores are available to chart.")

    with charts[0]:
        with st.container(border=True):
            st.subheader("Diagnostic status")
            st.caption(
                "Healthy reports have no findings; warning and critical "
                "reports need attention."
            )
            if diagnostic_total is None:
                st.info(
                    "Diagnostic status is unavailable because diagnostic "
                    "data could not be loaded."
                )
            elif not diagnostic_reports:
                st.info(
                    "No diagnostic data is available to chart. Generate a "
                    "diagnostic report from an execution's details."
                )
            else:
                diagnostic_status_counts = {
                    status: sum(
                        report.get("overall_status") == status
                        for report in diagnostic_reports
                    )
                    for status in _DIAGNOSTIC_STATUSES
                }
                st.bar_chart(
                    [
                        {
                            "Status": "Healthy",
                            "Reports": diagnostic_status_counts["healthy"],
                        },
                        {
                            "Status": "Warning",
                            "Reports": diagnostic_status_counts["warning"],
                        },
                        {
                            "Status": "Critical",
                            "Reports": diagnostic_status_counts["critical"],
                        },
                    ],
                    x="Status",
                    y="Reports",
                    height=260,
                    alt="Counts of healthy, warning, and critical diagnostic reports",
                )

    if usage_analytics is not None:
        _render_usage_analytics(usage_analytics)
    else:
        st.subheader("Usage and efficiency analytics")
        st.info(
            "Usage analytics are unavailable because complete execution events "
            "could not be loaded."
        )

    st.subheader("Recent executions")
    if execution_total is None:
        st.info(
            "Recent executions cannot be displayed while execution data "
            "is unavailable."
        )
    elif not executions:
        st.info("No executions have been recorded yet.")
    else:
        recent_executions = [
            {
                "Run ID": execution.get("run_id", "Unknown"),
                "Agent": execution.get("agent_name", "Unknown"),
                "Status": execution.get("status", "Unknown"),
                "Started": _format_local_datetime(execution.get("started_at")),
                "Duration (ms)": execution.get("duration_ms"),
            }
            for execution in executions[:10]
        ]
        st.dataframe(
            recent_executions,
            hide_index=True,
            column_config={
                "Started": st.column_config.TextColumn(width="large")
            },
            alt="Recent agent executions",
        )


def main() -> None:
    """Render the dashboard foundation and check backend connectivity."""
    st.set_page_config(
        page_title="AgentSentinel",
        page_icon="A",
        layout="wide",
    )
    st.title("AgentSentinel")
    st.caption("Monitor. Evaluate. Diagnose. Improve.")

    with st.sidebar:
        st.subheader("Navigation")
        selected_page = st.radio(
            "Dashboard",
            _NAVIGATION_ITEMS,
            key=_NAVIGATION_KEY,
        )
        st.button("Refresh data", type="primary", icon=":material/refresh:")

    st.header(selected_page)
    st.caption(_PAGE_DESCRIPTIONS[selected_page])

    try:
        client = AgentSentinelAPIClient()
        health = client.get("/health")
    except APIClientError as error:
        st.sidebar.error("Backend unavailable")
        st.warning(str(error))
        st.info(
            "Start the AgentSentinel API and confirm "
            "`AGENTSENTINEL_API_URL` points to the correct address."
        )
        return

    if not isinstance(health, dict) or health.get("status") != "healthy":
        st.sidebar.warning("Backend responded with an unhealthy status")
        st.warning("The backend is reachable but did not report healthy status.")
    else:
        st.sidebar.success("Backend connected")
        st.success(f"{health.get('application', 'Backend')} is healthy.")

    if selected_page == "Overview":
        _render_overview(client)
    elif selected_page == "Executions":
        _render_executions(client)
    elif selected_page == "Failures":
        _render_failures(client)
    elif selected_page == "Evaluations":
        _render_evaluations(client)
    elif selected_page == "Diagnostics":
        _render_diagnostics(client)
    elif selected_page == "Agent Playground":
        _render_agent_playground(client)
    else:
        st.info(f"The {selected_page.lower()} dashboard page is coming soon.")


if __name__ == "__main__":
    main()
