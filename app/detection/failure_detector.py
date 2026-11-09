"""Provider-independent analysis of AgentSentinel execution events."""

from collections import defaultdict
from typing import Any
from uuid import UUID

from app.core.sanitization import redact_sensitive_data
from app.monitoring.schemas import ExecutionEvent


class FailureDetector:
    """Summarize failures and completion status from execution events."""

    def analyze_run(self, events: list[ExecutionEvent]) -> dict[str, Any]:
        """Analyze events from one run and return a structured summary."""
        if not events:
            return {
                "run_id": None,
                "overall_status": "unknown",
                "total_events": 0,
                "failed_events": 0,
                "llm_failures": 0,
                "tool_failures": 0,
                "run_failures": 0,
                "errors": [],
                "total_duration_ms": None,
            }

        run_ids = {event.run_id for event in events}
        if len(run_ids) != 1:
            raise ValueError("analyze_run expects events belonging to one run.")

        failed_events = [
            event for event in events if event.status == "failure"
        ]
        llm_failures = sum(
            event.status == "failure" and event.event_type == "llm_call"
            for event in events
        )
        tool_failures = sum(
            event.status == "failure" and event.event_type == "tool_call"
            for event in events
        )
        run_failures = sum(
            event.event_type == "run_failed" and event.status == "failure"
            for event in events
        )

        if run_failures or any(
            event.event_type == "run_completed" and event.status == "failure"
            for event in events
        ):
            overall_status = "failure"
        elif any(
            event.event_type == "run_completed" and event.status == "success"
            for event in events
        ):
            overall_status = "success"
        else:
            overall_status = "unknown"

        total_duration_ms = next(
            (
                event.duration_ms
                for event in reversed(events)
                if event.event_type in {"run_completed", "run_failed"}
                and event.duration_ms is not None
            ),
            None,
        )

        return {
            "run_id": next(iter(run_ids)),
            "overall_status": overall_status,
            "total_events": len(events),
            "failed_events": len(failed_events),
            "llm_failures": llm_failures,
            "tool_failures": tool_failures,
            "run_failures": run_failures,
            "errors": [
                redact_sensitive_data(event.error)
                for event in events
                if event.error is not None
            ],
            "total_duration_ms": total_duration_ms,
        }

    def analyze_events(self, events: list[ExecutionEvent]) -> dict[str, Any]:
        """Group events by run and return aggregate and per-run summaries."""
        events_by_run: dict[UUID, list[ExecutionEvent]] = defaultdict(list)
        for event in events:
            events_by_run[event.run_id].append(event)

        run_analyses = [
            self.analyze_run(run_events)
            for run_events in events_by_run.values()
        ]
        successful_runs = sum(
            run["overall_status"] == "success" for run in run_analyses
        )
        failed_runs = sum(
            run["overall_status"] == "failure" for run in run_analyses
        )

        return {
            "total_runs": len(run_analyses),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "total_events": len(events),
            "total_failures": sum(
                run["failed_events"] for run in run_analyses
            ),
            "runs": run_analyses,
        }
