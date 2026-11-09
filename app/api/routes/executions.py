"""Endpoints for reading persisted agent execution history."""

from collections import defaultdict
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.schemas.execution import (
    ExecutionDetailResponse,
    ExecutionEventsResponse,
    ExecutionListResponse,
    ExecutionSummary,
)
from app.detection.failure_detector import FailureDetector
from app.monitoring.schemas import ExecutionEvent
from app.storage.database import get_db
from app.storage.event_repository import ExecutionEventRepository

router = APIRouter(prefix="/executions", tags=["executions"])
DatabaseSession = Annotated[Session, Depends(get_db)]


def _build_execution_summary(
    events: list[ExecutionEvent],
) -> ExecutionSummary:
    """Build a summary for a non-empty set of events from one execution."""
    if not events:
        raise ValueError("Cannot summarize an execution without events.")

    run_ids = {event.run_id for event in events}
    if len(run_ids) != 1:
        raise ValueError("Execution summary requires events from one run.")

    ordered_events = sorted(events, key=lambda event: event.timestamp)
    started_event = next(
        (event for event in ordered_events if event.event_type == "run_started"),
        ordered_events[0],
    )
    terminal_event = next(
        (
            event
            for event in reversed(ordered_events)
            if event.event_type in {"run_completed", "run_failed"}
        ),
        None,
    )

    analysis = FailureDetector().analyze_run(ordered_events)
    status_by_analysis = {
        "success": "completed",
        "failure": "failed",
    }
    status = status_by_analysis.get(analysis["overall_status"], "running")
    completed_at = terminal_event.timestamp if terminal_event is not None else None
    duration_ms = analysis["total_duration_ms"]
    if (
        duration_ms is None
        and completed_at is not None
        and completed_at >= started_event.timestamp
    ):
        duration_ms = (
            completed_at - started_event.timestamp
        ).total_seconds() * 1000

    return ExecutionSummary(
        run_id=next(iter(run_ids)),
        agent_name=started_event.agent_name,
        started_at=started_event.timestamp,
        completed_at=completed_at,
        status=status,
        total_events=len(ordered_events),
        duration_ms=duration_ms,
    )


def _get_execution_events(
    repository: ExecutionEventRepository,
    run_id: UUID,
) -> list[ExecutionEvent]:
    """Return events for a run or raise 404 when that run is unknown."""
    events = repository.get_by_run_id(run_id)
    if not events:
        raise HTTPException(status_code=404, detail="Execution not found.")
    return events


@router.get("", response_model=ExecutionListResponse)
def list_executions(
    session: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ExecutionListResponse:
    """List distinct executions, newest first, with pagination."""
    events = ExecutionEventRepository(session).get_all()
    events_by_run: dict[UUID, list[ExecutionEvent]] = defaultdict(list)
    for event in events:
        events_by_run[event.run_id].append(event)

    summaries = [
        _build_execution_summary(run_events)
        for run_events in events_by_run.values()
    ]
    summaries.sort(
        key=lambda summary: (summary.started_at, str(summary.run_id)),
        reverse=True,
    )

    return ExecutionListResponse(
        items=summaries[offset : offset + limit],
        total=len(summaries),
        limit=limit,
        offset=offset,
    )


@router.get("/{run_id}", response_model=ExecutionDetailResponse)
def get_execution(
    run_id: UUID,
    session: DatabaseSession,
) -> ExecutionDetailResponse:
    """Return one execution summary and its chronologically ordered events."""
    events = _get_execution_events(ExecutionEventRepository(session), run_id)
    return ExecutionDetailResponse(
        summary=_build_execution_summary(events),
        events=events,
    )


@router.get("/{run_id}/events", response_model=ExecutionEventsResponse)
def list_execution_events(
    run_id: UUID,
    session: DatabaseSession,
) -> ExecutionEventsResponse:
    """Return all events for an execution in chronological order."""
    events = _get_execution_events(ExecutionEventRepository(session), run_id)
    return ExecutionEventsResponse(run_id=run_id, events=events)
