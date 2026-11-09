"""Response schemas for persisted execution history."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from app.monitoring.schemas import ExecutionEvent


class ExecutionSummary(BaseModel):
    """Aggregated status and timing information for one execution."""

    run_id: UUID
    agent_name: str
    started_at: datetime
    completed_at: datetime | None
    status: Literal["completed", "failed", "running"]
    total_events: int
    duration_ms: float | None


class ExecutionListResponse(BaseModel):
    """Paginated execution summaries."""

    items: list[ExecutionSummary]
    total: int
    limit: int
    offset: int


class ExecutionDetailResponse(BaseModel):
    """An execution summary together with its recorded events."""

    summary: ExecutionSummary
    events: list[ExecutionEvent]


class ExecutionEventsResponse(BaseModel):
    """Chronologically ordered events for one execution."""

    run_id: UUID
    events: list[ExecutionEvent]
