"""Provider-independent data schemas for AgentSentinel execution events."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ExecutionEvent(BaseModel):
    """A normalized event describing one part of an agent execution."""

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this event.",
    )
    run_id: UUID = Field(
        description="Identifier shared by all events in one agent execution.",
    )
    event_type: Literal[
        "run_started",
        "llm_call",
        "tool_call",
        "run_completed",
        "run_failed",
    ] = Field(description="Kind of execution event.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time when this event was created.",
    )
    agent_name: str = Field(description="Name of the agent that emitted the event.")
    provider: str | None = Field(
        default=None,
        description="Optional name of the LLM provider.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model identifier.",
    )
    tool_name: str | None = Field(
        default=None,
        description="Optional name of the tool involved in the event.",
    )
    status: Literal["started", "success", "failure"] = Field(
        description="Status of the operation represented by this event.",
    )
    input: Any | None = Field(
        default=None,
        description="Optional input associated with the event.",
    )
    output: Any | None = Field(
        default=None,
        description="Optional output associated with the event.",
    )
    duration_ms: float | None = Field(
        default=None,
        ge=0,
        description="Optional non-negative duration of the operation in milliseconds.",
    )
    error: str | None = Field(
        default=None,
        description="Optional description of an error.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event information.",
    )
