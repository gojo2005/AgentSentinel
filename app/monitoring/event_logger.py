"""In-memory storage for normalized AgentSentinel execution events."""

from typing import Any, Literal
from uuid import UUID

from app.monitoring.interfaces import EventSink
from app.monitoring.schemas import ExecutionEvent


class EventLogger:
    """Create and keep execution events in memory for the current process."""

    def __init__(self, sink: EventSink | None = None) -> None:
        """Create an empty event collection with an optional event sink."""
        self._events: list[ExecutionEvent] = []
        self._sink = sink

    def log_event(
        self,
        *,
        run_id: UUID,
        event_type: Literal[
            "run_started",
            "llm_call",
            "tool_call",
            "run_completed",
            "run_failed",
        ],
        agent_name: str,
        status: Literal["started", "success", "failure"],
        provider: str | None = None,
        model: str | None = None,
        tool_name: str | None = None,
        input: Any | None = None,
        output: Any | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        """Create, store, and return an event for the given execution."""
        event = ExecutionEvent(
            run_id=run_id,
            event_type=event_type,
            agent_name=agent_name,
            status=status,
            provider=provider,
            model=model,
            tool_name=tool_name,
            input=input,
            output=output,
            duration_ms=duration_ms,
            error=error,
            metadata={} if metadata is None else metadata,
        )
        self._events.append(event)
        if self._sink is not None:
            self._sink.write_event(event)
        return event

    def get_events(self, run_id: UUID | None = None) -> list[ExecutionEvent]:
        """Return a copy of all events, optionally filtered by execution ID."""
        if run_id is None:
            return self._events.copy()
        return [event for event in self._events if event.run_id == run_id]

    def clear_events(self) -> None:
        """Remove all events from this logger."""
        self._events.clear()
