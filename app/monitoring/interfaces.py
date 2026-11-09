"""Framework-independent interfaces for monitoring integrations."""

from typing import Protocol

from app.monitoring.schemas import ExecutionEvent


class EventSink(Protocol):
    """A destination that accepts normalized execution events."""

    def write_event(self, event: ExecutionEvent) -> None:
        """Write one execution event to the sink."""
