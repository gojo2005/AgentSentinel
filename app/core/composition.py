"""Application composition helpers for AgentSentinel."""

from app.monitoring.event_logger import EventLogger
from app.storage.event_sink import SQLAlchemyEventSink


def create_production_event_logger() -> EventLogger:
    """Create an event logger backed by persistent SQLAlchemy storage."""
    return EventLogger(sink=SQLAlchemyEventSink())
