"""Tests for in-memory event logging and optional event sinks."""

from uuid import UUID, uuid4

import pytest

from app.monitoring.event_logger import EventLogger
from app.monitoring.schemas import ExecutionEvent


class FakeEventSink:
    """Collect events written through the EventSink protocol."""

    def __init__(self) -> None:
        self.events: list[ExecutionEvent] = []

    def write_event(self, event: ExecutionEvent) -> None:
        """Store an event received from the logger."""
        self.events.append(event)


class FailingEventSink:
    """Raise an error whenever the logger attempts to persist an event."""

    def write_event(self, event: ExecutionEvent) -> None:
        """Simulate a persistence failure."""
        raise RuntimeError("event persistence failed")


def log_started_event(
    logger: EventLogger,
    run_id: UUID | None = None,
) -> ExecutionEvent:
    """Log a valid run-start event, optionally using a given run ID."""
    return logger.log_event(
        run_id=uuid4() if run_id is None else run_id,
        event_type="run_started",
        agent_name="test-agent",
        status="started",
        input={"question": "test question"},
    )


def test_default_event_logger_works_without_a_sink() -> None:
    logger = EventLogger()

    event = log_started_event(logger)

    assert logger.get_events() == [event]


def test_logging_event_stores_it_in_memory() -> None:
    logger = EventLogger()

    event = log_started_event(logger)

    assert event in logger.get_events()


def test_get_events_returns_all_logged_events() -> None:
    logger = EventLogger()
    first = log_started_event(logger)
    second = log_started_event(logger)

    assert logger.get_events() == [first, second]


def test_get_events_filters_by_run_id() -> None:
    logger = EventLogger()
    selected_run_id = uuid4()
    other_run_id = uuid4()
    selected_first = log_started_event(logger, selected_run_id)
    log_started_event(logger, other_run_id)
    selected_second = log_started_event(logger, selected_run_id)

    assert logger.get_events(selected_run_id) == [selected_first, selected_second]


def test_get_events_returns_a_copy_of_the_internal_list() -> None:
    logger = EventLogger()
    event = log_started_event(logger)

    returned_events = logger.get_events()
    returned_events.clear()

    assert logger.get_events() == [event]


def test_clear_events_removes_all_events_from_memory() -> None:
    logger = EventLogger()
    log_started_event(logger)

    logger.clear_events()

    assert logger.get_events() == []


def test_configured_sink_receives_each_logged_event() -> None:
    sink = FakeEventSink()
    logger = EventLogger(sink=sink)

    first = log_started_event(logger)
    second = log_started_event(logger)

    assert sink.events == [first, second]


def test_log_event_returns_the_same_event_forwarded_to_sink() -> None:
    sink = FakeEventSink()
    logger = EventLogger(sink=sink)

    event = log_started_event(logger)

    assert sink.events[0] is event


def test_sink_exception_propagates_from_log_event() -> None:
    logger = EventLogger(sink=FailingEventSink())

    with pytest.raises(RuntimeError, match="event persistence failed"):
        log_started_event(logger)


def test_event_remains_in_memory_when_sink_persistence_fails() -> None:
    logger = EventLogger(sink=FailingEventSink())

    with pytest.raises(RuntimeError, match="event persistence failed"):
        log_started_event(logger)

    events = logger.get_events()
    assert len(events) == 1
    assert isinstance(events[0], ExecutionEvent)
