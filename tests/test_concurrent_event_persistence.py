
"""Tests for concurrent persistence of AgentSentinel execution events."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.monitoring.schemas import ExecutionEvent
from app.storage import event_sink
from app.storage.event_repository import ExecutionEventRepository
from app.storage.models import Base


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Create and dispose of an isolated temporary SQLite database."""
    database_path = tmp_path / "concurrent_events.db"
    test_engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create sessions bound to the temporary database."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )


def make_event(index: int) -> ExecutionEvent:
    """Create a valid event with unique event and run IDs."""
    return ExecutionEvent(
        event_id=uuid4(),
        run_id=uuid4(),
        event_type="tool_call",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)
        + timedelta(milliseconds=index),
        agent_name="concurrent-test-agent",
        status="success",
        tool_name="test-tool",
        input={"event_index": index},
        output={"result": f"result-{index}"},
        metadata={"test": "concurrent persistence", "index": index},
    )


def test_concurrent_event_writes_are_all_persisted(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
) -> None:
    """Verify that concurrent writes persist every submitted event."""
    events = [make_event(index) for index in range(20)]
    monkeypatch.setattr(event_sink, "SessionLocal", session_factory)
    sink = event_sink.SQLAlchemyEventSink()

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(sink.write_event, events))

    with session_factory() as session:
        persisted_events = ExecutionEventRepository(session).get_all()

    submitted_ids = {event.event_id for event in events}
    persisted_ids = {event.event_id for event in persisted_events}

    assert len(persisted_events) == len(events)
    assert persisted_ids == submitted_ids
    assert {event.run_id for event in persisted_events} == {
        event.run_id for event in events
    }


def test_concurrent_event_writes_create_and_close_one_session_each(
    monkeypatch: pytest.MonkeyPatch,
    engine: Engine,
) -> None:
    """Verify each concurrent write creates and closes its own session."""
    tracking_lock = Lock()
    created_sessions: list[Session] = []
    closed_sessions: set[Session] = set()

    class TrackingSession(Session):
        """Session that records closure for concurrent lifecycle assertions."""

        def close(self) -> None:
            super().close()
            with tracking_lock:
                closed_sessions.add(self)

    tracked_factory = sessionmaker(
        bind=engine,
        class_=TrackingSession,
        autoflush=False,
        autocommit=False,
    )

    def create_tracked_session() -> TrackingSession:
        session = tracked_factory()
        with tracking_lock:
            created_sessions.append(session)
        return session

    events = [make_event(index) for index in range(12)]
    monkeypatch.setattr(event_sink, "SessionLocal", create_tracked_session)
    sink = event_sink.SQLAlchemyEventSink()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(sink.write_event, events))

    assert len(created_sessions) == len(events)

    with tracking_lock:
        assert set(created_sessions) == closed_sessions


