"""Tests for SQLAlchemy-backed execution event persistence."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.monitoring.schemas import ExecutionEvent
from app.storage import event_sink
from app.storage.event_repository import ExecutionEventRepository
from app.storage.models import Base


class TrackingSession(Session):
    """SQLAlchemy session that records whether its close method was called."""

    was_closed: bool = False

    def close(self) -> None:
        """Record cleanup before delegating to SQLAlchemy."""
        self.was_closed = True
        super().close()


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Create and clean up an isolated SQLite database for a test."""
    database_path = tmp_path / "event_sink.db"
    test_engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound only to the temporary database."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )


def make_event(
    run_id: UUID,
    *,
    timestamp: datetime | None = None,
    output: object | None = None,
) -> ExecutionEvent:
    """Build a valid execution event with unique identifiers."""
    return ExecutionEvent(
        run_id=run_id,
        event_type="tool_call",
        timestamp=timestamp or datetime.now(timezone.utc),
        agent_name="test-agent",
        status="success",
        tool_name="test-tool",
        input={"query": "example"},
        output={"result": "ok"} if output is None else output,
        metadata={"source": "test"},
    )


def test_write_event_persists_event_to_temporary_database(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker[Session],
) -> None:
    event = make_event(uuid4())
    monkeypatch.setattr(event_sink, "SessionLocal", session_factory)

    event_sink.SQLAlchemyEventSink().write_event(event)

    with session_factory() as session:
        persisted = ExecutionEventRepository(session).get_by_id(event.event_id)
    assert persisted == event


def test_multiple_events_use_distinct_sessions_and_are_persisted(
    monkeypatch: pytest.MonkeyPatch,
    engine: Engine,
) -> None:
    created_sessions: list[TrackingSession] = []
    tracked_factory = sessionmaker(
        bind=engine,
        class_=TrackingSession,
        autoflush=False,
        autocommit=False,
    )

    def create_tracked_session() -> TrackingSession:
        session = tracked_factory()
        created_sessions.append(session)
        return session

    monkeypatch.setattr(event_sink, "SessionLocal", create_tracked_session)
    sink = event_sink.SQLAlchemyEventSink()
    run_id = uuid4()
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [
        make_event(run_id, timestamp=base_time),
        make_event(run_id, timestamp=base_time + timedelta(seconds=1)),
    ]

    for event in events:
        sink.write_event(event)

    assert len(created_sessions) == len(events)
    assert len({id(session) for session in created_sessions}) == len(events)
    assert all(session.was_closed for session in created_sessions)

    with tracked_factory() as session:
        persisted = ExecutionEventRepository(session).get_by_run_id(run_id)
    assert persisted == events


def test_session_is_closed_after_successful_save(
    monkeypatch: pytest.MonkeyPatch,
    engine: Engine,
) -> None:
    created_sessions: list[TrackingSession] = []
    tracked_factory = sessionmaker(
        bind=engine,
        class_=TrackingSession,
        autoflush=False,
        autocommit=False,
    )

    def create_tracked_session() -> TrackingSession:
        session = tracked_factory()
        created_sessions.append(session)
        return session

    monkeypatch.setattr(event_sink, "SessionLocal", create_tracked_session)

    event_sink.SQLAlchemyEventSink().write_event(make_event(uuid4()))

    assert len(created_sessions) == 1
    assert created_sessions[0].was_closed


def test_session_is_closed_and_error_propagates_when_save_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = FakeSession()

    class FailingRepository:
        def __init__(self, injected_session: FakeSession) -> None:
            assert injected_session is session

        def save(self, event: ExecutionEvent) -> ExecutionEvent:
            raise RuntimeError("simulated persistence failure")

    monkeypatch.setattr(event_sink, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        event_sink,
        "ExecutionEventRepository",
        FailingRepository,
    )

    with pytest.raises(RuntimeError, match="simulated persistence failure"):
        event_sink.SQLAlchemyEventSink().write_event(make_event(uuid4()))

    assert session.closed
