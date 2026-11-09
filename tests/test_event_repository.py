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
from app.storage.event_repository import ExecutionEventRepository
from app.storage.models import Base, ExecutionEventORM


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Create and clean up an isolated SQLite database for a test."""
    database_path = tmp_path / "events.db"
    test_engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Provide a session bound to the temporary test database."""
    test_session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    with test_session_factory() as test_session:
        yield test_session


def make_event(
    run_id: UUID,
    *,
    timestamp: datetime | None = None,
    event_type: str = "tool_call",
    status: str = "success",
    input: object | None = None,
    output: object | None = None,
    metadata: dict[str, object] | None = None,
) -> ExecutionEvent:
    """Build a valid event with unique identifiers and an aware timestamp."""
    return ExecutionEvent(
        run_id=run_id,
        event_type=event_type,
        timestamp=timestamp or datetime.now(timezone.utc),
        agent_name="test-agent",
        status=status,
        tool_name="test-tool",
        input={"query": "example"} if input is None else input,
        output={"result": "ok"} if output is None else output,
        metadata={"source": "test"} if metadata is None else metadata,
    )


def test_save_persists_event_and_preserves_fields(session: Session) -> None:
    run_id = uuid4()
    timestamp = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    event = make_event(
        run_id,
        timestamp=timestamp,
        input={"question": "What is 2 + 2?"},
        output={"answer": 4},
        metadata={
            "attempt": 1,
            "tags": ["test"],
            "token_usage": {
                "available": True,
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            },
            "estimated_cost": {
                "available": True,
                "amount_usd": "0.000092000000",
                "currency": "USD",
                "pricing_version": "test-version",
            },
        },
    )
    repository = ExecutionEventRepository(session)

    saved = repository.save(event)

    assert saved.event_id == event.event_id
    assert saved.run_id == event.run_id
    assert saved.timestamp == event.timestamp
    assert saved.input == event.input
    assert saved.output == event.output
    assert saved.metadata == event.metadata

    persisted = session.get(ExecutionEventORM, event.event_id)
    assert persisted is not None
    assert persisted.run_id == event.run_id
    assert persisted.timestamp == event.timestamp
    assert persisted.input == event.input
    assert persisted.output == event.output
    assert persisted.metadata_json == event.metadata


def test_get_by_id_returns_event_and_none_for_missing_id(session: Session) -> None:
    event = make_event(uuid4())
    repository = ExecutionEventRepository(session)
    repository.save(event)

    retrieved = repository.get_by_id(event.event_id)

    assert isinstance(retrieved, ExecutionEvent)
    assert retrieved == event
    assert repository.get_by_id(uuid4()) is None


def test_get_by_run_id_filters_and_orders_events(session: Session) -> None:
    repository = ExecutionEventRepository(session)
    requested_run_id = uuid4()
    other_run_id = uuid4()
    base_time = datetime(2026, 2, 1, tzinfo=timezone.utc)
    later_event = make_event(
        requested_run_id,
        timestamp=base_time + timedelta(minutes=2),
    )
    other_run_event = make_event(
        other_run_id,
        timestamp=base_time + timedelta(minutes=1),
    )
    earlier_event = make_event(requested_run_id, timestamp=base_time)

    repository.save(later_event)
    repository.save(other_run_event)
    repository.save(earlier_event)

    events = repository.get_by_run_id(requested_run_id)

    assert [event.event_id for event in events] == [
        earlier_event.event_id,
        later_event.event_id,
    ]
    assert all(event.run_id == requested_run_id for event in events)
    assert all(isinstance(event, ExecutionEvent) for event in events)


def test_get_all_returns_events_from_multiple_runs_in_timestamp_order(
    session: Session,
) -> None:
    repository = ExecutionEventRepository(session)
    first_run_id = uuid4()
    second_run_id = uuid4()
    base_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
    newest = make_event(
        first_run_id,
        timestamp=base_time + timedelta(minutes=2),
    )
    oldest = make_event(second_run_id, timestamp=base_time)
    middle = make_event(
        first_run_id,
        timestamp=base_time + timedelta(minutes=1),
    )

    repository.save(newest)
    repository.save(oldest)
    repository.save(middle)

    events = repository.get_all()

    assert [event.event_id for event in events] == [
        oldest.event_id,
        middle.event_id,
        newest.event_id,
    ]
    assert {event.run_id for event in events} == {first_run_id, second_run_id}
    assert all(isinstance(event, ExecutionEvent) for event in events)
