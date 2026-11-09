"""Persistence operations for normalized execution events."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.monitoring.schemas import ExecutionEvent
from app.storage.models import ExecutionEventORM


class ExecutionEventRepository:
    """Read and write execution events using an injected SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        """Store the session supplied by the caller."""
        self._session = session

    def save(self, event: ExecutionEvent) -> ExecutionEvent:
        """Persist an event and return its validated Pydantic representation."""
        record = ExecutionEventORM(
            event_id=event.event_id,
            run_id=event.run_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            agent_name=event.agent_name,
            provider=event.provider,
            model=event.model,
            tool_name=event.tool_name,
            status=event.status,
            input=event.input,
            output=event.output,
            duration_ms=event.duration_ms,
            error=event.error,
            metadata_json=event.metadata,
        )
        try:
            self._session.add(record)
            self._session.commit()
            self._session.refresh(record)
        except Exception:
            self._session.rollback()
            raise
        return self._to_schema(record)

    def get_by_id(self, event_id: UUID) -> ExecutionEvent | None:
        """Return the event with the given ID, or None if it does not exist."""
        statement = select(ExecutionEventORM).where(
            ExecutionEventORM.event_id == event_id
        )
        record = self._session.scalars(statement).first()
        if record is None:
            return None
        return self._to_schema(record)

    def get_by_run_id(self, run_id: UUID) -> list[ExecutionEvent]:
        """Return events for one run ordered by timestamp ascending."""
        statement = (
            select(ExecutionEventORM)
            .where(ExecutionEventORM.run_id == run_id)
            .order_by(ExecutionEventORM.timestamp.asc())
        )
        records = self._session.scalars(statement).all()
        return [self._to_schema(record) for record in records]

    def get_all(self) -> list[ExecutionEvent]:
        """Return all events ordered by timestamp ascending."""
        statement = select(ExecutionEventORM).order_by(
            ExecutionEventORM.timestamp.asc()
        )
        records = self._session.scalars(statement).all()
        return [self._to_schema(record) for record in records]

    @staticmethod
    def _to_schema(record: ExecutionEventORM) -> ExecutionEvent:
        """Convert an ORM event to the public Pydantic schema."""
        return ExecutionEvent(
            event_id=record.event_id,
            run_id=record.run_id,
            event_type=record.event_type,
            timestamp=record.timestamp,
            agent_name=record.agent_name,
            provider=record.provider,
            model=record.model,
            tool_name=record.tool_name,
            status=record.status,
            input=record.input,
            output=record.output,
            duration_ms=record.duration_ms,
            error=record.error,
            metadata=record.metadata_json,
        )
