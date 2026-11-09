"""SQLAlchemy-backed implementation of the monitoring event sink."""

from app.monitoring.interfaces import EventSink
from app.monitoring.schemas import ExecutionEvent
from app.storage.database import SessionLocal
from app.storage.event_repository import ExecutionEventRepository


class SQLAlchemyEventSink(EventSink):
    """Persist each execution event using a short-lived database session."""

    def write_event(self, event: ExecutionEvent) -> None:
        """Persist an event and always close the session used to write it."""
        session = SessionLocal()
        try:
            repository = ExecutionEventRepository(session)
            repository.save(event)
        finally:
            session.close()
