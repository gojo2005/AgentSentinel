"""Verify that execution events persist in the configured SQLite database."""

from collections import Counter

from app.storage.database import SessionLocal
from app.storage.event_repository import ExecutionEventRepository


def main() -> None:
    """Print persisted events and counts grouped by run."""
    with SessionLocal() as session:
        repository = ExecutionEventRepository(session)
        events = repository.get_all()

    if not events:
        raise AssertionError("The database contains no persisted execution events.")

    print(f"Total events: {len(events)}")
    for event in events:
        print(
            f"event_id={event.event_id}, "
            f"run_id={event.run_id}, "
            f"event_type={event.event_type}, "
            f"status={event.status}, "
            f"timestamp={event.timestamp}"
        )

    events_per_run = Counter(event.run_id for event in events)
    for run_id, event_count in events_per_run.items():
        print(f"run_id={run_id}: {event_count} event(s)")


if __name__ == "__main__":
    main()
