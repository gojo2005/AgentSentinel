"""Manual integration check for persistent AgentSentinel event logging."""

from app.agent.graph import run_agent
from app.core.composition import create_production_event_logger
from app.storage.database import SessionLocal
from app.storage.event_repository import ExecutionEventRepository


def main() -> None:
    """Run the Groq agent and verify its events were persisted."""
    question = "Calculate 125 multiplied by 48."
    logger = create_production_event_logger()

    answer = run_agent(question, logger=logger)
    print(f"Final answer: {answer}")

    in_memory_events = logger.get_events()
    for event in in_memory_events:
        print(
            f"event_type={event.event_type}, "
            f"status={event.status}, "
            f"tool_name={event.tool_name}, "
            f"duration_ms={event.duration_ms}"
        )

    if not in_memory_events:
        raise AssertionError("The agent did not produce any in-memory events.")

    run_ids = {event.run_id for event in in_memory_events}
    if len(run_ids) != 1:
        raise AssertionError("In-memory events do not share one run_id.")
    run_id = next(iter(run_ids))

    with SessionLocal() as session:
        repository = ExecutionEventRepository(session)
        persisted_events = repository.get_by_run_id(run_id)

    print(f"In-memory events: {len(in_memory_events)}")
    print(f"Persisted events: {len(persisted_events)}")

    if any(event.run_id != run_id for event in persisted_events):
        raise AssertionError("Persisted events contain a different run_id.")

    persisted_event_types = {event.event_type for event in persisted_events}
    if not {"run_started", "run_completed"}.issubset(persisted_event_types):
        raise AssertionError(
            "Persisted events must include run_started and run_completed."
        )

    in_memory_event_ids = {event.event_id for event in in_memory_events}
    persisted_event_ids = {event.event_id for event in persisted_events}
    if in_memory_event_ids != persisted_event_ids:
        raise AssertionError(
            "Persisted event IDs do not match the in-memory event IDs."
        )


if __name__ == "__main__":
    main()
