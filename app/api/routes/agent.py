"""Agent execution endpoints."""

from fastapi import APIRouter, HTTPException

from app.agent.graph import run_agent
from app.api.schemas.agent import AgentRunRequest, AgentRunResponse
from app.core.composition import create_production_event_logger

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run", response_model=AgentRunResponse)
def run_agent_endpoint(request: AgentRunRequest) -> AgentRunResponse:
    """Run the agent and return its answer with the recorded run identifier."""
    try:
        logger = create_production_event_logger()
        answer = run_agent(request.question, logger=logger)
        events = logger.get_events()
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        ) from None

    run_ids = {event.run_id for event in events}
    if len(run_ids) != 1:
        raise HTTPException(
            status_code=500,
            detail="Agent execution did not produce a valid run identifier.",
        )

    return AgentRunResponse(
        run_id=run_ids.pop(),
        question=request.question,
        answer=answer,
        status="completed",
    )
