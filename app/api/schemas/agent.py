"""Request and response schemas for agent execution."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, StringConstraints


class AgentRunRequest(BaseModel):
    """Request body for running the AgentSentinel agent."""

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
    ]


class AgentRunResponse(BaseModel):
    """Successful result of an agent execution."""

    run_id: UUID
    question: str
    answer: str
    status: Literal["completed"]
