"""Response schemas for persisted evaluation results."""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

from app.evaluation.schemas import EvaluationResult


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvaluationRequest(BaseModel):
    """Input for evaluating and persisting one execution result."""

    question: NonEmptyText
    answer: NonEmptyText
    reference_answer: NonEmptyText | None = None
    context: NonEmptyText | None = None
    evaluator: Literal["rule_based", "llm_judge"] = "rule_based"


class EvaluationListResponse(BaseModel):
    """Paginated evaluation results."""

    items: list[EvaluationResult]
    total: int
    limit: int
    offset: int
