"""Provider-independent schemas for AgentSentinel evaluation results."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Scores and feedback produced by evaluating one agent execution."""

    evaluation_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this evaluation result.",
    )
    run_id: UUID = Field(
        description="Identifier of the agent execution being evaluated.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time when this evaluation result was created.",
    )
    correctness_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional score for the factual correctness of the response.",
    )
    relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional score for the response's relevance to the request.",
    )
    context_relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional score for relevance to the supplied context.",
    )
    hallucination_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Degree of unsupported content: 0 means no detected hallucination "
            "and 1 means highly unsupported content."
        ),
    )
    overall_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional overall quality score from 0 to 100.",
    )
    passed: bool | None = Field(
        default=None,
        description="Whether the result meets the configured quality threshold.",
    )
    feedback: str | None = Field(
        default=None,
        description="Optional explanation from the evaluator.",
    )
    evaluator: str = Field(
        description="Name of the evaluation method, such as rule_based or llm_judge.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation details.",
    )


class EvaluationSummary(BaseModel):
    """Aggregate evaluation scores and counts across multiple agent runs."""

    total_evaluations: int = Field(
        ge=0,
        description="Total number of evaluation results included in the summary.",
    )
    average_correctness_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Average correctness score, from 0 to 1, when available.",
    )
    average_relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Average relevance score, from 0 to 1, when available.",
    )
    average_context_relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Average context relevance score, from 0 to 1, when available.",
    )
    average_hallucination_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Average hallucination score, from 0 to 1, when available.",
    )
    average_overall_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Average overall quality score, from 0 to 100, when available.",
    )
    pass_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Fraction of evaluations that passed, from 0 to 1, when available.",
    )
    evaluator_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Count of evaluation results grouped by evaluator name.",
    )
