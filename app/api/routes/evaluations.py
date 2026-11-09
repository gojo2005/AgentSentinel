"""Endpoints for reading persisted evaluation results."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.schemas.evaluation import EvaluationListResponse, EvaluationRequest
from app.core.workflow_errors import (
    EvaluationConfigurationError,
    EvaluationFailedError,
    EvaluationPersistenceError,
    ExecutionNotFoundError,
)
from app.evaluation.service import EvaluationService
from app.evaluation.schemas import EvaluationResult
from app.storage.database import get_db
from app.storage.event_repository import ExecutionEventRepository
from app.storage.evaluation_repository import EvaluationResultRepository

router = APIRouter(tags=["evaluations"])
DatabaseSession = Annotated[Session, Depends(get_db)]


@router.post(
    "/executions/{run_id}/evaluations",
    response_model=EvaluationResult,
    status_code=201,
)
def create_execution_evaluation(
    run_id: UUID,
    request: EvaluationRequest,
    session: DatabaseSession,
) -> EvaluationResult:
    """Evaluate one execution and persist a new result."""
    try:
        return EvaluationService(session).evaluate_run(
            run_id,
            question=request.question,
            answer=request.answer,
            reference_answer=request.reference_answer,
            context=request.context,
            evaluator=request.evaluator,
        )
    except ExecutionNotFoundError:
        raise HTTPException(status_code=404, detail="Execution not found.") from None
    except EvaluationConfigurationError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The configured LLM judge is unavailable. "
                f"Correlation ID: {error.correlation_id}"
            ),
        ) from None
    except EvaluationFailedError as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "Evaluation could not be completed. "
                f"Correlation ID: {error.correlation_id}"
            ),
        ) from None
    except EvaluationPersistenceError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Evaluation result could not be saved. "
                f"Correlation ID: {error.correlation_id}"
            ),
        ) from None


@router.get("/evaluations", response_model=EvaluationListResponse)
def list_evaluations(
    session: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EvaluationListResponse:
    """List persisted evaluations newest first, with pagination."""
    results = EvaluationResultRepository(session).get_all()
    results.sort(
        key=lambda result: (result.timestamp, str(result.evaluation_id)),
        reverse=True,
    )
    return EvaluationListResponse(
        items=results[offset : offset + limit],
        total=len(results),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/evaluations/{evaluation_id}",
    response_model=EvaluationResult,
)
def get_evaluation(
    evaluation_id: UUID,
    session: DatabaseSession,
) -> EvaluationResult:
    """Return one evaluation result or raise 404 if it does not exist."""
    result = EvaluationResultRepository(session).get_by_id(evaluation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found.")
    return result


@router.get(
    "/executions/{run_id}/evaluations",
    response_model=list[EvaluationResult],
)
def list_execution_evaluations(
    run_id: UUID,
    session: DatabaseSession,
) -> list[EvaluationResult]:
    """Return evaluations for an existing execution, oldest first."""
    events = ExecutionEventRepository(session).get_by_run_id(run_id)
    if not events:
        raise HTTPException(status_code=404, detail="Execution not found.")
    return EvaluationResultRepository(session).get_by_run_id(run_id)
