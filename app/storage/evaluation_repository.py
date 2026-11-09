"""Persistence operations for evaluation results."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluation.schemas import EvaluationResult
from app.storage.models import EvaluationResultORM


class EvaluationResultRepository:
    """Read and write evaluation results using an injected SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        """Store the session supplied by the caller."""
        self._session = session

    def save(self, result: EvaluationResult) -> EvaluationResult:
        """Persist a result and return its validated Pydantic representation."""
        record = EvaluationResultORM(
            evaluation_id=result.evaluation_id,
            run_id=result.run_id,
            timestamp=result.timestamp,
            correctness_score=result.correctness_score,
            relevance_score=result.relevance_score,
            context_relevance_score=result.context_relevance_score,
            hallucination_score=result.hallucination_score,
            overall_score=result.overall_score,
            passed=result.passed,
            feedback=result.feedback,
            evaluator=result.evaluator,
            metadata_json=result.metadata,
        )
        try:
            self._session.add(record)
            self._session.commit()
            self._session.refresh(record)
        except Exception:
            self._session.rollback()
            raise
        return self._to_schema(record)

    def get_by_id(self, evaluation_id: UUID) -> EvaluationResult | None:
        """Return the result with the given ID, or None if it does not exist."""
        statement = select(EvaluationResultORM).where(
            EvaluationResultORM.evaluation_id == evaluation_id
        )
        record = self._session.scalars(statement).first()
        if record is None:
            return None
        return self._to_schema(record)

    def get_by_run_id(self, run_id: UUID) -> list[EvaluationResult]:
        """Return results for one run ordered by timestamp ascending."""
        statement = (
            select(EvaluationResultORM)
            .where(EvaluationResultORM.run_id == run_id)
            .order_by(EvaluationResultORM.timestamp.asc())
        )
        records = self._session.scalars(statement).all()
        return [self._to_schema(record) for record in records]

    def get_all(self) -> list[EvaluationResult]:
        """Return all evaluation results ordered by timestamp ascending."""
        statement = select(EvaluationResultORM).order_by(
            EvaluationResultORM.timestamp.asc()
        )
        records = self._session.scalars(statement).all()
        return [self._to_schema(record) for record in records]

    @staticmethod
    def _to_schema(record: EvaluationResultORM) -> EvaluationResult:
        """Convert an ORM result to the public Pydantic schema."""
        return EvaluationResult(
            evaluation_id=record.evaluation_id,
            run_id=record.run_id,
            timestamp=record.timestamp,
            correctness_score=record.correctness_score,
            relevance_score=record.relevance_score,
            context_relevance_score=record.context_relevance_score,
            hallucination_score=record.hallucination_score,
            overall_score=record.overall_score,
            passed=record.passed,
            feedback=record.feedback,
            evaluator=record.evaluator,
            metadata=record.metadata_json,
        )
