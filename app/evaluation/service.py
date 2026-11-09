"""Application workflow for evaluating and persisting an execution."""

from collections.abc import Callable
from typing import Literal, Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.workflow_errors import (
    EvaluationConfigurationError,
    EvaluationFailedError,
    EvaluationPersistenceError,
    ExecutionNotFoundError,
    log_workflow_failure,
)
from app.evaluation.evaluator import RuleBasedEvaluator
from app.evaluation.llm_judge import LLMJudge
from app.evaluation.schemas import EvaluationResult
from app.monitoring.schemas import ExecutionEvent
from app.storage.event_repository import ExecutionEventRepository
from app.storage.evaluation_repository import EvaluationResultRepository

EvaluationMethod = Literal["rule_based", "llm_judge"]


class EvaluationRunner(Protocol):
    """Common evaluation interface used by both available evaluators."""

    def evaluate(
        self,
        question: str,
        answer: str,
        run_id: UUID,
        reference_answer: str | None = None,
        context: str | None = None,
    ) -> EvaluationResult:
        """Return a result for the supplied execution content."""


class EvaluationService:
    """Coordinate execution lookup, evaluation, and result persistence."""

    def __init__(
        self,
        session: Session,
        *,
        rule_evaluator: EvaluationRunner | None = None,
        llm_judge_factory: Callable[[], EvaluationRunner] | None = None,
    ) -> None:
        """Create a workflow with injectable evaluators for deterministic tests."""
        self._events = ExecutionEventRepository(session)
        self._results = EvaluationResultRepository(session)
        self._rule_evaluator = (
            RuleBasedEvaluator() if rule_evaluator is None else rule_evaluator
        )
        self._llm_judge_factory = (
            LLMJudge if llm_judge_factory is None else llm_judge_factory
        )

    def evaluate_run(
        self,
        run_id: UUID,
        *,
        question: str,
        answer: str,
        reference_answer: str | None = None,
        context: str | None = None,
        evaluator: EvaluationMethod = "rule_based",
    ) -> EvaluationResult:
        """Evaluate a known run and save a new historical result."""
        self._validate_inputs(question, answer, reference_answer, context)
        if evaluator not in {"rule_based", "llm_judge"}:
            raise ValueError("evaluator must be 'rule_based' or 'llm_judge'.")
        try:
            events = self._events.get_by_run_id(run_id)
        except Exception as error:
            correlation_id = log_workflow_failure("evaluation_lookup", error)
            raise EvaluationFailedError(
                "Evaluation could not be completed.",
                correlation_id,
            ) from error
        if not events:
            raise ExecutionNotFoundError("Execution not found.")

        if evaluator == "rule_based":
            result = self._evaluate_with_rules(
                question,
                answer,
                run_id,
                reference_answer,
                context,
            )
        else:
            result = self._evaluate_with_judge(
                question,
                answer,
                run_id,
                reference_answer,
                context,
            )
        if result.run_id != run_id:
            error = ValueError("Evaluator returned a result for another run.")
            correlation_id = log_workflow_failure("evaluation_association", error)
            raise EvaluationFailedError(
                "Evaluation could not be completed.",
                correlation_id,
            ) from error
        try:
            return self._results.save(result)
        except Exception as error:
            correlation_id = log_workflow_failure("evaluation_persistence", error)
            raise EvaluationPersistenceError(
                "Evaluation result could not be saved.",
                correlation_id,
            ) from error

    def _evaluate_with_rules(
        self,
        question: str,
        answer: str,
        run_id: UUID,
        reference_answer: str | None,
        context: str | None,
    ) -> EvaluationResult:
        try:
            return self._rule_evaluator.evaluate(
                question=question,
                answer=answer,
                run_id=run_id,
                reference_answer=reference_answer,
                context=context,
            )
        except Exception as error:
            correlation_id = log_workflow_failure("rule_evaluation", error)
            raise EvaluationFailedError(
                "Evaluation could not be completed.",
                correlation_id,
            ) from error

    def _evaluate_with_judge(
        self,
        question: str,
        answer: str,
        run_id: UUID,
        reference_answer: str | None,
        context: str | None,
    ) -> EvaluationResult:
        try:
            judge = self._llm_judge_factory()
        except Exception as error:
            correlation_id = log_workflow_failure("llm_judge_configuration", error)
            raise EvaluationConfigurationError(
                "The configured LLM judge is unavailable.",
                correlation_id,
            ) from error

        try:
            return judge.evaluate(
                question=question,
                answer=answer,
                run_id=run_id,
                reference_answer=reference_answer,
                context=context,
            )
        except Exception as error:
            correlation_id = log_workflow_failure("llm_judge_evaluation", error)
            raise EvaluationFailedError(
                "Evaluation could not be completed.",
                correlation_id,
            ) from error

    @staticmethod
    def _validate_inputs(
        question: str,
        answer: str,
        reference_answer: str | None,
        context: str | None,
    ) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string.")
        if reference_answer is not None and not reference_answer.strip():
            raise ValueError("reference_answer must not be empty when provided.")
        if context is not None and not context.strip():
            raise ValueError("context must not be empty when provided.")
