"""Application workflow for diagnosing and persisting an execution."""

from uuid import UUID

from sqlalchemy.orm import Session

from app.core.workflow_errors import (
    DiagnosticFailedError,
    ExecutionNotFoundError,
    log_workflow_failure,
)
from app.diagnostics.diagnostic_engine import RuleBasedDiagnosticEngine
from app.diagnostics.schemas import DiagnosticReport
from app.storage.diagnostic_repository import DiagnosticReportRepository
from app.storage.event_repository import ExecutionEventRepository
from app.storage.evaluation_repository import EvaluationResultRepository


class DiagnosticService:
    """Coordinate persisted evidence, diagnosis, and report persistence."""

    def __init__(
        self,
        session: Session,
        *,
        diagnostic_engine: RuleBasedDiagnosticEngine | None = None,
    ) -> None:
        """Create a workflow with an injectable diagnostic engine."""
        self._events = ExecutionEventRepository(session)
        self._evaluations = EvaluationResultRepository(session)
        self._reports = DiagnosticReportRepository(session)
        self._diagnostic_engine = diagnostic_engine or RuleBasedDiagnosticEngine()

    def diagnose_run(self, run_id: UUID) -> DiagnosticReport:
        """Diagnose a known run using its latest evaluation when available."""
        try:
            events = self._events.get_by_run_id(run_id)
        except Exception as error:
            correlation_id = log_workflow_failure("diagnostic_event_lookup", error)
            raise DiagnosticFailedError(
                "Diagnostics could not be completed.",
                correlation_id,
            ) from error
        if not events:
            raise ExecutionNotFoundError("Execution not found.")

        try:
            evaluations = self._evaluations.get_by_run_id(run_id)
            evaluation = max(
                evaluations,
                key=lambda result: (result.timestamp, str(result.evaluation_id)),
                default=None,
            )
            if evaluation is not None and evaluation.run_id != run_id:
                raise ValueError("Evaluation result belongs to another run.")
            report = self._diagnostic_engine.diagnose(
                run_id=run_id,
                events=events,
                evaluation=evaluation,
            )
            if report.run_id != run_id:
                raise ValueError("Diagnostic report belongs to another run.")
            return self._reports.save(report)
        except Exception as error:
            correlation_id = log_workflow_failure("diagnostic_generation", error)
            raise DiagnosticFailedError(
                "Diagnostics could not be completed.",
                correlation_id,
            ) from error
