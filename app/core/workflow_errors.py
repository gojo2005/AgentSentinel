"""Safe, application-level errors shared by evaluation and diagnostic workflows."""

import logging
from uuid import UUID, uuid4

_workflow_logger = logging.getLogger("agentsentinel.workflow")


def log_workflow_failure(operation: str, error: Exception) -> UUID:
    """Log safe failure metadata and return its correlation identifier."""
    correlation_id = uuid4()
    _workflow_logger.error(
        "Workflow operation failed: operation=%s correlation_id=%s "
        "exception_type=%s",
        operation,
        correlation_id,
        type(error).__name__,
    )
    return correlation_id


class CorrelatedWorkflowError(RuntimeError):
    """An unexpected workflow failure with a safe correlation identifier."""

    def __init__(self, message: str, correlation_id: UUID) -> None:
        """Store only a safe message and the generated correlation identifier."""
        super().__init__(message)
        self.correlation_id = correlation_id


class ExecutionNotFoundError(LookupError):
    """Raised when a requested run has no persisted execution events."""


class EvaluationConfigurationError(CorrelatedWorkflowError):
    """Raised when the selected LLM judge is not configured."""


class EvaluationFailedError(CorrelatedWorkflowError):
    """Raised when evaluation cannot complete safely."""


class EvaluationPersistenceError(CorrelatedWorkflowError):
    """Raised when an evaluation result cannot be persisted."""


class DiagnosticFailedError(CorrelatedWorkflowError):
    """Raised when diagnostic generation cannot complete safely."""
