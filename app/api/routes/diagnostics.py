"""Endpoints for reading persisted diagnostic reports."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.schemas.diagnostic import DiagnosticReportListResponse
from app.core.workflow_errors import DiagnosticFailedError, ExecutionNotFoundError
from app.diagnostics.service import DiagnosticService
from app.diagnostics.schemas import DiagnosticReport
from app.storage.database import get_db
from app.storage.diagnostic_repository import DiagnosticReportRepository
from app.storage.event_repository import ExecutionEventRepository

router = APIRouter(tags=["diagnostics"])
DatabaseSession = Annotated[Session, Depends(get_db)]


@router.post(
    "/executions/{run_id}/diagnostics",
    response_model=DiagnosticReport,
    status_code=201,
)
def create_execution_diagnostic_report(
    run_id: UUID,
    session: DatabaseSession,
) -> DiagnosticReport:
    """Diagnose one persisted execution and save a new report."""
    try:
        return DiagnosticService(session).diagnose_run(run_id)
    except ExecutionNotFoundError:
        raise HTTPException(status_code=404, detail="Execution not found.") from None
    except DiagnosticFailedError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Diagnostics could not be completed. "
                f"Correlation ID: {error.correlation_id}"
            ),
        ) from None


@router.get("/diagnostics", response_model=DiagnosticReportListResponse)
def list_diagnostic_reports(
    session: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DiagnosticReportListResponse:
    """List diagnostic reports newest first, with pagination."""
    reports = DiagnosticReportRepository(session).get_all()
    reports.sort(
        key=lambda report: (report.timestamp, str(report.report_id)),
        reverse=True,
    )
    return DiagnosticReportListResponse(
        items=reports[offset : offset + limit],
        total=len(reports),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/diagnostics/{report_id}",
    response_model=DiagnosticReport,
)
def get_diagnostic_report(
    report_id: UUID,
    session: DatabaseSession,
) -> DiagnosticReport:
    """Return one report with all findings or raise 404 if absent."""
    report = DiagnosticReportRepository(session).get_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Diagnostic report not found.")
    return report


@router.get(
    "/executions/{run_id}/diagnostics",
    response_model=list[DiagnosticReport],
)
def list_execution_diagnostic_reports(
    run_id: UUID,
    session: DatabaseSession,
) -> list[DiagnosticReport]:
    """Return reports for an existing execution, oldest first."""
    events = ExecutionEventRepository(session).get_by_run_id(run_id)
    if not events:
        raise HTTPException(status_code=404, detail="Execution not found.")
    return DiagnosticReportRepository(session).get_by_run_id(run_id)
