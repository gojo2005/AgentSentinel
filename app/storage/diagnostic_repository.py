"""Persistence operations for diagnostic reports and their findings."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.diagnostics.schemas import DiagnosticFinding, DiagnosticReport
from app.storage.models import DiagnosticFindingORM, DiagnosticReportORM


class DiagnosticReportRepository:
    """Read and write diagnostic reports using an injected SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        """Store the session supplied by the caller."""
        self._session = session

    def save(self, report: DiagnosticReport) -> DiagnosticReport:
        """Persist a report with all findings in one transaction."""
        record = DiagnosticReportORM(
            report_id=report.report_id,
            run_id=report.run_id,
            timestamp=report.timestamp,
            overall_status=report.overall_status,
            summary=report.summary,
            metadata_json=report.metadata,
            findings=[
                DiagnosticFindingORM(
                    finding_id=finding.finding_id,
                    category=finding.category,
                    severity=finding.severity,
                    title=finding.title,
                    description=finding.description,
                    evidence=finding.evidence,
                    recommendation=finding.recommendation,
                    metadata_json=finding.metadata,
                )
                for finding in report.findings
            ],
        )
        try:
            self._session.add(record)
            self._session.commit()
            self._session.refresh(record)
        except Exception:
            self._session.rollback()
            raise
        return self._to_schema(record)

    def get_by_id(self, report_id: UUID) -> DiagnosticReport | None:
        """Return a report and its findings by ID, or None if absent."""
        statement = (
            select(DiagnosticReportORM)
            .options(selectinload(DiagnosticReportORM.findings))
            .where(DiagnosticReportORM.report_id == report_id)
        )
        record = self._session.scalars(statement).first()
        if record is None:
            return None
        return self._to_schema(record)

    def get_by_run_id(self, run_id: UUID) -> list[DiagnosticReport]:
        """Return reports for a run ordered by timestamp ascending."""
        statement = (
            select(DiagnosticReportORM)
            .options(selectinload(DiagnosticReportORM.findings))
            .where(DiagnosticReportORM.run_id == run_id)
            .order_by(DiagnosticReportORM.timestamp.asc())
        )
        records = self._session.scalars(statement).all()
        return [self._to_schema(record) for record in records]

    def get_all(self) -> list[DiagnosticReport]:
        """Return all reports ordered by timestamp ascending."""
        statement = (
            select(DiagnosticReportORM)
            .options(selectinload(DiagnosticReportORM.findings))
            .order_by(DiagnosticReportORM.timestamp.asc())
        )
        records = self._session.scalars(statement).all()
        return [self._to_schema(record) for record in records]

    @staticmethod
    def _to_schema(record: DiagnosticReportORM) -> DiagnosticReport:
        """Convert an ORM report and its findings to Pydantic schemas."""
        findings = [
            DiagnosticFinding(
                finding_id=finding.finding_id,
                category=finding.category,
                severity=finding.severity,
                title=finding.title,
                description=finding.description,
                evidence=finding.evidence,
                recommendation=finding.recommendation,
                metadata=finding.metadata_json,
            )
            for finding in record.findings
        ]
        return DiagnosticReport(
            report_id=record.report_id,
            run_id=record.run_id,
            timestamp=record.timestamp,
            overall_status=record.overall_status,
            findings=findings,
            summary=record.summary,
            metadata=record.metadata_json,
        )
