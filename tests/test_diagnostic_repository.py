"""Tests for SQLAlchemy-backed diagnostic report persistence."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.diagnostics.schemas import DiagnosticFinding, DiagnosticReport
from app.storage.diagnostic_repository import DiagnosticReportRepository
from app.storage.models import Base, DiagnosticFindingORM, DiagnosticReportORM


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Create and clean up an isolated SQLite database for a test."""
    database_path = tmp_path / "diagnostics.db"
    test_engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Provide a session bound to the temporary test database."""
    test_session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    with test_session_factory() as test_session:
        yield test_session


def make_finding(
    *,
    finding_id: UUID | None = None,
    category: str = "reliability",
    severity: str = "medium",
    title: str = "Slow execution",
    description: str = "The run exceeded the expected duration.",
    evidence: list[str] | None = None,
    recommendation: str = "Review execution latency.",
    metadata: dict[str, object] | None = None,
) -> DiagnosticFinding:
    """Build a valid diagnostic finding with unique identifiers."""
    values: dict[str, object] = {
        "category": category,
        "severity": severity,
        "title": title,
        "description": description,
        "evidence": ["duration_ms=12000"] if evidence is None else evidence,
        "recommendation": recommendation,
        "metadata": {"rule": "latency"} if metadata is None else metadata,
    }
    if finding_id is not None:
        values["finding_id"] = finding_id
    return DiagnosticFinding(**values)


def make_report(
    run_id: UUID,
    *,
    report_id: UUID | None = None,
    timestamp: datetime | None = None,
    overall_status: str = "warning",
    findings: list[DiagnosticFinding] | None = None,
    summary: str = "The run completed with diagnostic findings.",
    metadata: dict[str, object] | None = None,
) -> DiagnosticReport:
    """Build a valid diagnostic report with a unique ID and aware timestamp."""
    values: dict[str, object] = {
        "run_id": run_id,
        "timestamp": timestamp or datetime.now(timezone.utc),
        "overall_status": overall_status,
        "findings": [] if findings is None else findings,
        "summary": summary,
        "metadata": {"source": "test"} if metadata is None else metadata,
    }
    if report_id is not None:
        values["report_id"] = report_id
    return DiagnosticReport(**values)


def test_save_persists_report_and_all_nested_finding_fields(
    session: Session,
) -> None:
    run_id = uuid4()
    timestamp = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    report = make_report(
        run_id,
        timestamp=timestamp,
        overall_status="critical",
        summary="The run failed and requires investigation.",
        metadata={"source": "integration-test", "attempt": 2},
        findings=[
            make_finding(
                category="reliability",
                severity="critical",
                title="Run failed",
                description="The agent run ended with a failure.",
                evidence=["run_failed", "database unavailable"],
                recommendation="Inspect the failing service.",
                metadata={"component": "database", "retryable": True},
            ),
            make_finding(
                category="quality",
                severity="high",
                title="Low relevance",
                description="The response did not address the question.",
                evidence=["relevance_score=0.2"],
                recommendation="Review response generation.",
                metadata={"score": 0.2},
            ),
        ],
    )
    repository = DiagnosticReportRepository(session)

    saved = repository.save(report)

    assert saved.report_id == report.report_id
    assert saved.run_id == report.run_id
    assert saved.timestamp == report.timestamp
    assert saved.overall_status == report.overall_status
    assert saved.summary == report.summary
    assert saved.metadata == report.metadata
    assert saved.findings == report.findings

    persisted_report = session.get(DiagnosticReportORM, report.report_id)
    assert persisted_report is not None
    assert persisted_report.run_id == report.run_id
    assert persisted_report.timestamp == report.timestamp
    assert persisted_report.overall_status == report.overall_status
    assert persisted_report.summary == report.summary
    assert persisted_report.metadata_json == report.metadata

    persisted_findings = session.scalars(
        select(DiagnosticFindingORM).where(
            DiagnosticFindingORM.report_id == report.report_id
        )
    ).all()
    assert len(persisted_findings) == len(report.findings)
    persisted_by_id = {finding.finding_id: finding for finding in persisted_findings}
    for expected in report.findings:
        persisted = persisted_by_id[expected.finding_id]
        assert persisted.report_id == report.report_id
        assert persisted.category == expected.category
        assert persisted.severity == expected.severity
        assert persisted.title == expected.title
        assert persisted.description == expected.description
        assert persisted.evidence == expected.evidence
        assert persisted.recommendation == expected.recommendation
        assert persisted.metadata_json == expected.metadata


def test_save_report_without_findings(session: Session) -> None:
    report = make_report(
        uuid4(),
        overall_status="healthy",
        findings=[],
        metadata={},
    )

    saved = DiagnosticReportRepository(session).save(report)

    assert saved.report_id == report.report_id
    assert saved.findings == []
    assert saved.metadata == {}


def test_get_by_id_returns_report_with_all_findings(session: Session) -> None:
    report = make_report(
        uuid4(),
        findings=[
            make_finding(title="First finding"),
            make_finding(title="Second finding", severity="high"),
        ],
    )
    repository = DiagnosticReportRepository(session)
    repository.save(report)

    retrieved = repository.get_by_id(report.report_id)

    assert retrieved is not None
    assert retrieved == report
    assert [finding.finding_id for finding in retrieved.findings] == [
        finding.finding_id for finding in report.findings
    ]


def test_get_by_id_returns_none_for_unknown_report(session: Session) -> None:
    assert DiagnosticReportRepository(session).get_by_id(uuid4()) is None


def test_get_by_run_id_filters_reports_and_orders_by_timestamp(
    session: Session,
) -> None:
    repository = DiagnosticReportRepository(session)
    requested_run_id = uuid4()
    other_run_id = uuid4()
    base_time = datetime(2026, 2, 1, tzinfo=timezone.utc)
    later_report = make_report(
        requested_run_id,
        timestamp=base_time + timedelta(minutes=2),
    )
    other_run_report = make_report(
        other_run_id,
        timestamp=base_time + timedelta(minutes=1),
    )
    earlier_report = make_report(requested_run_id, timestamp=base_time)

    repository.save(later_report)
    repository.save(other_run_report)
    repository.save(earlier_report)

    reports = repository.get_by_run_id(requested_run_id)

    assert [report.report_id for report in reports] == [
        earlier_report.report_id,
        later_report.report_id,
    ]
    assert all(report.run_id == requested_run_id for report in reports)


def test_get_all_returns_reports_across_runs_in_timestamp_order(
    session: Session,
) -> None:
    repository = DiagnosticReportRepository(session)
    first_run_id = uuid4()
    second_run_id = uuid4()
    base_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
    newest = make_report(
        first_run_id,
        timestamp=base_time + timedelta(minutes=2),
    )
    oldest = make_report(second_run_id, timestamp=base_time)
    middle = make_report(
        first_run_id,
        timestamp=base_time + timedelta(minutes=1),
    )

    repository.save(newest)
    repository.save(oldest)
    repository.save(middle)

    reports = repository.get_all()

    assert [report.report_id for report in reports] == [
        oldest.report_id,
        middle.report_id,
        newest.report_id,
    ]
    assert {report.run_id for report in reports} == {
        first_run_id,
        second_run_id,
    }


def test_failed_save_rolls_back_report_and_findings(session: Session) -> None:
    repository = DiagnosticReportRepository(session)
    duplicate_finding_id = uuid4()
    first_report = make_report(
        uuid4(),
        findings=[make_finding(finding_id=duplicate_finding_id)],
    )
    repository.save(first_report)

    second_report = make_report(
        uuid4(),
        findings=[make_finding(finding_id=duplicate_finding_id)],
    )

    with pytest.raises(IntegrityError):
        repository.save(second_report)

    assert session.get(DiagnosticReportORM, second_report.report_id) is None
    persisted_finding_ids = set(
        session.scalars(select(DiagnosticFindingORM.finding_id)).all()
    )
    assert persisted_finding_ids == {duplicate_finding_id}
