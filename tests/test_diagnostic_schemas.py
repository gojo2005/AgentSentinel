"""Tests for provider-independent diagnostic Pydantic schemas."""

from datetime import timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.diagnostics.schemas import DiagnosticFinding, DiagnosticReport


def make_finding(**overrides: object) -> DiagnosticFinding:
    """Create a valid finding, allowing individual fields to be overridden."""
    values: dict[str, object] = {
        "category": "reliability",
        "severity": "medium",
        "title": "Slow tool response",
        "description": "A tool call took longer than expected.",
        "recommendation": "Review tool latency.",
    }
    values.update(overrides)
    return DiagnosticFinding(**values)


def make_report(**overrides: object) -> DiagnosticReport:
    """Create a valid report, allowing individual fields to be overridden."""
    values: dict[str, object] = {
        "run_id": uuid4(),
        "overall_status": "warning",
        "summary": "The run completed with one finding.",
    }
    values.update(overrides)
    return DiagnosticReport(**values)


def test_diagnostic_finding_can_be_created_with_valid_values() -> None:
    finding = make_finding(
        evidence=["Tool call took 1.5 seconds."],
        metadata={"tool": "calculator"},
    )

    assert finding.category == "reliability"
    assert finding.severity == "medium"
    assert finding.title == "Slow tool response"
    assert finding.evidence == ["Tool call took 1.5 seconds."]
    assert finding.recommendation == "Review tool latency."
    assert finding.metadata == {"tool": "calculator"}


def test_finding_id_is_generated_automatically() -> None:
    first = make_finding()
    second = make_finding()

    assert first.finding_id is not None
    assert first.finding_id != second.finding_id


def test_invalid_finding_severity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_finding(severity="urgent")


def test_finding_evidence_defaults_to_empty_list() -> None:
    finding = make_finding()

    assert finding.evidence == []


def test_finding_metadata_defaults_to_empty_dictionary() -> None:
    finding = make_finding()

    assert finding.metadata == {}


def test_diagnostic_report_can_include_findings() -> None:
    finding = make_finding()
    report = make_report(
        overall_status="critical",
        findings=[finding],
    )

    assert report.overall_status == "critical"
    assert report.findings == [finding]
    assert report.run_id is not None
    assert report.summary == "The run completed with one finding."


def test_report_id_is_generated_automatically() -> None:
    first = make_report()
    second = make_report()

    assert first.report_id is not None
    assert first.report_id != second.report_id


def test_report_timestamp_is_timezone_aware() -> None:
    report = make_report()

    assert report.timestamp.tzinfo is not None
    assert report.timestamp.utcoffset() is not None
    assert report.timestamp.utcoffset().total_seconds() == 0
    assert report.timestamp.tzinfo == timezone.utc


def test_invalid_report_overall_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_report(overall_status="failed")


def test_report_findings_default_to_empty_list() -> None:
    report = make_report()

    assert report.findings == []


def test_findings_have_independent_mutable_defaults() -> None:
    first = make_finding()
    second = make_finding()

    first.evidence.append("Observed timeout.")
    first.metadata["source"] = "test"

    assert second.evidence == []
    assert second.metadata == {}


def test_reports_have_independent_mutable_defaults() -> None:
    first = make_report()
    second = make_report()

    first.findings.append(make_finding())
    first.metadata["source"] = "test"

    assert second.findings == []
    assert second.metadata == {}
