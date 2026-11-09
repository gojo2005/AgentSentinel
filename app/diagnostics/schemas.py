"""Provider-independent Pydantic schemas for diagnostic findings and reports."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DiagnosticFinding(BaseModel):
    """A detected issue or improvement opportunity with supporting evidence."""

    finding_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this diagnostic finding.",
    )
    category: str = Field(
        description="Type or grouping of the finding, such as reliability or quality.",
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Impact level assigned to the finding.",
    )
    title: str = Field(
        description="Short, human-readable title for the finding.",
    )
    description: str = Field(
        description="Details explaining what was found and why it matters.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this finding.",
    )
    recommendation: str = Field(
        description="Suggested action to address or investigate the finding.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured information about the finding.",
    )


class DiagnosticReport(BaseModel):
    """A run-level summary of health status and diagnostic findings."""

    report_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this diagnostic report.",
    )
    run_id: UUID = Field(
        description="Identifier of the agent execution covered by this report.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time when this diagnostic report was created.",
    )
    overall_status: Literal["healthy", "warning", "critical"] = Field(
        description="Overall health classification for the execution.",
    )
    findings: list[DiagnosticFinding] = Field(
        default_factory=list,
        description="Diagnostic findings associated with the execution.",
    )
    summary: str = Field(
        description="Concise human-readable overview of the report.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured information about the report.",
    )
