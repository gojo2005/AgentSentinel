"""Response schemas for persisted diagnostic reports."""

from pydantic import BaseModel

from app.diagnostics.schemas import DiagnosticReport


class DiagnosticReportListResponse(BaseModel):
    """Paginated diagnostic reports."""

    items: list[DiagnosticReport]
    total: int
    limit: int
    offset: int
