"""create initial storage tables

Revision ID: fa9f61ca9077
Revises:
Create Date: 2026-11-05 11:56:04.288364

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# Revision identifiers, used by Alembic.
revision: str = "fa9f61ca9077"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the initial AgentSentinel storage tables."""

    op.create_table(
        "diagnostic_reports",
        sa.Column("report_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("overall_status", sa.String(length=16), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("report_id"),
    )

    op.create_index(
        op.f("ix_diagnostic_reports_run_id"),
        "diagnostic_reports",
        ["run_id"],
        unique=False,
    )

    op.create_table(
        "evaluation_results",
        sa.Column("evaluation_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("correctness_score", sa.Float(), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column("context_relevance_score", sa.Float(), nullable=True),
        sa.Column("hallucination_score", sa.Float(), nullable=True),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("evaluator", sa.String(length=100), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("evaluation_id"),
    )

    op.create_index(
        "ix_evaluation_results_run_timestamp",
        "evaluation_results",
        ["run_id", "timestamp"],
        unique=False,
    )

    op.create_table(
        "execution_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("agent_name", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=True),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("tool_name", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("input", sa.JSON(), nullable=True),
        sa.Column("output", sa.JSON(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
    )

    op.create_index(
        "ix_execution_events_run_timestamp",
        "execution_events",
        ["run_id", "timestamp"],
        unique=False,
    )

    op.create_table(
        "diagnostic_findings",
        sa.Column("finding_id", sa.Uuid(), nullable=False),
        sa.Column("report_id", sa.Uuid(), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("recommendation", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["report_id"],
            ["diagnostic_reports.report_id"],
        ),
        sa.PrimaryKeyConstraint("finding_id"),
    )

    op.create_index(
        op.f("ix_diagnostic_findings_category"),
        "diagnostic_findings",
        ["category"],
        unique=False,
    )

    op.create_index(
        op.f("ix_diagnostic_findings_report_id"),
        "diagnostic_findings",
        ["report_id"],
        unique=False,
    )

    op.create_index(
        op.f("ix_diagnostic_findings_severity"),
        "diagnostic_findings",
        ["severity"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the initial AgentSentinel storage tables."""

    op.drop_index(
        op.f("ix_diagnostic_findings_severity"),
        table_name="diagnostic_findings",
    )
    op.drop_index(
        op.f("ix_diagnostic_findings_report_id"),
        table_name="diagnostic_findings",
    )
    op.drop_index(
        op.f("ix_diagnostic_findings_category"),
        table_name="diagnostic_findings",
    )
    op.drop_table("diagnostic_findings")

    op.drop_index(
        "ix_execution_events_run_timestamp",
        table_name="execution_events",
    )
    op.drop_table("execution_events")

    op.drop_index(
        "ix_evaluation_results_run_timestamp",
        table_name="evaluation_results",
    )
    op.drop_table("evaluation_results")

    op.drop_index(
        op.f("ix_diagnostic_reports_run_id"),
        table_name="diagnostic_reports",
    )
    op.drop_table("diagnostic_reports")