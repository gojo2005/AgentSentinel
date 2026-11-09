"""Deterministic, provider-independent diagnostics for agent executions."""

from collections.abc import Iterable
import json
from typing import Literal
from uuid import UUID

from app.core.sanitization import redact_sensitive_data
from app.detection.failure_detector import FailureDetector
from app.diagnostics.schemas import DiagnosticFinding, DiagnosticReport
from app.evaluation.schemas import EvaluationResult
from app.monitoring.schemas import ExecutionEvent


class RuleBasedDiagnosticEngine:
    """Create actionable diagnostic findings from execution and evaluation data."""

    def __init__(self) -> None:
        """Create an engine with the project's standard failure detector."""
        self._failure_detector = FailureDetector()

    def diagnose(
        self,
        run_id: UUID,
        events: list[ExecutionEvent],
        evaluation: EvaluationResult | None = None,
    ) -> DiagnosticReport:
        """Analyze one run and return a report with deterministic findings."""
        if any(event.run_id != run_id for event in events):
            raise ValueError("All events must match the supplied run_id.")
        if evaluation is not None and evaluation.run_id != run_id:
            raise ValueError("The evaluation run_id must match the supplied run_id.")

        run_analysis = self._failure_detector.analyze_run(events)
        findings: list[DiagnosticFinding] = []

        if not events and evaluation is None:
            findings.append(
                DiagnosticFinding(
                    category="insufficient_data",
                    severity="medium",
                    title="Insufficient execution data",
                    description=(
                        "No execution events or evaluation results were "
                        "available to diagnose this run."
                    ),
                    evidence=[
                        "Neither execution events nor an evaluation result "
                        "was supplied."
                    ],
                    recommendation=(
                        "Verify that execution monitoring and evaluation "
                        "results are being captured."
                    ),
                )
            )

        failed_events = [event for event in events if event.status == "failure"]
        if run_analysis["overall_status"] == "failure":
            evidence = [
                self._describe_event(event)
                for event in failed_events
            ]
            if not evidence:
                evidence.append("FailureDetector classified the run as failed.")
            findings.append(
                DiagnosticFinding(
                    category="run_failure",
                    severity="critical",
                    title="Agent run failed",
                    description=(
                        "The execution contains a failed terminal event and "
                        "should be investigated before relying on its result."
                    ),
                    evidence=self._unique(evidence),
                    recommendation=(
                        "Inspect the failed component and its error details, "
                        "then rerun the agent after resolving the cause."
                    ),
                    metadata={
                        "failed_events": run_analysis["failed_events"],
                        "errors": run_analysis["errors"],
                    },
                )
            )

        tool_failures = [
            event
            for event in events
            if event.event_type == "tool_call" and event.status == "failure"
        ]
        if tool_failures:
            tool_names = sorted(
                {
                    event.tool_name
                    for event in tool_failures
                    if event.tool_name is not None
                }
            )
            findings.append(
                DiagnosticFinding(
                    category="tool_failure",
                    severity="high",
                    title="Tool execution failed",
                    description=(
                        f"{len(tool_failures)} tool call(s) failed during this run."
                    ),
                    evidence=self._unique(
                        self._describe_event(event) for event in tool_failures
                    ),
                    recommendation=(
                        "Inspect the affected tool's inputs and implementation, "
                        "and verify any services or data it depends on."
                    ),
                    metadata={
                        "failed_tool_count": len(tool_failures),
                        "tool_names": tool_names,
                    },
                )
            )

        llm_failures = [
            event
            for event in events
            if event.event_type == "llm_call" and event.status == "failure"
        ]
        if llm_failures:
            findings.append(
                DiagnosticFinding(
                    category="llm_failure",
                    severity="high",
                    title="LLM call failed",
                    description=(
                        f"{len(llm_failures)} LLM call(s) failed during this run."
                    ),
                    evidence=self._unique(
                        self._describe_event(event) for event in llm_failures
                    ),
                    recommendation=(
                        "Check provider configuration, credentials, service "
                        "availability, and any reported error details."
                    ),
                    metadata={"failed_llm_call_count": len(llm_failures)},
                )
            )

        if evaluation is not None:
            if (
                evaluation.hallucination_score is not None
                and evaluation.hallucination_score >= 0.70
            ):
                findings.append(
                    DiagnosticFinding(
                        category="high_hallucination",
                        severity="high",
                        title="High hallucination risk",
                        description=(
                            "The evaluation indicates a high degree of "
                            "unsupported content."
                        ),
                        evidence=[
                            "hallucination_score="
                            f"{evaluation.hallucination_score:.2f}"
                        ],
                        recommendation=(
                            "Review the response against its source context and "
                            "improve grounding or abstention behavior."
                        ),
                        metadata={
                            "hallucination_score": evaluation.hallucination_score
                        },
                    )
                )

            if (
                evaluation.relevance_score is not None
                and evaluation.relevance_score < 0.40
            ):
                findings.append(
                    DiagnosticFinding(
                        category="low_relevance",
                        severity="high",
                        title="Low response relevance",
                        description=(
                            "The evaluation indicates that the response did not "
                            "adequately address the question."
                        ),
                        evidence=(
                            f"relevance_score={evaluation.relevance_score:.2f}",
                        ),
                        recommendation=(
                            "Review the question-to-answer path and ensure the "
                            "agent responds directly to the user's request."
                        ),
                        metadata={"relevance_score": evaluation.relevance_score},
                    )
                )

            if (
                evaluation.overall_score is not None
                and evaluation.overall_score < 70
            ):
                severity: Literal["high", "medium"] = (
                    "high" if evaluation.overall_score < 50 else "medium"
                )
                findings.append(
                    DiagnosticFinding(
                        category="low_overall_score",
                        severity=severity,
                        title="Low overall evaluation score",
                        description=(
                            "The evaluation's overall quality score is below "
                            "the passing threshold."
                        ),
                        evidence=(
                            f"overall_score={evaluation.overall_score:.2f}/100",
                        ),
                        recommendation=(
                            "Review the evaluation feedback and improve the "
                            "response quality before relying on this result."
                        ),
                        metadata={"overall_score": evaluation.overall_score},
                    )
                )

        run_duration_ms = run_analysis["total_duration_ms"]
        if run_duration_ms is not None and run_duration_ms > 10_000:
            findings.append(
                DiagnosticFinding(
                    category="high_execution_duration",
                    severity="medium",
                    title="High execution duration",
                    description=(
                        "The run exceeded the 10-second execution-duration "
                        "threshold."
                    ),
                    evidence=[f"selected run duration_ms={run_duration_ms:g}"],
                    recommendation=(
                        "Inspect agent, model, and tool latency to identify "
                        "opportunities to reduce total execution time."
                    ),
                    metadata={
                        "terminal_durations_ms": [run_duration_ms]
                    },
                )
            )

        if any(finding.severity == "critical" for finding in findings):
            overall_status = "critical"
        elif findings:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        if findings:
            summary = (
                f"{len(findings)} diagnostic finding(s) detected: "
                + "; ".join(finding.title for finding in findings)
                + "."
            )
        else:
            summary = "No diagnostic issues were detected for this run."

        return DiagnosticReport(
            run_id=run_id,
            overall_status=overall_status,
            findings=findings,
            summary=summary,
            metadata={
                "total_events": len(events),
                "total_duration_ms": run_analysis["total_duration_ms"],
            },
        )

    @staticmethod
    def _describe_event(event: ExecutionEvent) -> str:
        """Format relevant event details as concise diagnostic evidence."""
        details = [f"event_type={event.event_type}"]
        if event.tool_name is not None:
            details.append(f"tool_name={event.tool_name}")
        if event.provider is not None:
            details.append(f"provider={event.provider}")
        if event.model is not None:
            details.append(f"model={event.model}")
        if event.error is not None:
            safe_error = redact_sensitive_data(event.error)
            if not isinstance(safe_error, str):
                safe_error = json.dumps(safe_error, ensure_ascii=False, default=str)
            details.append(f"error={safe_error}")
        return ", ".join(details)

    @staticmethod
    def _unique(items: Iterable[str]) -> list[str]:
        """Return unique evidence strings while preserving their original order."""
        return list(dict.fromkeys(items))
