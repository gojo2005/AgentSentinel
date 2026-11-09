"""Unit tests for evaluation-result aggregation."""

from uuid import uuid4

import pytest

from app.evaluation.aggregator import EvaluationAggregator
from app.evaluation.schemas import EvaluationResult


def make_result(
    *,
    correctness_score: float | None = None,
    relevance_score: float | None = None,
    context_relevance_score: float | None = None,
    hallucination_score: float | None = None,
    overall_score: float | None = None,
    passed: bool | None = None,
    evaluator: str = "rule_based",
) -> EvaluationResult:
    """Create a valid evaluation result with a fresh run identifier."""
    return EvaluationResult(
        run_id=uuid4(),
        correctness_score=correctness_score,
        relevance_score=relevance_score,
        context_relevance_score=context_relevance_score,
        hallucination_score=hallucination_score,
        overall_score=overall_score,
        passed=passed,
        evaluator=evaluator,
    )


def test_aggregate_calculates_all_metric_averages() -> None:
    results = [
        make_result(
            correctness_score=1.0,
            relevance_score=0.6,
            context_relevance_score=0.8,
            hallucination_score=0.2,
            overall_score=80,
            passed=True,
        ),
        make_result(
            correctness_score=0.5,
            relevance_score=0.8,
            context_relevance_score=0.4,
            hallucination_score=0.6,
            overall_score=60,
            passed=False,
        ),
    ]

    summary = EvaluationAggregator().aggregate(results)

    assert summary.average_correctness_score == pytest.approx(0.75)
    assert summary.average_relevance_score == pytest.approx(0.7)
    assert summary.average_context_relevance_score == pytest.approx(0.6)
    assert summary.average_hallucination_score == pytest.approx(0.4)
    assert summary.average_overall_score == pytest.approx(70)


def test_aggregate_excludes_none_metric_values() -> None:
    results = [
        make_result(correctness_score=0.8),
        make_result(correctness_score=None),
        make_result(correctness_score=0.4),
    ]

    summary = EvaluationAggregator().aggregate(results)

    assert summary.average_correctness_score == pytest.approx(0.6)


def test_metric_with_all_none_values_returns_none() -> None:
    results = [make_result(), make_result()]

    summary = EvaluationAggregator().aggregate(results)

    assert summary.average_correctness_score is None
    assert summary.average_relevance_score is None
    assert summary.average_context_relevance_score is None
    assert summary.average_hallucination_score is None
    assert summary.average_overall_score is None


def test_pass_rate_excludes_unknown_pass_results() -> None:
    results = [
        make_result(passed=True),
        make_result(passed=False),
        make_result(passed=None),
    ]

    summary = EvaluationAggregator().aggregate(results)

    assert summary.pass_rate == pytest.approx(0.5)


def test_pass_rate_is_none_when_all_results_are_unknown() -> None:
    summary = EvaluationAggregator().aggregate(
        [make_result(passed=None), make_result(passed=None)]
    )

    assert summary.pass_rate is None


def test_evaluator_breakdown_counts_each_evaluator() -> None:
    results = [
        make_result(evaluator="rule_based"),
        make_result(evaluator="llm_judge"),
        make_result(evaluator="rule_based"),
        make_result(evaluator="llm_judge"),
        make_result(evaluator="rule_based"),
    ]

    summary = EvaluationAggregator().aggregate(results)

    assert summary.evaluator_breakdown == {"rule_based": 3, "llm_judge": 2}


def test_empty_input_returns_empty_summary() -> None:
    summary = EvaluationAggregator().aggregate([])

    assert summary.total_evaluations == 0
    assert summary.average_correctness_score is None
    assert summary.average_relevance_score is None
    assert summary.average_context_relevance_score is None
    assert summary.average_hallucination_score is None
    assert summary.average_overall_score is None
    assert summary.pass_rate is None
    assert summary.evaluator_breakdown == {}


def test_single_evaluation_produces_matching_summary_values() -> None:
    result = make_result(
        correctness_score=0.9,
        relevance_score=0.8,
        context_relevance_score=0.7,
        hallucination_score=0.1,
        overall_score=85,
        passed=True,
        evaluator="llm_judge",
    )

    summary = EvaluationAggregator().aggregate([result])

    assert summary.total_evaluations == 1
    assert summary.average_correctness_score == 0.9
    assert summary.average_relevance_score == 0.8
    assert summary.average_context_relevance_score == 0.7
    assert summary.average_hallucination_score == 0.1
    assert summary.average_overall_score == 85
    assert summary.pass_rate == 1.0
    assert summary.evaluator_breakdown == {"llm_judge": 1}


def test_total_evaluation_count_matches_input_length() -> None:
    results = [make_result() for _ in range(4)]

    summary = EvaluationAggregator().aggregate(results)

    assert summary.total_evaluations == len(results)
