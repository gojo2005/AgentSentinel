"""Integration test for AgentSentinel's evaluation pipeline."""

from typing import Any
from uuid import uuid4

import pytest

from app.evaluation.aggregator import EvaluationAggregator
from app.evaluation.evaluator import RuleBasedEvaluator
from app.evaluation.llm_judge import LLMJudge
from app.evaluation.schemas import EvaluationResult


class FakeStructuredModel:
    """Return a predetermined judge response without making an API call."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        return self.response


class FakeChatModel:
    """Implement the structured-output interface used by LLMJudge."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def with_structured_output(
        self,
        schema: dict[str, Any],
        *,
        method: str,
    ) -> FakeStructuredModel:
        assert method == "json_mode"
        assert "correctness_score" in schema["required"]
        return FakeStructuredModel(self.response)


def test_evaluation_pipeline_aggregates_rule_and_llm_judge_results() -> None:
    question = "What is the capital of France?"
    answer = "Paris is the capital of France."
    reference_answer = "Paris is the capital of France."
    context = "Paris is the capital of France."
    run_id = uuid4()
    fake_response = {
        "correctness_score": 0.8,
        "relevance_score": 0.6,
        "context_relevance_score": 0.4,
        "hallucination_score": 0.2,
        "feedback": "The answer is supported and relevant.",
    }

    rule_result = RuleBasedEvaluator().evaluate(
        question=question,
        answer=answer,
        run_id=run_id,
        reference_answer=reference_answer,
        context=context,
    )
    judge_result = LLMJudge(FakeChatModel(fake_response)).evaluate(
        question=question,
        answer=answer,
        run_id=run_id,
        reference_answer=reference_answer,
        context=context,
    )
    summary = EvaluationAggregator().aggregate([rule_result, judge_result])

    assert isinstance(rule_result, EvaluationResult)
    assert isinstance(judge_result, EvaluationResult)
    assert rule_result.run_id == run_id
    assert judge_result.run_id == run_id
    assert summary.total_evaluations == 2
    assert summary.evaluator_breakdown == {
        "rule_based": 1,
        "llm_judge": 1,
    }

    assert rule_result.overall_score is not None
    assert judge_result.overall_score is not None
    expected_overall_average = (
        rule_result.overall_score + judge_result.overall_score
    ) / 2
    assert summary.average_overall_score == pytest.approx(
        expected_overall_average
    )

    assert summary.average_correctness_score == pytest.approx((1.0 + 0.8) / 2)
    assert summary.average_relevance_score == pytest.approx((1.0 + 0.6) / 2)
    assert summary.average_context_relevance_score == pytest.approx(
        (1.0 + 0.4) / 2
    )
    assert summary.average_hallucination_score == pytest.approx((0.0 + 0.2) / 2)
    assert summary.pass_rate == pytest.approx(
        sum(result.passed is True for result in [rule_result, judge_result]) / 2
    )
