"""Unit tests for LLMJudge using deterministic injected model responses."""

from typing import Any
from uuid import UUID, uuid4

import pytest

from app.evaluation.llm_judge import LLMJudge
from app.evaluation.schemas import EvaluationResult


class FakeStructuredModel:
    """Return a fixed response through LangChain's runnable interface."""

    def __init__(self, response: Any) -> None:
        self.response = response

    def invoke(self, messages: list[Any]) -> Any:
        return self.response


class FakeChatModel:
    """Provide a fake implementation of the structured-output model interface."""

    def __init__(self, response: Any) -> None:
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


@pytest.fixture
def valid_scores() -> dict[str, Any]:
    """Provide all required deterministic judge fields."""
    return {
        "correctness_score": 0.9,
        "relevance_score": 0.8,
        "context_relevance_score": 0.7,
        "hallucination_score": 0.2,
        "feedback": "The answer is accurate and relevant.",
    }


def evaluate_response(
    response: Any,
    *,
    question: str = "What is the capital of France?",
    answer: str = "Paris is the capital of France.",
    run_id: UUID | str | None = None,
    reference_answer: str | None = None,
    context: str | None = None,
) -> EvaluationResult:
    """Evaluate using a fake model, never a live provider."""
    return LLMJudge(FakeChatModel(response)).evaluate(
        question=question,
        answer=answer,
        run_id=uuid4() if run_id is None else run_id,
        reference_answer=reference_answer,
        context=context,
    )


def test_valid_evaluation_with_context_returns_all_scores(
    valid_scores: dict[str, Any],
) -> None:
    result = evaluate_response(valid_scores, context="Paris is in France.")

    assert result.correctness_score == 0.9
    assert result.relevance_score == 0.8
    assert result.context_relevance_score == 0.7
    assert result.hallucination_score == 0.2


def test_evaluation_without_context_accepts_none_context_score(
    valid_scores: dict[str, Any],
) -> None:
    response = {**valid_scores, "context_relevance_score": None}

    result = evaluate_response(response)

    assert result.context_relevance_score is None


def test_overall_score_excludes_unavailable_context_relevance(
    valid_scores: dict[str, Any],
) -> None:
    response = {**valid_scores, "context_relevance_score": None}

    result = evaluate_response(response)

    expected_score = (0.9 + 0.8 + (1 - 0.2)) / 3 * 100
    assert result.overall_score == pytest.approx(expected_score)


def test_overall_score_inverts_hallucination_score(
    valid_scores: dict[str, Any],
) -> None:
    result = evaluate_response(valid_scores, context="Paris is in France.")

    expected_score = (0.9 + 0.8 + 0.7 + (1 - 0.2)) / 4 * 100
    assert result.overall_score == pytest.approx(expected_score)


def test_malformed_json_raises_value_error() -> None:
    with pytest.raises(ValueError, match="malformed JSON"):
        evaluate_response('{"correctness_score": 0.9')


def test_missing_required_fields_raise_value_error(
    valid_scores: dict[str, Any],
) -> None:
    response = dict(valid_scores)
    del response["relevance_score"]

    with pytest.raises(ValueError, match="missing required fields"):
        evaluate_response(response)


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_out_of_range_scores_raise_value_error(
    valid_scores: dict[str, Any],
    score: float,
) -> None:
    response = {**valid_scores, "correctness_score": score}

    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_response(response)


def test_boolean_score_raises_value_error(
    valid_scores: dict[str, Any],
) -> None:
    response = {**valid_scores, "relevance_score": True}

    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_response(response)


def test_non_string_feedback_raises_value_error(
    valid_scores: dict[str, Any],
) -> None:
    response = {**valid_scores, "feedback": ["not", "text"]}

    with pytest.raises(ValueError, match="feedback field must be a string"):
        evaluate_response(response)


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("", "A valid answer."),
        ("A valid question?", " \t\n"),
    ],
)
def test_empty_question_or_answer_raises_value_error(
    valid_scores: dict[str, Any],
    question: str,
    answer: str,
) -> None:
    with pytest.raises(ValueError):
        evaluate_response(valid_scores, question=question, answer=answer)


def test_invalid_run_id_raises_value_error(
    valid_scores: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        evaluate_response(valid_scores, run_id="not-a-valid-uuid")


def test_successful_result_uses_llm_judge_evaluator_name(
    valid_scores: dict[str, Any],
) -> None:
    result = evaluate_response(valid_scores)

    assert result.evaluator == "llm_judge"
