"""Unit tests for the deterministic rule-based evaluator."""

from uuid import uuid4

import pytest

from app.evaluation.evaluator import RuleBasedEvaluator


@pytest.fixture
def evaluator() -> RuleBasedEvaluator:
    """Provide a fresh evaluator for each test."""
    return RuleBasedEvaluator()


def test_exact_reference_answer_match_scores_one(
    evaluator: RuleBasedEvaluator,
) -> None:
    result = evaluator.evaluate(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        run_id=uuid4(),
        reference_answer=" paris is the capital of france! ",
    )

    assert result.correctness_score == 1.0


def test_different_reference_answer_scores_zero(
    evaluator: RuleBasedEvaluator,
) -> None:
    result = evaluator.evaluate(
        question="What is the capital of France?",
        answer="Paris",
        run_id=uuid4(),
        reference_answer="Lyon",
    )

    assert result.correctness_score == 0.0


def test_missing_reference_answer_has_no_correctness_score(
    evaluator: RuleBasedEvaluator,
) -> None:
    result = evaluator.evaluate(
        question="Name a planet.",
        answer="Mars",
        run_id=uuid4(),
    )

    assert result.correctness_score is None


def test_context_scores_are_within_zero_and_one(
    evaluator: RuleBasedEvaluator,
) -> None:
    result = evaluator.evaluate(
        question="What color is the sky?",
        answer="The sky is usually blue.",
        run_id=uuid4(),
        context="The sky often appears blue during the day.",
    )

    assert result.context_relevance_score is not None
    assert 0.0 <= result.context_relevance_score <= 1.0
    assert result.hallucination_score is not None
    assert 0.0 <= result.hallucination_score <= 1.0


def test_missing_context_has_no_context_scores(
    evaluator: RuleBasedEvaluator,
) -> None:
    result = evaluator.evaluate(
        question="What color is the sky?",
        answer="The sky is blue.",
        run_id=uuid4(),
    )

    assert result.context_relevance_score is None
    assert result.hallucination_score is None


def test_empty_question_raises_value_error(
    evaluator: RuleBasedEvaluator,
) -> None:
    with pytest.raises(ValueError, match="question"):
        evaluator.evaluate("", "An answer.", uuid4())


def test_empty_answer_raises_value_error(
    evaluator: RuleBasedEvaluator,
) -> None:
    with pytest.raises(ValueError, match="answer"):
        evaluator.evaluate("A question?", "  \n", uuid4())


def test_whitespace_only_reference_answer_raises_value_error(
    evaluator: RuleBasedEvaluator,
) -> None:
    with pytest.raises(
        ValueError,
        match="reference_answer must not be empty when provided",
    ):
        evaluator.evaluate(
            "A question?",
            "An answer.",
            uuid4(),
            reference_answer=" \t ",
        )


def test_overall_score_stays_within_zero_and_one_hundred(
    evaluator: RuleBasedEvaluator,
) -> None:
    result = evaluator.evaluate(
        question="What color is the sky?",
        answer="Blue",
        run_id=uuid4(),
        reference_answer="Blue",
        context="The sky is blue.",
    )

    assert result.overall_score is not None
    assert 0.0 <= result.overall_score <= 100.0


def test_evaluator_name_is_rule_based(
    evaluator: RuleBasedEvaluator,
) -> None:
    result = evaluator.evaluate(
        question="Name a planet.",
        answer="Mars",
        run_id=uuid4(),
    )

    assert result.evaluator == "rule_based"
