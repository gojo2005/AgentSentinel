"""Deterministic, provider-independent evaluation for agent responses."""

import re
from uuid import UUID

from app.evaluation.schemas import EvaluationResult


_STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def _meaningful_words(text: str) -> set[str]:
    """Return unique lowercase words after removing punctuation and stop words."""
    return {
        word
        for word in re.findall(r"\b\w+\b", text.lower())
        if word not in _STOP_WORDS
    }


def _normalized_text(text: str) -> str:
    """Lowercase text, trim it, and remove basic punctuation for exact matching."""
    return re.sub(r"[^\w\s]", "", text.lower().strip())


class RuleBasedEvaluator:
    """Evaluate responses using simple deterministic text-matching rules."""

    def evaluate(
        self,
        question: str,
        answer: str,
        run_id: UUID,
        reference_answer: str | None = None,
        context: str | None = None,
    ) -> EvaluationResult:
        """Score response text and return an EvaluationResult.

        Scores are lexical heuristics only; they do not establish semantic
        correctness or reliably detect hallucinations.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string.")
        if reference_answer is not None and not reference_answer.strip():
            raise ValueError("reference_answer must not be empty when provided.")

        correctness_score: float | None = None
        if reference_answer is not None:
            correctness_score = float(
                _normalized_text(answer) == _normalized_text(reference_answer)
            )

        question_words = _meaningful_words(question)
        answer_words = _meaningful_words(answer)
        relevance_score = (
            len(question_words & answer_words) / len(question_words)
            if question_words
            else None
        )

        context_relevance_score: float | None = None
        hallucination_score: float | None = None
        if context is not None:
            context_words = _meaningful_words(context)
            if context_words:
                context_relevance_score = (
                    len(context_words & answer_words) / len(context_words)
                )
                hallucination_score = (
                    len(answer_words - context_words) / len(answer_words)
                    if answer_words
                    else 0.0
                )

        available_scores: list[float] = []
        if correctness_score is not None:
            available_scores.append(correctness_score)
        if relevance_score is not None:
            available_scores.append(relevance_score)
        if context_relevance_score is not None:
            available_scores.append(context_relevance_score)
        if hallucination_score is not None:
            available_scores.append(1.0 - hallucination_score)

        overall_score = (
            sum(available_scores) / len(available_scores) * 100
            if available_scores
            else None
        )
        passed = overall_score >= 70 if overall_score is not None else None

        available_metrics = []
        if correctness_score is not None:
            available_metrics.append("correctness (exact normalized match)")
        if relevance_score is not None:
            available_metrics.append("question keyword overlap")
        if context_relevance_score is not None:
            available_metrics.append("context keyword coverage")
        if hallucination_score is not None:
            available_metrics.append("unsupported-word heuristic")

        if available_metrics:
            feedback = "Metrics available: " + ", ".join(available_metrics) + "."
        else:
            feedback = "No metrics could be calculated from the supplied text."
        feedback += (
            " These lexical rules do not assess semantic correctness; "
            "the unsupported-word score is not genuine hallucination detection."
        )

        return EvaluationResult(
            run_id=run_id,
            correctness_score=correctness_score,
            relevance_score=relevance_score,
            context_relevance_score=context_relevance_score,
            hallucination_score=hallucination_score,
            overall_score=overall_score,
            passed=passed,
            feedback=feedback,
            evaluator="rule_based",
        )
