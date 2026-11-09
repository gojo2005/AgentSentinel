"""Provider-independent aggregation of AgentSentinel evaluation results."""

from collections import Counter

from app.evaluation.schemas import EvaluationResult, EvaluationSummary


class EvaluationAggregator:
    """Calculate aggregate metrics for a collection of evaluation results."""

    def aggregate(
        self,
        results: list[EvaluationResult],
    ) -> EvaluationSummary:
        """Return averages, pass rate, and evaluator counts for the results."""
        correctness_scores = [
            result.correctness_score
            for result in results
            if result.correctness_score is not None
        ]
        relevance_scores = [
            result.relevance_score
            for result in results
            if result.relevance_score is not None
        ]
        context_relevance_scores = [
            result.context_relevance_score
            for result in results
            if result.context_relevance_score is not None
        ]
        hallucination_scores = [
            result.hallucination_score
            for result in results
            if result.hallucination_score is not None
        ]
        overall_scores = [
            result.overall_score
            for result in results
            if result.overall_score is not None
        ]
        pass_results = [
            result.passed
            for result in results
            if result.passed is not None
        ]

        return EvaluationSummary(
            total_evaluations=len(results),
            average_correctness_score=self._average(correctness_scores),
            average_relevance_score=self._average(relevance_scores),
            average_context_relevance_score=self._average(
                context_relevance_scores
            ),
            average_hallucination_score=self._average(hallucination_scores),
            average_overall_score=self._average(overall_scores),
            pass_rate=(
                sum(passed is True for passed in pass_results) / len(pass_results)
                if pass_results
                else None
            ),
            evaluator_breakdown=dict(Counter(result.evaluator for result in results)),
        )

    @staticmethod
    def _average(values: list[float]) -> float | None:
        """Return the arithmetic mean, or None when no metric values exist."""
        return sum(values) / len(values) if values else None
