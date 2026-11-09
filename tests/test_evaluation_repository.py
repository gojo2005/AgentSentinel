"""Tests for SQLAlchemy-backed evaluation result persistence."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.evaluation.schemas import EvaluationResult
from app.storage.evaluation_repository import EvaluationResultRepository
from app.storage.models import Base, EvaluationResultORM


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Create and clean up an isolated SQLite database for a test."""
    database_path = tmp_path / "evaluations.db"
    test_engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """Provide a session bound to the temporary test database."""
    test_session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )
    with test_session_factory() as test_session:
        yield test_session


def make_result(
    run_id: UUID,
    *,
    timestamp: datetime | None = None,
    correctness_score: float | None = 0.9,
    relevance_score: float | None = 0.8,
    context_relevance_score: float | None = 0.75,
    hallucination_score: float | None = 0.1,
    overall_score: float | None = 87.5,
    passed: bool | None = True,
    feedback: str | None = "The answer is well supported.",
    evaluator: str = "rule_based",
    metadata: dict[str, object] | None = None,
) -> EvaluationResult:
    """Build a valid evaluation result with a unique ID and aware timestamp."""
    return EvaluationResult(
        run_id=run_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        correctness_score=correctness_score,
        relevance_score=relevance_score,
        context_relevance_score=context_relevance_score,
        hallucination_score=hallucination_score,
        overall_score=overall_score,
        passed=passed,
        feedback=feedback,
        evaluator=evaluator,
        metadata={"source": "test"} if metadata is None else metadata,
    )


def test_save_persists_evaluation_and_preserves_all_fields(session: Session) -> None:
    run_id = uuid4()
    timestamp = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    result = make_result(
        run_id,
        timestamp=timestamp,
        correctness_score=0.91,
        relevance_score=0.82,
        context_relevance_score=0.73,
        hallucination_score=0.14,
        overall_score=86.5,
        passed=True,
        feedback="Evidence supports the response.",
        evaluator="llm_judge",
        metadata={"model": "test-judge", "round": 2},
    )
    repository = EvaluationResultRepository(session)

    saved = repository.save(result)

    assert saved.evaluation_id == result.evaluation_id
    assert saved.run_id == result.run_id
    assert saved.timestamp == result.timestamp
    assert saved.correctness_score == result.correctness_score
    assert saved.relevance_score == result.relevance_score
    assert saved.context_relevance_score == result.context_relevance_score
    assert saved.hallucination_score == result.hallucination_score
    assert saved.overall_score == result.overall_score
    assert saved.passed == result.passed
    assert saved.feedback == result.feedback
    assert saved.evaluator == result.evaluator
    assert saved.metadata == result.metadata
    assert isinstance(saved, EvaluationResult)

    persisted = session.get(EvaluationResultORM, result.evaluation_id)
    assert persisted is not None
    assert persisted.run_id == result.run_id
    assert persisted.timestamp == result.timestamp
    assert persisted.correctness_score == result.correctness_score
    assert persisted.relevance_score == result.relevance_score
    assert persisted.context_relevance_score == result.context_relevance_score
    assert persisted.hallucination_score == result.hallucination_score
    assert persisted.overall_score == result.overall_score
    assert persisted.passed == result.passed
    assert persisted.feedback == result.feedback
    assert persisted.evaluator == result.evaluator
    assert persisted.metadata_json == result.metadata


def test_get_by_id_returns_existing_result_and_none_for_missing_id(
    session: Session,
) -> None:
    result = make_result(uuid4())
    repository = EvaluationResultRepository(session)
    repository.save(result)

    retrieved = repository.get_by_id(result.evaluation_id)

    assert isinstance(retrieved, EvaluationResult)
    assert retrieved == result
    assert repository.get_by_id(uuid4()) is None


def test_get_by_run_id_filters_results_and_orders_by_timestamp(
    session: Session,
) -> None:
    repository = EvaluationResultRepository(session)
    requested_run_id = uuid4()
    other_run_id = uuid4()
    base_time = datetime(2026, 2, 1, tzinfo=timezone.utc)
    later_result = make_result(
        requested_run_id,
        timestamp=base_time + timedelta(minutes=2),
    )
    other_run_result = make_result(
        other_run_id,
        timestamp=base_time + timedelta(minutes=1),
    )
    earlier_result = make_result(requested_run_id, timestamp=base_time)

    repository.save(later_result)
    repository.save(other_run_result)
    repository.save(earlier_result)

    results = repository.get_by_run_id(requested_run_id)

    assert [item.evaluation_id for item in results] == [
        earlier_result.evaluation_id,
        later_result.evaluation_id,
    ]
    assert all(item.run_id == requested_run_id for item in results)
    assert all(isinstance(item, EvaluationResult) for item in results)


def test_get_all_returns_results_across_runs_in_timestamp_order(
    session: Session,
) -> None:
    repository = EvaluationResultRepository(session)
    first_run_id = uuid4()
    second_run_id = uuid4()
    base_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
    newest = make_result(
        first_run_id,
        timestamp=base_time + timedelta(minutes=2),
    )
    oldest = make_result(second_run_id, timestamp=base_time)
    middle = make_result(
        first_run_id,
        timestamp=base_time + timedelta(minutes=1),
    )

    repository.save(newest)
    repository.save(oldest)
    repository.save(middle)

    results = repository.get_all()

    assert [item.evaluation_id for item in results] == [
        oldest.evaluation_id,
        middle.evaluation_id,
        newest.evaluation_id,
    ]
    assert {item.run_id for item in results} == {first_run_id, second_run_id}
    assert all(isinstance(item, EvaluationResult) for item in results)


def test_save_and_retrieve_preserve_optional_none_scores(session: Session) -> None:
    result = make_result(
        uuid4(),
        correctness_score=None,
        relevance_score=None,
        context_relevance_score=None,
        hallucination_score=None,
        overall_score=None,
        passed=None,
        feedback=None,
    )
    repository = EvaluationResultRepository(session)

    saved = repository.save(result)
    retrieved = repository.get_by_id(result.evaluation_id)

    assert retrieved is not None
    for item in (saved, retrieved):
        assert item.correctness_score is None
        assert item.relevance_score is None
        assert item.context_relevance_score is None
        assert item.hallucination_score is None
        assert item.overall_score is None
        assert item.passed is None
        assert item.feedback is None
