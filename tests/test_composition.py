"""Tests for AgentSentinel application composition helpers."""

from app.core.composition import create_production_event_logger
from app.monitoring.event_logger import EventLogger
from app.storage.event_sink import SQLAlchemyEventSink


def test_factory_returns_event_logger_with_sqlalchemy_sink() -> None:
    logger = create_production_event_logger()

    assert isinstance(logger, EventLogger)
    assert isinstance(logger._sink, SQLAlchemyEventSink)


def test_factory_creates_fresh_logger_and_sink_each_time() -> None:
    first_logger = create_production_event_logger()
    second_logger = create_production_event_logger()

    assert first_logger is not second_logger
    assert first_logger._sink is not second_logger._sink
