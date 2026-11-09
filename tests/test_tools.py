"""Tests for the AgentSentinel sample-agent tools."""

from pathlib import Path

import pytest

from app.agent import tools


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3", "2 + 3 = 5"),
        ("9 - 4", "9 - 4 = 5"),
        ("6 * 7", "6 * 7 = 42"),
        ("8 / 2", "8 / 2 = 4.0"),
        ("2 ** 3", "2 ** 3 = 8"),
        ("(2 + 3) * 4", "(2 + 3) * 4 = 20"),
    ],
)
def test_calculator_returns_arithmetic_result(
    expression: str,
    expected: str,
) -> None:
    assert tools.calculator.invoke({"expression": expression}) == expected


def test_calculator_handles_division_by_zero() -> None:
    result = tools.calculator.invoke({"expression": "10 / 0"})

    assert result == "Error: Division by zero is not allowed."


def test_calculator_handles_invalid_syntax() -> None:
    result = tools.calculator.invoke({"expression": "2 +"})

    assert result == "Error: The expression has invalid syntax."


@pytest.mark.parametrize("expression", ["import os", "abs(-1)"])
def test_calculator_rejects_unsupported_python_expressions(
    expression: str,
) -> None:
    result = tools.calculator.invoke({"expression": expression})

    assert result.startswith("Error:")


@pytest.fixture
def temporary_knowledge_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    monkeypatch.setattr(tools, "_KNOWLEDGE_BASE_DIRECTORY", knowledge_base)
    return knowledge_base


def test_knowledge_base_search_finds_markdown_and_text_matches(
    temporary_knowledge_base: Path,
) -> None:
    (temporary_knowledge_base / "agent.md").write_text(
        "# Agent notes\nAgents can call tools.",
        encoding="utf-8",
    )
    (temporary_knowledge_base / "operations.txt").write_text(
        "Monitoring records each tool call.",
        encoding="utf-8",
    )

    result = tools.knowledge_base_search.invoke({"query": "tool tools"})

    assert "- agent.md" in result
    assert "Agents can call tools." in result
    assert "- operations.txt" in result
    assert "Monitoring records each tool call." in result


def test_knowledge_base_search_is_case_insensitive(
    temporary_knowledge_base: Path,
) -> None:
    (temporary_knowledge_base / "notes.md").write_text(
        "The AGENT tracks Failures.",
        encoding="utf-8",
    )

    result = tools.knowledge_base_search.invoke({"query": "agent"})

    assert "- notes.md" in result
    assert "The AGENT tracks Failures." in result


def test_knowledge_base_search_reports_no_matching_documents(
    temporary_knowledge_base: Path,
) -> None:
    (temporary_knowledge_base / "notes.md").write_text(
        "This document discusses agents.",
        encoding="utf-8",
    )

    result = tools.knowledge_base_search.invoke({"query": "weather"})

    assert result == "No matching documents found."


def test_knowledge_base_search_reports_empty_directory(
    temporary_knowledge_base: Path,
) -> None:
    result = tools.knowledge_base_search.invoke({"query": "agent"})

    assert result == "No .txt or .md documents found in the knowledge base."


def test_knowledge_base_search_matches_filename(
    temporary_knowledge_base: Path,
) -> None:
    (temporary_knowledge_base / "weather-guide.md").write_text(
        "Information about forecasts.",
        encoding="utf-8",
    )

    result = tools.knowledge_base_search.invoke({"query": "weather"})

    assert "- weather-guide.md" in result
    assert "Filename matched; no matching text line found." in result


@pytest.mark.parametrize(
    ("city", "expected_details"),
    [
        ("Jaipur", "34 degrees C, 28% humidity, Sunny"),
        ("Delhi", "31 degrees C, 42% humidity, Partly cloudy"),
        ("Mumbai", "29 degrees C, 76% humidity, Humid with light clouds"),
        ("Bangalore", "26 degrees C, 54% humidity, Mild and cloudy"),
    ],
)
def test_mock_weather_returns_simulated_weather(
    city: str,
    expected_details: str,
) -> None:
    result = tools.mock_weather.invoke({"city": city})

    assert f"Simulated weather for {city}:" in result
    assert expected_details in result
    assert "This is simulated data, not real-time weather." in result


def test_mock_weather_handles_unsupported_city() -> None:
    result = tools.mock_weather.invoke({"city": "Toronto"})

    assert "Unsupported city: Toronto." in result
    assert "Jaipur, Delhi, Mumbai, Bangalore." in result


def test_mock_weather_accepts_case_insensitive_city_input() -> None:
    result = tools.mock_weather.invoke({"city": "bAnGaLoRe"})

    assert "Simulated weather for Bangalore:" in result
    assert "26 degrees C, 54% humidity, Mild and cloudy" in result
