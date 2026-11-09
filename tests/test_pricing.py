"""Tests for versioned provider model cost estimation."""

from datetime import date
from decimal import Decimal

import pytest

from app.monitoring.pricing import PRICING_VERSION, estimate_model_cost


@pytest.mark.parametrize(
    ("provider", "model", "rate_input", "rate_output"),
    [
        ("Anthropic", "anthropic/claude-sonnet-5", "2", "10"),
        ("Gemini", "gemini/gemini-4-argon", "2", "10"),
    ],
)
def test_model_cost_uses_separate_decimal_rates_and_records_provenance(
    provider: str,
    model: str,
    rate_input: str,
    rate_output: str,
) -> None:
    estimate = estimate_model_cost(
        provider,
        model,
        {
            "available": True,
            "input_tokens": 1_000,
            "output_tokens": 500,
            "total_tokens": 1_500,
        },
        as_of=date(2026, 11, 9),
    )

    expected = (
        Decimal("1000") * Decimal(rate_input)
        + Decimal("500") * Decimal(rate_output)
    ) / Decimal(1_000_000)
    assert estimate["available"] is True
    assert Decimal(estimate["amount_usd"]) == expected
    assert estimate["currency"] == "USD"
    assert estimate["unit"] == "USD per 1 million tokens"
    assert estimate["pricing_version"] == PRICING_VERSION
    assert estimate["source"].startswith("https://")
    assert estimate["verified_on"] == "2026-11-09"


def test_unknown_model_pricing_is_unavailable() -> None:
    estimate = estimate_model_cost(
        "Groq",
        "configured-groq-model",
        {
            "available": True,
            "input_tokens": 1_000,
            "output_tokens": 500,
            "total_tokens": 1_500,
        },
        as_of=date(2026, 11, 9),
    )

    assert estimate["available"] is False
    assert estimate["amount_usd"] is None
    assert estimate["reason"] == "model_pricing_unavailable"


def test_unavailable_token_usage_is_not_priced() -> None:
    estimate = estimate_model_cost(
        "Gemini",
        "gemini/gemini-4-argon",
        {
            "available": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
        as_of=date(2026, 11, 9),
    )

    assert estimate["available"] is False
    assert estimate["amount_usd"] is None
    assert estimate["reason"] == "token_usage_unavailable"


def test_expired_pricing_is_not_applied() -> None:
    estimate = estimate_model_cost(
        "Gemini",
        "gemini/gemini-4-argon",
        {
            "available": True,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
        as_of=date(2026, 12, 1),
    )

    assert estimate["available"] is False
    assert estimate["reason"] == "pricing_expired"
