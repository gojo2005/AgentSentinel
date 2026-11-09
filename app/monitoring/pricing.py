"""Versioned model pricing for estimated provider token costs.

Rates are USD per one million tokens, taken from each provider's official
pricing documentation and verified on the date recorded with the rate.
Unknown model identifiers are intentionally not priced.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any


PRICING_VERSION = "2026-11-09-v1"
_TOKENS_PER_MILLION = Decimal(1_000_000)
_COST_QUANTUM = Decimal("0.000000000001")


@dataclass(frozen=True)
class ModelPricing:
    provider: str
    model: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    currency: str
    unit: str
    source: str
    verified_on: date
    effective_through: date | None = None


_PRICING_CATALOG = {
    ("anthropic", "anthropic/claude-sonnet-5"): ModelPricing(
        provider="Anthropic",
        model="anthropic/claude-sonnet-5",
        input_usd_per_million=Decimal("2"),
        output_usd_per_million=Decimal("10"),
        currency="USD",
        unit="USD per 1 million tokens",
        source="https://platform.claude.com/docs/en/about-claude/pricing",
        verified_on=date(2026, 11, 9),
    ),
    ("gemini", "gemini/gemini-4-argon"): ModelPricing(
        provider="Gemini",
        model="gemini/gemini-4-argon",
        input_usd_per_million=Decimal("2"),
        output_usd_per_million=Decimal("10"),
        currency="USD",
        unit="USD per 1 million tokens",
        source="https://ai.google.dev/gemini-api/docs/pricing",
        verified_on=date(2026, 11, 9),
        effective_through=date(2026, 11, 30),
    ),
}


def _unavailable(
    reason: str,
    *,
    pricing: ModelPricing | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "amount_usd": None,
        "currency": pricing.currency if pricing is not None else "USD",
        "unit": (
            pricing.unit if pricing is not None else "USD per 1 million tokens"
        ),
        "pricing_version": PRICING_VERSION,
        "reason": reason,
    }
    if pricing is not None:
        result.update(
            {
                "input_usd_per_million": str(pricing.input_usd_per_million),
                "output_usd_per_million": str(pricing.output_usd_per_million),
                "source": pricing.source,
                "verified_on": pricing.verified_on.isoformat(),
            }
        )
    return result


def estimate_model_cost(
    provider: str | None,
    model: str | None,
    token_usage: dict[str, Any],
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Estimate a single LLM call cost or return an explicit unavailable result."""
    catalog_key = (
        provider.casefold(),
        model,
    ) if provider and model else None
    pricing = _PRICING_CATALOG.get(catalog_key) if catalog_key else None
    if pricing is None:
        return _unavailable("model_pricing_unavailable", pricing=None)

    effective_date = as_of or date.today()
    if effective_date < pricing.verified_on:
        return _unavailable("pricing_not_yet_verified", pricing=pricing)
    if (
        pricing.effective_through is not None
        and effective_date > pricing.effective_through
    ):
        return _unavailable("pricing_expired", pricing=pricing)

    input_tokens = token_usage.get("input_tokens")
    output_tokens = token_usage.get("output_tokens")
    if (
        token_usage.get("available") is not True
        or not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or input_tokens < 0
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens < 0
    ):
        return _unavailable("token_usage_unavailable", pricing=pricing)

    amount = (
        Decimal(input_tokens) * pricing.input_usd_per_million
        + Decimal(output_tokens) * pricing.output_usd_per_million
    ) / _TOKENS_PER_MILLION
    amount = amount.quantize(_COST_QUANTUM)
    return {
        "available": True,
        "amount_usd": format(amount, "f"),
        "currency": pricing.currency,
        "unit": pricing.unit,
        "pricing_version": PRICING_VERSION,
        "input_usd_per_million": str(pricing.input_usd_per_million),
        "output_usd_per_million": str(pricing.output_usd_per_million),
        "source": pricing.source,
        "verified_on": pricing.verified_on.isoformat(),
    }
