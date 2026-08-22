"""Per-model token pricing, for showing what each reply cost.

DigitalOcean publishes prices at
https://docs.digitalocean.com/products/inference/details/pricing/ but keys the
tables by marketing name ("Claude Opus 5"), never by the API model id, and the
API itself reports no prices. So the mapping below is maintained by hand from
that page, and anything that cannot be mapped confidently is left out — an
unpriced model reports "cost n/a" rather than a made-up number.

OpenRouter, the other gateway, publishes its rates in its own /v1/models
response, so on that provider nothing here has to be maintained by hand - see
from_records below.

Prices are USD per million tokens, transcribed from the page dated PRICE_DATE.
Override or extend them without touching this file by dropping a pricing.json
next to dba.py:

    {"some-model-id": {"input": 0.5, "output": 1.5}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .. import PROJECT_DIR

PRICE_DATE = "2026-08-20"
PRICING_URL = "https://docs.digitalocean.com/products/inference/details/pricing/"
OVERRIDE_FILE = PROJECT_DIR / "pricing.json"

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    """Input/output rates in USD per million tokens.

    Some models are billed at a higher rate once the prompt crosses a
    threshold ("Prompts >200K tokens"); tier_at holds that boundary.
    """

    input: float
    output: float
    tier_at: int | None = None
    tier_input: float | None = None
    tier_output: float | None = None

    def rates_for(self, prompt_tokens: int) -> tuple[float, float]:
        if self.tier_at is not None and prompt_tokens > self.tier_at and self.tier_input is not None:
            return self.tier_input, self.tier_output if self.tier_output is not None else self.output
        return self.input, self.output


_K200 = 200_000
_K272 = 272_000

# Keyed by API model id. Names in comments are the rows on the pricing page.
PRICES: dict[str, Price] = {
    # -- Anthropic ---------------------------------------------------------
    "anthropic-claude-fable-5": Price(10.00, 50.00),
    "anthropic-claude-haiku-4.5": Price(1.00, 5.00),
    "anthropic-claude-opus-5": Price(5.00, 25.00),  # Fast Mode ($10/$50) is not a separate id
    "anthropic-claude-opus-4.8": Price(5.00, 25.00),
    "anthropic-claude-opus-4.7": Price(5.00, 25.00),
    "anthropic-claude-opus-4.6": Price(5.00, 25.00),
    "anthropic-claude-opus-4.5": Price(5.00, 25.00),
    "anthropic-claude-5-sonnet": Price(2.00, 10.00),  # "Claude Sonnet 5"
    "anthropic-claude-4.6-sonnet": Price(3.00, 15.00),
    "anthropic-claude-4.5-sonnet": Price(3.00, 15.00, _K200, 6.00, 22.50),
    # -- OpenAI ------------------------------------------------------------
    "openai-gpt-oss-120b": Price(0.10, 0.70),
    "openai-gpt-oss-20b": Price(0.05, 0.45),
    "openai-gpt-5.6-sol": Price(5.00, 30.00, _K272, 10.00, 45.00),
    "openai-gpt-5.6-terra": Price(2.00, 12.00, _K272, 4.00, 18.00),
    "openai-gpt-5.6-luna": Price(0.20, 1.20, _K272, 0.40, 1.80),
    "openai-gpt-5.5": Price(5.00, 30.00, _K272, 10.00, 45.00),
    "openai-gpt-5.4": Price(2.50, 15.00, _K272, 5.00, 22.50),
    "openai-gpt-5.4-mini": Price(0.75, 4.50),
    "openai-gpt-5.4-nano": Price(0.20, 1.25),
    "openai-gpt-5.4-pro": Price(30.00, 180.00, _K272, 60.00, 270.00),
    "openai-gpt-5.3-codex": Price(1.75, 14.00),
    "openai-gpt-5.2": Price(1.75, 14.00),
    "openai-gpt-5.2-pro": Price(21.00, 168.00),
    "openai-gpt-5": Price(1.25, 10.00),
    "openai-gpt-5-mini": Price(0.25, 2.00),
    "openai-gpt-5-nano": Price(0.05, 0.40),
    "openai-gpt-4.1": Price(2.00, 8.00),
    "openai-gpt-4o": Price(2.50, 10.00),
    "openai-gpt-4o-mini": Price(0.15, 0.60),
    "openai-o1": Price(15.00, 60.00),
    "openai-o3": Price(2.00, 8.00),
    "openai-o3-mini": Price(1.10, 4.40),
    # -- DigitalOcean-hosted ----------------------------------------------
    "qwen3.8-max": Price(2.00, 6.00),  # "Qwen3.8-2.4T-A95B", the only 3.8 model
    "qwen3.5-397b-a17b": Price(0.302, 1.925),
    "deepseek-v4-pro-0813": Price(1.32, 3.96),
    "deepseek-v4-flash-0731": Price(0.080, 0.252),
    "deepseek-v4-pro": Price(0.87, 1.74),
    "deepseek-4-flash": Price(0.068, 0.168),  # "DeepSeek V4 Flash"
    "deepseek-3.2": Price(0.25, 0.80),  # "DeepSeek V3.2"
    "gemma-4-31B-it": Price(0.18, 0.50),  # "Gemma 4"
    "minimax-m2.5": Price(0.225, 0.90),
    "kimi-k3": Price(2.85, 14.25),
    "kimi-k2.6": Price(0.76, 3.20),
    "kimi-k2.5": Price(0.375, 2.025),
    "llama-4-maverick": Price(0.20, 0.696),
    "mistral-3-14B": Price(0.20, 0.20),  # page says "Ministral 3 14B Instruct"
    "nemotron-3-ultra-550b": Price(0.90, 1.70),  # "Nemotron 3 Ultra"
    "nvidia-nemotron-3-super-120b": Price(0.165, 0.358),
    "nemotron-3-nano-omni": Price(0.50, 0.90),  # "Nemotron Nano 3 Omni"
    "nemotron-nano-12b-v2-vl": Price(0.20, 0.60),
    "mimo-v2.5-pro": Price(0.40, 1.50),
    "glm-5.2": Price(0.70, 2.20),
    "glm-5.1": Price(0.975, 4.30),
    "glm-5": Price(0.75, 2.40),
    "arcee-trinity-large-thinking": Price(0.25, 0.90),  # "Trinity Large"
    # Deliberately absent: router:* aliases bill as whatever model they pick,
    # which is not knowable from the request alone.
}


def load_overrides() -> tuple[dict[str, Price], str | None]:
    """Merge pricing.json over the built-in table. Returns (prices, error)."""
    prices = dict(PRICES)
    if not OVERRIDE_FILE.is_file():
        return prices, None
    try:
        raw = json.loads(OVERRIDE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return prices, f"ignoring {OVERRIDE_FILE.name}: {exc}"
    if not isinstance(raw, dict):
        return prices, f"ignoring {OVERRIDE_FILE.name}: expected an object of model -> rates"

    bad: list[str] = []
    for model_id, entry in raw.items():
        if not isinstance(entry, dict):
            bad.append(str(model_id))
            continue
        try:
            price = Price(
                input=float(entry["input"]),
                output=float(entry["output"]),
                tier_at=int(entry["tier_at"]) if entry.get("tier_at") else None,
                tier_input=float(entry["tier_input"]) if entry.get("tier_input") else None,
                tier_output=float(entry["tier_output"]) if entry.get("tier_output") else None,
            )
        except (KeyError, TypeError, ValueError):
            bad.append(str(model_id))
            continue
        prices[str(model_id)] = price

    error = f"{OVERRIDE_FILE.name}: skipped bad entries ({', '.join(sorted(bad))})" if bad else None
    return prices, error


def from_records(records: Iterable[dict]) -> dict[str, Price]:
    """Rates a gateway reports about itself, as a price table.

    OpenRouter puts `pricing` in its /v1/models response, in USD per token, so
    for that provider nothing needs transcribing and no rate goes stale.
    Per-request, image and web-search fees are ignored: this is token cost.
    """
    prices: dict[str, Price] = {}
    for record in records:
        model_id = str(record.get("id") or "").strip()
        reported = record.get("pricing")
        if not model_id or not isinstance(reported, dict):
            continue
        try:
            prompt = float(reported.get("prompt"))
            completion = float(reported.get("completion"))
        except (TypeError, ValueError):
            continue
        # A negative rate means "varies" on some gateways; a guess is worse than
        # "cost n/a", which is what leaving it out produces.
        if prompt < 0 or completion < 0:
            continue
        prices[model_id] = Price(prompt * _PER_MILLION, completion * _PER_MILLION)
    return prices


class PriceBook:
    """Looks up rates and turns token counts into dollars."""

    def __init__(self, prices: dict[str, Price] | None = None, warning: str | None = None):
        if prices is None:
            prices, warning = load_overrides()
        self._prices = prices
        self.warning = warning

    def learn(self, prices: dict[str, Price]) -> int:
        """Add rates discovered at runtime; keep any already known. Returns how many.

        pricing.json is the user's last word and the built-in table was checked
        by hand, so both win over whatever a gateway says about itself.
        """
        added = 0
        for model_id, price in prices.items():
            if model_id not in self._prices:
                self._prices[model_id] = price
                added += 1
        return added

    def has(self, model: str) -> bool:
        return model in self._prices

    def get(self, model: str) -> Price | None:
        return self._prices.get(model)

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
        """USD for one exchange, or None if the model has no published price."""
        price = self._prices.get(model)
        if price is None:
            return None
        in_rate, out_rate = price.rates_for(prompt_tokens)
        return (prompt_tokens * in_rate + completion_tokens * out_rate) / _PER_MILLION


def format_rate(per_million: float) -> str:
    """A per-million rate: at least two decimals, more only when they matter."""
    text = f"{per_million:.4f}".rstrip("0")
    whole, _, fraction = text.partition(".")
    return f"{whole}.{fraction.ljust(2, '0')}"


def format_cost(usd: float | None) -> str:
    """Money at a readable precision; these amounts are usually tiny."""
    if usd is None:
        return "cost n/a"
    if usd <= 0:
        return "$0.00"
    if usd >= 1:
        return f"${usd:,.2f}"
    if usd >= 0.01:
        return f"${usd:.4f}"
    if usd >= 0.000001:
        return f"${usd:.6f}"
    return "<$0.000001"
