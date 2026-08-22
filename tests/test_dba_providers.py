"""The provider layer, checked offline against OpenRouter-shaped records.

No network and no key: this pins the parts that decide where a run goes and what
it is billed at - provider lookup, credential discovery, default-model choice,
the chat/non-chat split, rates read from the gateway's own model list, and the
hand-kept DigitalOcean table with its >200K price tiers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from do_dba.inference import providers
from do_dba.inference.catalog import Catalog
from do_dba.inference.client import (
    COST_ACCOUNTING,
    InferenceClient,
    _apply_fix,
    _billed,
    _diagnose_bad_request,
)
from do_dba.inference.config import ConfigError
from do_dba.inference.pricing import PRICES, Price, PriceBook, format_cost, from_records

# Shaped like a real GET https://openrouter.ai/api/v1/models response: no
# owned_by, ids as vendor/model, prices as strings in USD per token.
OPENROUTER_MODELS = [
    {
        "id": "anthropic/claude-sonnet-4.5",
        "name": "Anthropic: Claude Sonnet 4.5",
        "context_length": 1000000,
        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        "pricing": {"prompt": "0.000003", "completion": "0.000015", "request": "0", "image": "0.0048"},
    },
    {
        "id": "anthropic/claude-opus-4.5",
        "context_length": 200000,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "pricing": {"prompt": "0.000005", "completion": "0.000025"},
    },
    {
        "id": "anthropic/claude-sonnet-4.5:free",
        "context_length": 200000,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "0", "completion": "0"},
    },
    {
        "id": "openai/gpt-5.1",
        "context_length": 400000,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "0.00000125", "completion": "0.00001"},
    },
    {
        "id": "google/gemini-2.5-pro",
        "context_length": 1048576,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "0.00000125", "completion": "0.00001"},
    },
    {
        "id": "black-forest-labs/flux-1.1-pro",
        "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
        "pricing": {"prompt": "0", "completion": "0", "image": "0.04"},
    },
    {
        "id": "x-ai/grok-4",
        "context_length": 256000,
        "architecture": {"output_modalities": ["text"]},
        # A rate the gateway will not commit to: better left unpriced than guessed.
        "pricing": {"prompt": "-1", "completion": "-1"},
    },
]


def check(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def clear_keys() -> dict[str, str]:
    """Drop every credential from the environment, returning what was removed."""
    removed = {}
    for provider in providers.PROVIDERS.values():
        for name in provider.key_env:
            if name in os.environ:
                removed[name] = os.environ.pop(name)
    return removed


def main() -> int:
    failures: list[str] = []
    catalog = Catalog(OPENROUTER_MODELS)
    openrouter = providers.get("openrouter")
    digitalocean = providers.get("digitalocean")

    # ------------------------------------------------------- provider lookup
    check(failures, providers.get("or").name == providers.OPENROUTER, "'or' should mean openrouter")
    check(failures, providers.get("do").name == providers.DIGITALOCEAN, "'do' should mean digitalocean")
    check(failures, providers.get("OpenRouter").name == providers.OPENROUTER, "lookup is case sensitive")
    check(failures, providers.get("digital").name == providers.DIGITALOCEAN, "a prefix should resolve")
    try:
        providers.get("bedrock")
        failures.append("an unknown provider was accepted")
    except ConfigError as exc:
        check(failures, "openrouter" in str(exc), "the error should list the known providers")

    check(failures, openrouter.base_url == "https://openrouter.ai/api/v1", "wrong OpenRouter base url")
    check(failures, "X-Title" in openrouter.headers, "attribution header missing")

    # -------------------------------------------------------- credentials
    removed = clear_keys()
    try:
        try:
            openrouter.api_key()
            failures.append("a missing key was not reported")
        except ConfigError as exc:
            check(failures, "OPENROUTER_API_KEY" in str(exc), "the error should name the variable")
            check(failures, "--provider digitalocean" in str(exc),
                  "the error should mention the other provider")
        try:
            digitalocean.api_key()
            failures.append("a missing DO key was not reported")
        except ConfigError as exc:
            check(failures, "DIGITALOCEAN_INFERENCE_KEY" in str(exc), "the DO error lost its variable name")

        os.environ["OPENROUTER_KEY"] = "sk-or-v1-second-name"
        check(failures, openrouter.api_key() == "sk-or-v1-second-name", "an alternative key name was ignored")
        os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-preferred"
        check(failures, openrouter.api_key() == "sk-or-v1-preferred", "the preferred key name lost")

        os.environ["OPENROUTER_BASE_URL"] = "http://127.0.0.1:8080/v1"
        check(failures, openrouter.base() == "http://127.0.0.1:8080/v1", "the base url override was ignored")
        del os.environ["OPENROUTER_BASE_URL"]
        check(failures, openrouter.base() == openrouter.base_url, "the default base url did not come back")

        os.environ["OPENROUTER_MODEL"] = "openai/gpt-5.1"
        check(failures, openrouter.model_from_env() == "openai/gpt-5.1", "OPENROUTER_MODEL was ignored")
        del os.environ["OPENROUTER_MODEL"]
    finally:
        for name in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
            os.environ.pop(name, None)
        os.environ.update(removed)

    # ------------------------------------------------------- default model
    check(failures, openrouter.choose_default(catalog) == "anthropic/claude-sonnet-4.5",
          "the pinned default was not chosen")

    without_pin = Catalog([m for m in OPENROUTER_MODELS if m["id"] != "anthropic/claude-sonnet-4.5"])
    fallback = openrouter.choose_default(without_pin)
    check(failures, fallback == "anthropic/claude-opus-4.5",
          f"a retired default should fall to the next family, got {fallback}")
    check(failures, ":free" not in fallback, "a :free variant must not be picked by default")

    only_openai = Catalog([m for m in OPENROUTER_MODELS if m["id"].startswith("openai/")])
    check(failures, openrouter.choose_default(only_openai) == "openai/gpt-5.1",
          "the openai family was not reached")

    # Newest wins inside a family, and 10 comes after 9.
    versions = Catalog([
        {"id": "openai/gpt-5.9", "architecture": {"output_modalities": ["text"]}},
        {"id": "openai/gpt-5.10", "architecture": {"output_modalities": ["text"]}},
    ])
    picked = openrouter.choose_default(versions)
    check(failures, picked == "openai/gpt-5.10", f"version ordering is wrong: picked {picked}")

    images_only = Catalog([m for m in OPENROUTER_MODELS if m["id"].startswith("black-forest")])
    check(failures, openrouter.choose_default(images_only) == "",
          "with no usable model the harness must say so, not pick one")

    # ---------------------------------------------------- chat/non-chat split
    ids = {model.id for model in catalog.chat}
    check(failures, "black-forest-labs/flux-1.1-pro" not in ids, "an image model was offered as chat")
    check(failures, "google/gemini-2.5-pro" in ids, "a text model was hidden")
    check(failures, catalog.get("anthropic/claude-sonnet-4.5").provider == "Anthropic",
          "vendor-prefixed ids should group under the vendor")
    check(failures, catalog.get("x-ai/grok-4").provider == "X Ai",
          f"unknown vendors should use the slug, got {catalog.get('x-ai/grok-4').provider}")
    check(failures, catalog.get("anthropic/claude-sonnet-4.5").context_label == "1M ctx",
          "context_length was not read")
    check(failures, catalog.get("openai/gpt-5.1").context_label == "400K ctx",
          "context_length was not read")

    # A model whose name says "image" but which answers in text is still chat:
    # the reported modalities are the authority, not the id.
    named_badly = Catalog([{
        "id": "vendor/vision-image-reader",
        "architecture": {"input_modalities": ["image"], "output_modalities": ["text"]},
    }])
    check(failures, len(named_badly.chat) == 1, "modalities should beat the id when both are present")

    # ------------------------------------------------------------ -m lookup
    check(failures, catalog.resolve("openai/gpt-5.1")[0].id == "openai/gpt-5.1",
          "an exact id should resolve to itself")
    check(failures, catalog.resolve("gemini")[0].id == "google/gemini-2.5-pro",
          "a unique fragment should resolve")
    # A variant suffix makes the plain name ambiguous, which is the right answer:
    # :free is the same model on different capacity, and picking one silently
    # would be picking a billing arrangement on the operator's behalf.
    variant, variants = catalog.resolve("claude-sonnet-4.5")
    check(failures, variant is None and {m.id for m in variants} == {
        "anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5:free"},
        f"a name shared with a variant should list both, got {[m.id for m in variants]}")
    ambiguous, candidates = catalog.resolve("anthropic")
    check(failures, ambiguous is None and len(candidates) > 1,
          "an ambiguous name should return candidates rather than guess")
    # --list-models prints no numbers, so a bare number is a fragment like any
    # other and must not silently select the nth model.
    numbered, numbered_candidates = catalog.resolve("5")
    check(failures, numbered is None and len(numbered_candidates) > 1,
          f"a bare number picked a model: {numbered.id if numbered else None}")
    check(failures, catalog.resolve("no-such-model") == (None, []),
          "a name matching nothing should return nothing")

    # ---------------------------------------------------------------- pricing
    learned = from_records(OPENROUTER_MODELS)
    check(failures, learned["anthropic/claude-sonnet-4.5"] == Price(3.0, 15.0),
          f"per-token to per-million conversion is wrong: {learned.get('anthropic/claude-sonnet-4.5')}")
    check(failures, learned["openai/gpt-5.1"] == Price(1.25, 10.0), "sub-dollar rates converted wrongly")
    check(failures, "x-ai/grok-4" not in learned, "a negative rate should be left unpriced")
    check(failures, learned["anthropic/claude-sonnet-4.5:free"] == Price(0.0, 0.0),
          "a free model should price at zero, not be dropped")

    book = PriceBook(prices={"anthropic/claude-sonnet-4.5": Price(99.0, 99.0)}, warning=None)
    added = book.learn(learned)
    check(failures, book.get("anthropic/claude-sonnet-4.5") == Price(99.0, 99.0),
          "a hand-set rate must win over what the gateway reports")
    check(failures, added == len(learned) - 1, f"learn() should report what it added, said {added}")
    check(failures, book.get("openai/gpt-5.1") == Price(1.25, 10.0), "a learned rate is missing")
    check(failures, book.cost("x-ai/grok-4", 1000, 100) is None, "an unpriced model must not invent a cost")

    cost = book.cost("openai/gpt-5.1", 24_000, 3_400)
    expected = (24_000 * 1.25 + 3_400 * 10.0) / 1_000_000
    check(failures, cost is not None and abs(cost - expected) < 1e-12,
          f"cost is {cost}, expected {expected}")
    check(failures, format_cost(cost) == "$0.0640", f"cost formatting drifted: {format_cost(cost)}")

    # ------------------------------------------------- the hand-kept DO table
    # from_records covers OpenRouter, where rates arrive with the model list. On
    # DigitalOcean the table in pricing.py is all there is, and its tiered rows
    # are the only place the >200K boundary is exercised at all.
    builtin = PriceBook(prices=dict(PRICES), warning=None)
    tiered = [
        ("anthropic-claude-opus-5", 42, 4, 42 * 5 + 4 * 25),
        ("llama-4-maverick", 42, 4, 42 * 0.20 + 4 * 0.696),
        # 250K crosses into $6/$22.50; 200K exactly does not - the page says
        # "Prompts >200K", so the boundary is exclusive and the low rate holds.
        ("anthropic-claude-4.5-sonnet", 250_000, 4, 250_000 * 6 + 4 * 22.50),
        ("anthropic-claude-4.5-sonnet", 200_000, 4, 200_000 * 3 + 4 * 15),
        ("anthropic-claude-4.5-sonnet", 200_001, 4, 200_001 * 6 + 4 * 22.50),
    ]
    for model, prompt_tokens, completion_tokens, per_million in tiered:
        want = per_million / 1_000_000
        got = builtin.cost(model, prompt_tokens, completion_tokens)
        check(failures, got is not None and abs(got - want) < 1e-12,
              f"{model} at {prompt_tokens} in / {completion_tokens} out cost {got}, expected {want}")
    check(failures, builtin.cost("router:general", 100, 10) is None,
          "a router alias bills as whatever it picked, so it must stay unpriced")

    # ------------------------------------------- what the reply actually cost
    # A rate table is an estimate however carefully it is kept: it cannot know
    # that half the prompt was served from a cache, or which of several upstream
    # providers took the request and at what rate. OpenRouter will report what it
    # charged if the request asks, and that figure is the billed one. DigitalOcean
    # is not asked, because a gateway is entitled to refuse an unknown field.
    check(failures, openrouter.usage_accounting and not digitalocean.usage_accounting,
          "the wrong gateways are being asked what a reply cost")
    unreachable = "http://127.0.0.1:1/v1"  # nothing is sent; only the parameters are read
    asked = InferenceClient(api_key="k", base_url=unreachable, usage_accounting=True)._accounting()
    check(failures, asked == {"extra_body": {"usage": {"include": True}}},
          f"the cost report is not being asked for: {asked}")
    check(failures, InferenceClient(api_key="k", base_url=unreachable)._accounting() == {},
          "a gateway that reports no costs should be sent nothing extra")

    for raw, want, why in (
        ({"cost": 0.0123}, 0.0123, "a reported cost"),
        ({"cost": "0.0123"}, 0.0123, "a cost that arrived as a string"),
        # Zero is an answer, not a silence: a free model and a fully cached prompt
        # both really do cost nothing, and estimating over the gateway's own zero
        # would put a number in the report that is not on the bill.
        ({"cost": 0}, 0.0, "a reply that was free"),
        ({}, None, "a gateway that said nothing"),
        ({"cost": None}, None, "an explicit null"),
        ({"cost": "n/a"}, None, "junk in the cost field"),
        ({"cost": -1}, None, "a negative cost"),
    ):
        got = _billed(raw)
        check(failures, got == want and type(got) is type(want),
              f"{why} read as {got!r}, want {want!r}")

    # The accounting flag is an extension, so a gateway that never heard of it may
    # refuse the whole request. Losing the exact figure is worth a retry; losing
    # the reply is not.
    params = {"model": "m", "messages": [], "extra_body": dict(COST_ACCOUNTING)}
    fix = _diagnose_bad_request("Unrecognized request argument supplied: usage", params, set())
    check(failures, fix == "drop:usage", f"a refused cost report was diagnosed as {fix!r}")
    retried = _apply_fix("drop:usage", params)
    check(failures, "extra_body" not in retried and retried["model"] == "m",
          f"dropping the cost report mangled the request: {retried}")
    check(failures, "extra_body" in params, "the fix mutated the caller's parameters")
    check(failures, _diagnose_bad_request("no usage here either", params, {"drop:usage"}) is None,
          "the same fix was offered twice, which would retry until it gave up")

    # Rubbish in the pricing block must not take the run down with it.
    junk = from_records([
        {"id": "a/b"},
        {"id": "a/c", "pricing": "free"},
        {"id": "a/d", "pricing": {"prompt": "abc", "completion": "1"}},
        {"id": "", "pricing": {"prompt": "1", "completion": "1"}},
        {"pricing": {"prompt": "1", "completion": "1"}},
    ])
    check(failures, junk == {}, f"malformed pricing should be skipped, got {junk}")

    print(f"{'FAILURES' if failures else 'all checks passed'}")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
