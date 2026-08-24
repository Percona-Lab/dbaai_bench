"""The provider layer, checked offline against OpenRouter-shaped records.

No network and no key: this pins the parts that decide where a run goes and what
it is billed at - provider lookup, credential discovery, default-model choice,
the chat/non-chat split, rates read from the gateway's own model list, and the
hand-kept DigitalOcean table with its >200K price tiers.

The self-hosted gateway is here too, from LM Studio-shaped records: no key, no
prices, and a model list that is whatever was loaded onto the box that morning -
plus the second endpoint that fills in what /v1/models has no field for, which
context lengths and whether the weights are in memory come from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from do_dba.inference import details, providers
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


# Shaped like a real GET /v1/models from the LM Studio server on the Mac Studio:
# no prices, no context lengths, and owned_by naming the box rather than whoever
# made the model. Ids come both ways, with a vendor prefix and without.
SELFHOSTED_MODELS = [
    {"id": "openai/gpt-oss-20b", "object": "model", "owned_by": "organization_owner"},
    {"id": "qwen/qwen3.8-27b", "object": "model", "owned_by": "organization_owner"},
    {"id": "glm-5.2", "object": "model", "owned_by": "organization_owner"},
    {"id": "minimax-m3-mlx", "object": "model", "owned_by": "organization_owner"},
    # An embedding model, which /v1/chat/completions will not serve.
    {"id": "text-embedding-nomic-embed-text-v1.5", "object": "model",
     "owned_by": "organization_owner"},
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

    # ------------------------------------------------- the self-hosted gateway
    # A server somebody runs themselves is the same API with two things taken
    # away: no credential to send, and no bill. Both are properties of the
    # gateway, so both are checked here rather than in the client.
    selfhosted = providers.get("selfhosted")
    for typed in ("local", "self-hosted", "lmstudio", "vllm", "ollama", "self"):
        check(failures, providers.get(typed).name == providers.SELFHOSTED,
              f"'{typed}' should mean the self-hosted gateway")
    check(failures, selfhosted.base_url == "https://mac-studio-lm.int.percona.com/v1",
          f"wrong default endpoint: {selfhosted.base_url}")
    check(failures, not selfhosted.metered and not selfhosted.usage_accounting,
          "a server you own bills nothing per token and reports no costs")
    check(failures, openrouter.metered and digitalocean.metered,
          "a hosted gateway does bill per token")
    check(failures, selfhosted.first_token_wait > providers.DEFAULT_STALL_TIMEOUT,
          "a cold server loading weights needs longer than a hosted gateway")

    lm_removed = clear_keys()
    try:
        # The key is the difference between this provider and the others: a
        # missing one is the normal case, not a misconfiguration to report.
        check(failures, selfhosted.api_key() == providers.PLACEHOLDER_KEY,
              f"a self-hosted server should need no key, got {selfhosted.api_key()!r}")
        os.environ["DBA_SELFHOSTED_KEY"] = "behind-a-proxy"
        check(failures, selfhosted.api_key() == "behind-a-proxy",
              "a key was ignored on a server that wants one")

        # The address people quote is the host alone, because that is what the
        # server's own interface shows. /v1 is where the API lives either way.
        for given, want in (
            ("https://mac-studio-lm.int.percona.com",
             "https://mac-studio-lm.int.percona.com/v1"),
            ("https://mac-studio-lm.int.percona.com/",
             "https://mac-studio-lm.int.percona.com/v1"),
            ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1"),
            ("http://127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1"),
            ("http://127.0.0.1:1234/v1/", "http://127.0.0.1:1234/v1"),
            # Already mounted somewhere: left alone, because a proxy can put it
            # anywhere and guessing would break the one setup that was explicit.
            ("https://gateway.example/openai/v1", "https://gateway.example/openai/v1"),
        ):
            os.environ["DBA_SELFHOSTED_BASE_URL"] = given
            got = selfhosted.base()
            check(failures, got == want, f"{given} became {got}, want {want}")
        del os.environ["DBA_SELFHOSTED_BASE_URL"]
        check(failures, selfhosted.base() == selfhosted.base_url,
              "the default endpoint did not come back")
    finally:
        os.environ.pop("DBA_SELFHOSTED_KEY", None)
        os.environ.pop("DBA_SELFHOSTED_BASE_URL", None)
        os.environ.update(lm_removed)

    loaded = Catalog(SELFHOSTED_MODELS)
    check(failures, [m.id for m in loaded.other] == ["text-embedding-nomic-embed-text-v1.5"],
          f"the embedding model should not be offered as chat: {[m.id for m in loaded.other]}")
    # Nothing is pinned, because what this gateway serves is whatever was loaded
    # onto it. With several models the operator says which; a 20B and a 27B on one
    # box are not interchangeable, and picking the first listed would decide the
    # run silently.
    check(failures, selfhosted.choose_default(loaded) == "",
          f"a loaded box should not be guessed at: {selfhosted.choose_default(loaded)}")
    single = Catalog([SELFHOSTED_MODELS[1], SELFHOSTED_MODELS[4]])
    check(failures, selfhosted.choose_default(single) == "qwen/qwen3.8-27b",
          "one chat model needs no choosing")
    embeddings_only = Catalog([SELFHOSTED_MODELS[4]])
    check(failures, selfhosted.choose_default(embeddings_only) == "",
          "a server with nothing chat-capable must say so, not pick the embedder")
    # The rule is gated on there being no pinned default: a hosted gateway whose
    # default was retired must still refuse rather than run on whatever is left.
    check(failures, openrouter.choose_default(single) == "" and digitalocean.choose_default(single) == "",
          "a pinned provider inherited the take-the-only-model rule")

    # owned_by names the box, not the model's author, so it must not become a
    # heading: an id with no known vendor in it falls back to "Other".
    anonymous = Catalog([{"id": "some-local-finetune", "owned_by": "organization_owner"}])
    check(failures, anonymous.all[0].provider == "Other",
          f"the server's own owned_by leaked into the listing: {anonymous.all[0].provider}")
    check(failures, loaded.get("glm-5.2").provider == "Z.ai",
          "a flat id should still group under its vendor")
    check(failures, loaded.get("qwen/qwen3.8-27b").provider == "Alibaba / Qwen",
          "a vendor-prefixed id should group under its vendor")
    check(failures, loaded.resolve("qwen3.8")[0].id == "qwen/qwen3.8-27b",
          "-m should take a fragment of a self-hosted id")

    # ------------------------------------ what /v1/models could not have said
    # LM Studio answers the questions the OpenAI listing has no field for on a
    # REST endpoint beside it, so the address is derived from the API root with
    # the /v1 taken off - and any path a proxy mounted the server under kept.
    lm_removed = clear_keys()
    try:
        for given, want in (
            ("https://mac-studio-lm.int.percona.com",
             "https://mac-studio-lm.int.percona.com/api/v0/models"),
            ("https://mac-studio-lm.int.percona.com/v1",
             "https://mac-studio-lm.int.percona.com/api/v0/models"),
            ("http://127.0.0.1:1234/", "http://127.0.0.1:1234/api/v0/models"),
            ("https://gateway.example/openai/v1", "https://gateway.example/openai/api/v0/models"),
        ):
            os.environ["DBA_SELFHOSTED_BASE_URL"] = given
            got = selfhosted.detail_url()
            check(failures, got == want, f"{given} asks about models at {got}, want {want}")
    finally:
        os.environ.pop("DBA_SELFHOSTED_BASE_URL", None)
        os.environ.update(lm_removed)
    # A gateway with no such endpoint is not asked, and details.described then
    # hands the records straight back rather than reaching for a URL of its own.
    for hosted in (openrouter, digitalocean):
        check(failures, hosted.detail_url() == "",
              f"{hosted.label} has no detail endpoint but named one: {hosted.detail_url()}")
    check(failures, details.described(SELFHOSTED_MODELS, "") == SELFHOSTED_MODELS,
          "a provider that cannot say more should leave the listing alone")
    check(failures, details.fetch("") == [], "an empty URL should not be fetched")

    # The merge, which is an addition and never a correction: /v1/models is the
    # listing the chat endpoint agrees with, so a detail record may fill fields it
    # left out and nothing else. Ids match case-insensitively; a model only the
    # detail endpoint knows about is dropped, because it cannot be asked for.
    merged = details.merge(
        [{"id": "qwen/qwen3.8-27b", "owned_by": "organization_owner", "type": "llm"},
         {"id": "glm-5.2", "owned_by": "organization_owner"}],
        [{"id": "QWEN/QWEN3.8-27B", "type": "embeddings", "state": "loaded",
          "max_context_length": 262144, "loaded_context_length": 32768, "publisher": "qwen"},
         {"id": "qwen/qwen3.8-27b-q4", "type": "llm", "max_context_length": 262144}],
    )
    check(failures, [record["id"] for record in merged] == ["qwen/qwen3.8-27b", "glm-5.2"],
          f"the merge changed which models exist: {[r['id'] for r in merged]}")
    check(failures, merged[0]["type"] == "llm",
          f"the detail endpoint overwrote what /v1/models said: {merged[0]['type']!r}")
    check(failures, merged[0]["state"] == "loaded" and merged[0]["publisher"] == "qwen",
          f"the absent fields were not filled: {merged[0]}")
    check(failures, "state" not in merged[1],
          "a model with no detail record was given another model's fields")
    check(failures, details.merge(SELFHOSTED_MODELS, []) == SELFHOSTED_MODELS,
          "an endpoint that answered nothing should cost the listing nothing")

    # And what the catalog then makes of those fields. The window a request will
    # actually hit is the one the weights were loaded with, not the larger one the
    # file allows; `state` is the only way to know a step will wait for a 27B model
    # to be read off disk; and `type` settles chat-or-not for a model whose name
    # says neither - "muse-glimmer-30b" is an id no hint would catch either way.
    enriched = Catalog(details.merge(SELFHOSTED_MODELS, [
        {"id": "qwen/qwen3.8-27b", "type": "vlm", "state": "loaded",
         "max_context_length": 262144, "loaded_context_length": 32768},
        {"id": "openai/gpt-oss-20b", "type": "llm", "state": "not-loaded",
         "max_context_length": 131072},
        {"id": "glm-5.2", "type": "embeddings", "state": "loading", "max_context_length": 1048576},
    ]))
    warm = enriched.get("qwen/qwen3.8-27b")
    cold = enriched.get("openai/gpt-oss-20b")
    check(failures, warm.context_window == 32768 and warm.context_label == "32K ctx",
          f"the loaded window lost to the maximum: {warm.context_window}")
    check(failures, warm.loaded is True, f"a model in memory read as {warm.loaded!r}")
    check(failures, cold.context_window == 131072 and cold.loaded is False,
          f"a model on disk read as {cold.context_window}/{cold.loaded!r}")
    check(failures, enriched.get("glm-5.2").loaded is False,
          "a model still loading is not loaded yet")
    check(failures, not enriched.get("glm-5.2").is_chat,
          "a declared embedding model was offered as chat")
    check(failures, enriched.get("minimax-m3-mlx").loaded is None,
          "a model the server said nothing about was reported as cold")
    # A hosted gateway keeps its own models warm and never mentions it, so there is
    # nothing to print - and nothing to mistake for a model that is not there.
    check(failures, all(model.loaded is None for model in catalog.all),
          "a hosted gateway was reported as having models in and out of memory")

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
    billed = InferenceClient(api_key="k", base_url=unreachable, usage_accounting=True)
    plain = InferenceClient(api_key="k", base_url=unreachable)
    asked = billed._extensions()
    check(failures, asked == {"extra_body": {"usage": {"include": True}}},
          f"the cost report is not being asked for: {asked}")
    check(failures, plain._extensions() == {},
          "a gateway that reports no costs should be sent nothing extra")

    # Both extensions share one extra_body, so asking for thinking must not stop a
    # run asking what it cost - the two travel in the same dict and the second
    # would otherwise replace the first.
    both = billed._extensions("high")
    check(failures, both == {"extra_body": {"usage": {"include": True},
                                            "reasoning": {"effort": "high"}}},
          f"an effort displaced the cost report: {both}")
    check(failures, plain._extensions("low") == {"extra_body": {"reasoning": {"effort": "low"}}},
          "an effort must travel even where the gateway reports no costs")
    check(failures, plain._extensions(None) == {} and billed._extensions("") == asked,
          "an unset effort must send nothing, not an empty reasoning block")

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
