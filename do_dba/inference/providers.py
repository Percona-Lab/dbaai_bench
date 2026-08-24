"""Where the models come from: an OpenAI-compatible gateway and its credential.

Every gateway here speaks the same wire format, so the client, catalog, and cost
accounting are shared. What differs is the base URL, which environment variable
holds the key, and how model ids are spelled - DigitalOcean uses flat ids
(`anthropic-claude-opus-5`), OpenRouter uses `vendor/model`.

OpenRouter also reports its own prices through /v1/models, so on that provider
nothing has to be transcribed by hand and no rate goes stale - and it will report
what it actually charged for each reply, which is better still, since a rate table
cannot know about cached prompt tokens or which upstream provider served the
request.

The third is a server somebody runs themselves - LM Studio, vLLM, llama.cpp,
Ollama - which is the same API again with two things taken away: there is usually
no credential to send, and there is no bill, because the hardware was paid for
before the run started. Both of those are properties of the gateway rather than
special cases in the client, so they live here as key_optional and metered.

It has one thing to add, too: LM Studio describes what it is serving on an endpoint
of its own, which is where a context length comes from on a gateway whose
/v1/models carries none. That is detail_path, read by inference/details.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_STALL_TIMEOUT,
    DO_KEY_HELP,
    KEY_ENV_VARS,
    ConfigError,
    stall_timeout,
)

OPENROUTER = "openrouter"
DIGITALOCEAN = "digitalocean"
SELFHOSTED = "selfhosted"

# The Percona box this was written for. It is only a default: point
# DBA_SELFHOSTED_BASE_URL at any OpenAI-compatible server and the rest follows,
# including a laptop running LM Studio on http://127.0.0.1:1234.
SELFHOSTED_BASE_URL = "https://mac-studio-lm.int.percona.com/v1"

# What goes out as the bearer token when a self-hosted server needs no key. The
# SDK will not build a client without a string, and these servers do not read it.
PLACEHOLDER_KEY = "self-hosted"

# Variant suffixes (`:free`, `:nitro`, `:floor`) route the same model through
# different capacity or billing. Fine to ask for, wrong to pick by default.
_VARIANT = re.compile(r":")


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    base_url: str
    base_url_env: str
    key_env: tuple[str, ...]
    model_env: tuple[str, ...]
    # Tried first when no model is named; the hints cover it having been renamed
    # or retired, which on OpenRouter happens week to week.
    default_model: str
    default_hints: tuple[str, ...] = ()
    key_help: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    # Whether the gateway will report what it charged for each reply if asked.
    # OpenRouter will, and that figure is the billed one: it accounts for cache
    # reads, the provider that actually served the request, and the gateway's own
    # rounding, none of which a rate table can reproduce. Where it is available it
    # is used instead of tokens x published rate.
    usage_accounting: bool = False
    # A server on your own network usually has no accounts and no keys, so a
    # missing credential is the normal case rather than a misconfiguration.
    key_optional: bool = False
    # Whether calling this gateway costs money per token. A self-hosted server
    # does not: the run reports $0.00 rather than "cost n/a", and --max-cost has
    # nothing to trip on.
    metered: bool = True
    # How long to wait for a first token before calling the request stalled.
    # DO_INFERENCE_TIMEOUT overrides it either way.
    first_token_wait: float = DEFAULT_STALL_TIMEOUT
    # Where this gateway says more about its models than /v1/models can - a path
    # beside the API root, not under it. See inference/details.py; a gateway
    # without one, or one that has stopped answering, costs nothing but the
    # fields it would have filled in.
    detail_path: str = ""

    def api_key(self) -> str:
        for name in self.key_env:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        if self.key_optional:
            return PLACEHOLDER_KEY
        raise ConfigError(self.key_help)

    def base(self) -> str:
        return _api_root(os.environ.get(self.base_url_env, "").strip() or self.base_url)

    def read_timeout(self) -> float:
        return stall_timeout(self.first_token_wait)

    def detail_url(self) -> str:
        """Where to ask about the models themselves, or "" if this gateway cannot say.

        Beside the API root rather than under it: LM Studio serves the OpenAI
        listing at /v1/models and its own at /api/v0/models, so the /v1 comes off
        first. A base URL mounted somewhere else - behind a proxy, under a path -
        keeps that path, because the two endpoints move together.
        """
        if not self.detail_path:
            return ""
        root = self.base().rstrip("/")
        if not root:
            return ""
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return root + self.detail_path

    def model_from_env(self) -> str:
        for name in self.model_env:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    def choose_default(self, catalog) -> str:
        """The model to use when the operator named none.

        The pinned id first; failing that, the newest-looking member of the first
        preferred family the gateway actually offers. A pinned id alone would
        leave the harness with nothing to run the week it is retired.
        """
        exact = catalog.get(self.default_model) if self.default_model else None
        if exact is not None and exact.is_chat:
            return exact.id
        for hint in self.default_hints:
            family = [
                model for model in catalog.chat
                if hint in model.id.lower() and not _VARIANT.search(model.id)
            ]
            if family:
                return max(family, key=lambda model: _natural(model.id)).id
        # A gateway with nothing pinned is a server that serves whatever was
        # loaded onto it, so there is no default to keep here. One chat model
        # means there is nothing to choose; several means the operator chooses,
        # because a 4B and a 400B on the same box are not interchangeable and
        # picking whichever came back first would decide the run silently.
        if not self.default_model and len(catalog.chat) == 1:
            return catalog.chat[0].id
        return ""


def _api_root(url: str) -> str:
    """A base URL with the API root filled in when it was left off.

    The address quoted for a self-hosted server is usually the host on its own -
    `https://mac-studio-lm.int.percona.com`, `http://127.0.0.1:1234` - because
    that is what its own interface shows. The SDK appends `chat/completions` to
    whatever it is given, and every OpenAI-compatible server mounts that under
    `/v1`, so a bare host would otherwise 404 on the first request. A URL that
    already carries a path is left exactly as it is: a server behind a proxy can
    be mounted anywhere.
    """
    trimmed = (url or "").strip().rstrip("/")
    if not trimmed or urlsplit(trimmed).path:
        return trimmed
    return f"{trimmed}/v1"


def _natural(text: str) -> list:
    """Sort key where 10 comes after 9, so the newest version wins."""
    # re.split on a capturing group alternates non-digit, digit, non-digit, so
    # the same position always holds the same type and comparison is safe.
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text.lower())]


PROVIDERS: dict[str, Provider] = {
    OPENROUTER: Provider(
        name=OPENROUTER,
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        base_url_env="OPENROUTER_BASE_URL",
        key_env=("OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPEN_ROUTER_API_KEY"),
        model_env=("OPENROUTER_MODEL",),
        default_model="anthropic/claude-sonnet-4.5",
        default_hints=(
            "anthropic/claude-sonnet",
            "anthropic/claude-opus",
            "openai/gpt-5",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-chat",
        ),
        key_help=(
            "No OpenRouter credential found.\n\n"
            "Create a key at https://openrouter.ai/keys, then either export it or\n"
            "add it to the .env file next to dba.py:\n\n"
            "    OPENROUTER_API_KEY=sk-or-v1-...\n\n"
            "To use DigitalOcean Serverless Inference instead, pass --provider digitalocean."
        ),
        # Optional attribution OpenRouter shows in its app rankings.
        headers={"X-Title": "do-dba"},
        usage_accounting=True,
    ),
    DIGITALOCEAN: Provider(
        name=DIGITALOCEAN,
        label="DigitalOcean",
        base_url=DEFAULT_BASE_URL,
        base_url_env="DO_INFERENCE_BASE_URL",
        key_env=KEY_ENV_VARS,
        model_env=("DO_INFERENCE_MODEL",),
        default_model=DEFAULT_MODEL,
        default_hints=("anthropic-claude-opus", "anthropic-claude", "openai-gpt-oss"),
        key_help=DO_KEY_HELP,
    ),
    SELFHOSTED: Provider(
        name=SELFHOSTED,
        label="Self-hosted",
        base_url=SELFHOSTED_BASE_URL,
        base_url_env="DBA_SELFHOSTED_BASE_URL",
        key_env=("DBA_SELFHOSTED_KEY", "SELFHOSTED_API_KEY"),
        model_env=("DBA_SELFHOSTED_MODEL",),
        # Nothing pinned and no hints: what this gateway serves is whatever is
        # loaded on it today, which no table here can know. choose_default takes
        # the only chat model when there is one, and otherwise -m says which.
        default_model="",
        key_help=(
            "This server wants a credential.\n\n"
            "Export it or add it to the .env file next to dba.py:\n\n"
            "    DBA_SELFHOSTED_KEY=your_key_here\n\n"
            "Most self-hosted servers need none, which is why the run got as far "
            "as asking."
        ),
        key_optional=True,
        metered=False,
        # The first request to a server like this often loads the weights off
        # disk before it answers - minutes for a large model on a cold box - and
        # a run that dies at step 1 having asked for nothing is the worst way to
        # find that out. Later requests answer at once; a wait costs nothing when
        # nothing is waiting.
        first_token_wait=900.0,
        # LM Studio's own listing, which is where the context length and whether
        # the weights are loaded come from. Nothing depends on it answering: the
        # servers that have no such endpoint say 404 and the listing stands.
        detail_path="/api/v0/models",
    ),
}

NAMES = tuple(PROVIDERS)
# What people actually type.
ALIASES = {
    "do": DIGITALOCEAN,
    "digital-ocean": DIGITALOCEAN,
    "digital_ocean": DIGITALOCEAN,
    "gradient": DIGITALOCEAN,
    "or": OPENROUTER,
    "open-router": OPENROUTER,
    "open_router": OPENROUTER,
    "local": SELFHOSTED,
    "self-hosted": SELFHOSTED,
    "self_hosted": SELFHOSTED,
    "self": SELFHOSTED,
    "lmstudio": SELFHOSTED,
    "lm-studio": SELFHOSTED,
    "vllm": SELFHOSTED,
    "ollama": SELFHOSTED,
}


def get(name: str) -> Provider:
    """Look a provider up by name, alias, or any unambiguous prefix of one."""
    wanted = (name or "").strip().lower()
    wanted = ALIASES.get(wanted, wanted)
    if wanted in PROVIDERS:
        return PROVIDERS[wanted]
    matches = [provider for key, provider in PROVIDERS.items() if key.startswith(wanted)] if wanted else []
    if len(matches) == 1:
        return matches[0]
    raise ConfigError(f"unknown provider {name!r} - choose one of {', '.join(NAMES)}")
