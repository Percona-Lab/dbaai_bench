"""Where the models come from: an OpenAI-compatible gateway and its credential.

Both gateways here speak the same wire format, so the client, catalog, and cost
accounting are shared. What differs is the base URL, which environment variable
holds the key, and how model ids are spelled - DigitalOcean uses flat ids
(`anthropic-claude-opus-5`), OpenRouter uses `vendor/model`.

OpenRouter also reports its own prices through /v1/models, so on that provider
nothing has to be transcribed by hand and no rate goes stale - and it will report
what it actually charged for each reply, which is better still, since a rate table
cannot know about cached prompt tokens or which upstream provider served the
request.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, DO_KEY_HELP, KEY_ENV_VARS, ConfigError

OPENROUTER = "openrouter"
DIGITALOCEAN = "digitalocean"

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

    def api_key(self) -> str:
        for name in self.key_env:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        raise ConfigError(self.key_help)

    def base(self) -> str:
        return os.environ.get(self.base_url_env, "").strip() or self.base_url

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
        exact = catalog.get(self.default_model)
        if exact is not None and exact.is_chat:
            return exact.id
        for hint in self.default_hints:
            family = [
                model for model in catalog.chat
                if hint in model.id.lower() and not _VARIANT.search(model.id)
            ]
            if family:
                return max(family, key=lambda model: _natural(model.id)).id
        return ""


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
