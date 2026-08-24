"""Credential discovery, .env loading, and the gateway defaults built on them."""

from __future__ import annotations

import os
from pathlib import Path

from .. import PROJECT_DIR

DEFAULT_BASE_URL = "https://inference.do-ai.run/v1"
DEFAULT_MODEL = "anthropic-claude-opus-5"
# Generous enough for a reasoning model to think before its first token, short
# enough that a model which never responds does not hang the run.
DEFAULT_STALL_TIMEOUT = 180.0

# A model access key is the recommended credential; a DigitalOcean personal
# access token also works. We accept whichever name the user happened to use.
KEY_ENV_VARS = (
    "DIGITALOCEAN_INFERENCE_KEY",
    "DO_INFERENCE_KEY",
    "MODEL_ACCESS_KEY",
    "DO_MODEL_ACCESS_KEY",
    "DIGITALOCEAN_TOKEN",
    "DIGITALOCEAN_ACCESS_TOKEN",
)


DO_KEY_HELP = (
    "No DigitalOcean inference credential found.\n\n"
    "Create a model access key in the DigitalOcean control panel under\n"
    "Gradient AI Platform -> Serverless Inference -> Model access keys, then\n"
    "either export it or add it to the .env file next to dba.py:\n\n"
    "    DIGITALOCEAN_INFERENCE_KEY=your_key_here\n\n"
    f"Recognized variable names: {', '.join(KEY_ENV_VARS)}"
)


class ConfigError(RuntimeError):
    """A misconfiguration worth reporting without a traceback."""


def load_dotenv(path: Path | None = None) -> None:
    """Merge a .env file into os.environ without overriding real env vars."""
    env_path = path or (PROJECT_DIR / ".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def find_api_key() -> str:
    """Return the first DigitalOcean credential found, or explain how to make one.

    Provider.api_key() is what the harness uses; this is for the small scripts
    that only ever talk to DigitalOcean.
    """
    for name in KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise ConfigError(DO_KEY_HELP)


def base_url() -> str:
    return os.environ.get("DO_INFERENCE_BASE_URL", "").strip() or DEFAULT_BASE_URL


def stall_timeout(default: float = DEFAULT_STALL_TIMEOUT) -> float:
    """Seconds to wait for the next byte before giving up on a stream.

    DO_INFERENCE_TIMEOUT wins wherever it is set. The default is the caller's,
    because how long a first token can reasonably take is a property of where the
    model is: a hosted gateway has the weights loaded already, and a server on
    your own network may be reading 60GB off a disk before it answers at all.
    """
    raw = os.environ.get("DO_INFERENCE_TIMEOUT", "").strip()
    try:
        value = float(raw)
        return value if value > 0 else default
    except ValueError:
        return default
