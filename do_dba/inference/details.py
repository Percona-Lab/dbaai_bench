"""What /v1/models leaves out, from wherever the gateway will say it.

The OpenAI model list is three fields - id, object, owned_by - and that is all a
self-hosted server has to give: no context length, no prices, no way to tell an
embedding model from a chat one except by its name. LM Studio publishes the rest
on a REST endpoint of its own, /api/v0/models, alongside the OpenAI one:

    {"id": "qwen/qwen3.8-27b", "type": "vlm", "state": "loaded",
     "max_context_length": 262144, "loaded_context_length": 262144,
     "quantization": "8bit", "arch": "qwen3_5", ...}

Which is worth having. The context length is the one number a run can be measured
against, and `state` says whether the weights are in memory - the difference
between a first step that answers in seconds and one that waits minutes for a
27B model to be read off disk.

This is an enrichment and never a requirement. vLLM, llama.cpp and Ollama have no
such endpoint and answer 404; a proxy may refuse it; a future version may change
its shape. Anything that fails, times out, or comes back unrecognisable leaves the
catalog exactly as /v1/models described it, and the caller is not told - there is
nothing for an operator to do about a server that simply has less to say.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

# Short on purpose. This is a side lookup on a listing that has already been
# fetched, so a server that is slow to answer it should not hold up the run - and
# unlike a first chat request, nothing here is waiting for weights to load.
DEFAULT_TIMEOUT = 10.0

# The fields worth carrying over. Everything else LM Studio reports describes the
# file on disk rather than the model as the harness sees it.
_CARRIED = (
    "loaded_context_length",
    "max_context_length",
    "state",
    "type",
    "publisher",
    "quantization",
    "arch",
    "capabilities",
)


def fetch(url: str, api_key: str = "", timeout: float = DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
    """The records at a model-detail endpoint, or [] if there are none to be had.

    Deliberately not through the OpenAI SDK: this is not an OpenAI endpoint, it
    sits outside /v1, and the SDK would append its base path to the URL.
    """
    if not url:
        return []
    request = Request(url, headers={"Accept": "application/json"})
    if api_key:
        # Only of use behind a proxy that authenticates; LM Studio itself ignores
        # it, as it ignores the one on every other request.
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - the URL is the operator's own
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (URLError, OSError, ValueError):
        # HTTPError, a refused connection, a timeout, a body that is not JSON:
        # all the same event here, which is that the server did not answer this.
        return []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    return [record for record in data if isinstance(record, dict) and record.get("id")]


def merge(records: list[dict[str, Any]], details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The /v1/models records with the detail endpoint's fields filled in.

    What /v1/models said is never overwritten, only added to: the OpenAI listing
    is the one the chat endpoint agrees with, and a model that appears in the
    details but not in it is not offered - it cannot be asked for anyway.
    """
    by_id = {str(record["id"]).strip().lower(): record for record in details}
    if not by_id:
        return list(records)
    merged: list[dict[str, Any]] = []
    for record in records:
        extra = by_id.get(str(record.get("id", "")).strip().lower())
        if not extra:
            merged.append(record)
            continue
        filled = dict(record)
        for key in _CARRIED:
            if key not in filled and extra.get(key) is not None:
                filled[key] = extra[key]
        merged.append(filled)
    return merged


def described(
    records: list[dict[str, Any]],
    url: str,
    api_key: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """One call for the whole job: fetch the details, if any, and merge them in."""
    if not url:
        return list(records)
    return merge(records, fetch(url, api_key=api_key, timeout=timeout))
