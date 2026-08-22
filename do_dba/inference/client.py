"""Thin wrapper over the OpenAI SDK, pointed at whichever gateway is in use.

DigitalOcean Serverless Inference and OpenRouter both expose the OpenAI API, so
the only per-gateway details here are the base URL, any extra headers, the name to
blame in an error message, and whether the gateway will say what a reply cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    Timeout,
)

from .config import DEFAULT_BASE_URL, stall_timeout


class InferenceError(RuntimeError):
    """An API failure that should be shown to the user without a traceback."""


@dataclass
class Completion:
    """A whole reply, for callers that drive a loop rather than a screen."""

    text: str = ""
    reasoning: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""
    # What the gateway says it charged for this reply, in USD, or None when it
    # says nothing - see COST_ACCOUNTING below. When it is set it is the billed
    # amount and beats anything worked out from a rate table.
    cost: float | None = None
    # The gateway's own id for the reply. OpenRouter calls this a generation id
    # and keys its activity page by it, so recording it is what makes a run's
    # cost line checkable against the bill afterwards.
    id: str = ""


@dataclass
class Chunk:
    """One streamed delta: visible text, hidden reasoning, or a usage report."""

    text: str = ""
    reasoning: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    # The model the service says served this request, which is not always the
    # one asked for: a router:* alias reports whatever it picked. Billing
    # follows the model that ran, so this is what the cost should be priced on.
    model: str = ""
    cost: float | None = None
    id: str = ""


# OpenRouter's usage accounting: ask for it in the body and the usage object
# comes back with `cost`, the amount deducted from the account for that reply.
# It is the only figure that agrees with the gateway's own report, because a
# price table cannot know what a run is actually billed:
#
#   - cached prompt tokens are charged at a fraction of the input rate, and
#     writing to the cache costs more than not using it;
#   - the same model id is served by several upstream providers at different
#     rates, and a fallback provider prices differently from the first choice;
#   - the gateway rounds, adds per-request fees for some providers, and applies
#     whatever discounts the account has.
#
# Not sent to gateways that would reject an unknown field; the provider says
# whether its gateway understands it. With a bring-your-own-key account `cost`
# is the gateway's fee alone, the upstream charge having gone to the provider
# directly, so it still answers "what did this run cost me here".
COST_ACCOUNTING = {"usage": {"include": True}}


# Parameters that some hosted models reject outright. When a model complains we
# retry without it rather than forcing the user to know which knobs each model
# supports (reasoning models, for instance, refuse temperature and top_p).
_DROPPABLE_PARAMS = ("temperature", "top_p", "stream_options")


class InferenceClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        read_timeout: float | None = None,
        headers: dict[str, str] | None = None,
        label: str = "The service",
        usage_accounting: bool = False,
    ):
        self.base_url = base_url
        # Named in credential errors, because "DigitalOcean rejected the key" is
        # a confusing thing to read when the key was OpenRouter's.
        self.label = label
        # Ask for the charged amount per reply where the gateway reports it.
        self.usage_accounting = usage_accounting
        self.read_timeout = read_timeout if read_timeout is not None else stall_timeout()
        # A per-read timeout rather than one deadline for the whole request: a
        # long reply that keeps streaming never trips it, but a model that
        # accepts the request and then sends nothing (router:* has been seen to
        # do this) fails in a couple of minutes instead of hanging for ten.
        # Timeout comes from openai rather than httpx directly: the SDK sits on
        # httpx in 1.x/2.x and httpx2 in 3.x, and the type has to match its own.
        timeout = Timeout(self.read_timeout, connect=15.0, write=30.0, pool=15.0)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2,
            default_headers=headers or None,
        )

    # ---------------------------------------------------------------- models

    def list_models(self) -> list[dict[str, Any]]:
        """GET /v1/models -> raw model records, as reported by the service."""
        try:
            page = self._client.models.list()
        except AuthenticationError as exc:
            raise InferenceError(self._rejected()) from exc
        except APITimeoutError as exc:
            raise InferenceError(f"Timed out listing models after {self.read_timeout:.0f}s.") from exc
        except APIConnectionError as exc:
            raise InferenceError(f"Could not reach {self.base_url}: {exc}") from exc
        except APIStatusError as exc:
            raise InferenceError(_readable(exc)) from exc

        models: list[dict[str, Any]] = []
        for model in page.data:
            record = model.model_dump() if hasattr(model, "model_dump") else dict(model)
            if record.get("id"):
                models.append(record)
        return models

    # ------------------------------------------------------------------ chat

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        on_note: Callable[[str], None] | None = None,
    ) -> Iterator[Chunk]:
        """Stream a chat completion, yielding Chunks as they arrive."""
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(self._accounting())

        stream = self._create(params, on_note=on_note)

        # A stream can also stall part-way through, once the response is already
        # open, so the iteration needs the same treatment as opening it.
        try:
            for event in stream:
                served = getattr(event, "model", "") or ""
                usage = getattr(event, "usage", None)
                if usage is not None:
                    raw = _raw(usage)
                    yield Chunk(usage=_usage_dict(raw), cost=_billed(raw), model=served,
                                id=getattr(event, "id", "") or "")

                for choice in getattr(event, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    # Reasoning models surface their scratchpad under one of
                    # these names depending on the upstream provider.
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                    text = getattr(delta, "content", None)
                    if reasoning or text:
                        yield Chunk(
                            text=text or "",
                            reasoning=reasoning if isinstance(reasoning, str) else "",
                            model=served,
                        )
        except APIStatusError as exc:
            raise InferenceError(_readable(exc)) from exc
        except Exception as exc:
            # Mid-stream failures arrive raw from the underlying http library
            # (httpx.ReadTimeout / httpx2.ReadTimeout) because the SDK only
            # wraps timeouts on the initial request, not while iterating. The
            # library differs across SDK majors, so classify by type name
            # instead of importing one. KeyboardInterrupt and GeneratorExit are
            # BaseException, so a user cancelling still passes straight through.
            raise self._stream_failure(exc) from exc

    def _rejected(self) -> str:
        return (
            f"{self.label} rejected the credential (401). Check that the key is "
            "active and copied in full."
        )

    def _stream_failure(self, exc: Exception) -> InferenceError:
        name = type(exc).__name__
        if "Timeout" in name:
            return InferenceError(
                f"The stream stalled for {self.read_timeout:.0f}s and timed out. "
                "Any text above is what arrived; raise DO_INFERENCE_TIMEOUT to wait longer."
            )
        return InferenceError(f"The stream failed ({name}): {exc}")

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_note: Callable[[str], None] | None = None,
    ) -> Completion:
        """One whole reply, not streamed - for agent loops that parse the text."""
        params: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(self._accounting())

        response = self._create(params, on_note=on_note)
        choice = (getattr(response, "choices", None) or [None])[0]
        message = getattr(choice, "message", None)
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        usage = getattr(response, "usage", None)
        raw = _raw(usage) if usage is not None else {}
        return Completion(
            text=(getattr(message, "content", None) or "") if message else "",
            reasoning=reasoning if isinstance(reasoning, str) else "",
            usage=_usage_dict(raw) if raw else {},
            model=getattr(response, "model", "") or model,
            finish_reason=(getattr(choice, "finish_reason", "") or "") if choice else "",
            cost=_billed(raw),
            id=getattr(response, "id", "") or "",
        )

    def _accounting(self) -> dict[str, Any]:
        """The body extension that asks the gateway what the reply cost.

        extra_body rather than a named parameter: it is not part of the OpenAI
        schema, so the SDK would refuse it as a keyword and pass it through here.
        """
        return {"extra_body": dict(COST_ACCOUNTING)} if self.usage_accounting else {}

    def _create(self, params: dict[str, Any], on_note: Callable[[str], None] | None):
        """Send the request, retrying once per parameter the model rejects."""
        attempt = dict(params)
        already_fixed: set[str] = set()

        for _ in range(len(_DROPPABLE_PARAMS) + 2):
            try:
                return self._client.chat.completions.create(**attempt)
            except AuthenticationError as exc:
                raise InferenceError(self._rejected()) from exc
            except BadRequestError as exc:
                fix = _diagnose_bad_request(str(exc), attempt, already_fixed)
                if fix is None:
                    raise InferenceError(_readable(exc)) from exc
                already_fixed.add(fix)
                attempt = _apply_fix(fix, attempt)
                if on_note:
                    on_note(_FIX_NOTES[fix])
            except APITimeoutError as exc:
                raise InferenceError(
                    f"{attempt.get('model')} sent nothing for {self.read_timeout:.0f}s and timed out. "
                    "Try another model, or raise DO_INFERENCE_TIMEOUT."
                ) from exc
            except APIConnectionError as exc:
                raise InferenceError(f"Could not reach {self.base_url}: {exc}") from exc
            except APIStatusError as exc:
                raise InferenceError(_readable(exc)) from exc

        raise InferenceError("Gave up after repeatedly adjusting request parameters.")


_FIX_NOTES = {
    "drop:temperature": "this model does not accept temperature - sent without it",
    "drop:top_p": "this model does not accept top_p - sent without it",
    "drop:stream_options": "this model does not report token usage while streaming",
    "drop:usage": "this gateway does not report what a reply cost - "
                  "the cost line will be worked out from published rates",
    "rename:max_tokens": "this model wants max_completion_tokens - renamed automatically",
}


def _diagnose_bad_request(message: str, params: dict[str, Any], already_fixed: set[str]) -> str | None:
    """Pick a parameter to drop or rename based on the model's complaint."""
    lowered = message.lower()

    if "max_completion_tokens" in lowered and "max_tokens" in params:
        candidate = "rename:max_tokens"
        if candidate not in already_fixed:
            return candidate

    # The cost report is an extension, so a gateway that never heard of it is
    # entitled to refuse the request outright. Losing the exact figure is worth a
    # retry; losing the reply is not.
    if "usage" in lowered and params.get("extra_body", {}).get("usage") is not None:
        candidate = "drop:usage"
        if candidate not in already_fixed:
            return candidate

    for name in _DROPPABLE_PARAMS:
        if name in lowered and name in params:
            candidate = f"drop:{name}"
            if candidate not in already_fixed:
                return candidate
    return None


def _apply_fix(fix: str, params: dict[str, Any]) -> dict[str, Any]:
    updated = dict(params)
    action, name = fix.split(":", 1)
    if fix == "drop:usage":
        extra = {key: value for key, value in updated.get("extra_body", {}).items()
                 if key != "usage"}
        updated = {key: value for key, value in updated.items() if key != "extra_body"}
        if extra:
            updated["extra_body"] = extra
    elif action == "drop":
        updated.pop(name, None)
    elif action == "rename" and name == "max_tokens":
        updated["max_completion_tokens"] = updated.pop("max_tokens")
    return updated


def _raw(usage: Any) -> dict[str, Any]:
    """The usage object as a plain dict, however the SDK version models it."""
    return usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)


def _billed(raw: dict[str, Any]) -> float | None:
    """What the gateway charged for the reply, or None if it did not say.

    Zero is an answer and not a missing one: free models and a fully cached
    prompt both really do cost nothing, and reporting an estimate over the
    gateway's own zero would put a number on the bill that is not on it.
    """
    if not isinstance(raw, dict) or raw.get("cost") is None:
        return None
    try:
        cost = float(raw["cost"])
    except (TypeError, ValueError):
        return None
    return cost if cost >= 0 else None


def _usage_dict(raw: dict[str, Any]) -> dict[str, int]:
    details = raw.get("prompt_tokens_details")
    cached = 0
    if isinstance(details, dict):
        # Reported by some providers when part of the prompt was served from a
        # cache. Those tokens are billed at a lower rate, so a cost worked out
        # from prompt_tokens alone is an upper bound whenever this is non-zero.
        cached = int(details.get("cached_tokens") or 0)
    return {
        "prompt_tokens": int(raw.get("prompt_tokens") or 0),
        "completion_tokens": int(raw.get("completion_tokens") or 0),
        "total_tokens": int(raw.get("total_tokens") or 0),
        "cached_tokens": cached,
    }


def _readable(exc: APIStatusError) -> str:
    """Turn a verbose SDK error into one line the user can act on."""
    detail = ""
    try:
        body = exc.response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif isinstance(error, str):
                detail = error
            detail = detail or str(body.get("message") or "")
    except Exception:
        detail = ""
    if not detail:
        detail = (getattr(exc, "message", "") or str(exc)).strip()
    status = getattr(exc, "status_code", None)
    return f"API error {status}: {detail}" if status else f"API error: {detail}"
