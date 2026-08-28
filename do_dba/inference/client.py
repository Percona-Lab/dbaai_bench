"""Thin wrapper over the OpenAI SDK, pointed at whichever gateway is in use.

DigitalOcean Serverless Inference and OpenRouter both expose the OpenAI API, so
the only per-gateway details here are the base URL, any extra headers, the name to
blame in an error message, and whether the gateway will say what a reply cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from json import JSONDecodeError
from typing import Any, Callable, Iterator

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
    Timeout,
)

from .config import DEFAULT_BASE_URL, rate_limit_wait, stall_timeout


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

# How much thinking to ask for. The gateway maps one word onto whatever the
# upstream model wants - a token budget for Anthropic, a reasoning_effort for
# OpenAI - which is why an effort travels where a per-provider knob would not.
# Nothing is sent unless an effort is asked for: a model that thinks by default
# goes on thinking, and one that does not is not made to pay for it.
EFFORTS = ("low", "medium", "high")


def reasoning_body(effort: str | None) -> dict[str, Any]:
    """The body extension that asks a model to think, or nothing."""
    return {"reasoning": {"effort": effort}} if effort else {}


# Parameters that some hosted models reject outright. When a model complains we
# retry without it rather than forcing the user to know which knobs each model
# supports (reasoning models, for instance, refuse temperature and top_p).
_DROPPABLE_PARAMS = ("temperature", "top_p", "stream_options")

# The same, for the two body extensions. They travel inside extra_body rather
# than at the top level, so dropping one edits that dict and not the request.
_DROPPABLE_EXTRAS = ("usage", "reasoning")

# A self-hosted server can put the model away between two steps of a run: LM Studio
# unloads on an idle timer, and the next request comes back 400 "Model unloaded."
# That is a pause rather than a refusal - the weights are still on disk and the
# request that follows loads them again - so it is worth asking once more instead of
# ending a run halfway through installing a database, which is what a recorded run
# did at step 4 while the model thought about step 3. Asking again costs one request
# on a server that has just-in-time loading turned off, and the run then fails as it
# would have anyway.
_UNLOADED_MARKS = ("model unloaded", "no models loaded", "model not loaded")
_RELOAD_WAIT = 2.0  # long enough for the unload to finish; the load itself is the server's wait

# A 429 is a wait, not a refusal, and it is the one failure that ends a run for a
# reason that has nothing to do with the run: an OpenRouter cell died at step 2
# with `API error 429: Provider returned error`, having spent $0.007 of a $3.50
# cap and one of 120 steps, on two servers that were already paid for. The SDK's
# own max_retries covers a 429 but only immediately - three requests inside a
# second, against a limit measured in tens of them - so the waiting is done here.
#
# Growing, because the first 429 of a minute-long window and the last are the
# same status code and want different patience. A gateway that sends Retry-After
# overrules the schedule; most send nothing.
#
# How many waits there may be is not a number of its own: the budget is the whole
# answer, and a count beside it would silently cap a budget somebody had raised on
# the advice of the message below. One second is the floor on a single wait, so a
# budget can pay for at most one wait per second of it - which is what bounds the
# loop in _create without bounding the patience.
_RATE_LIMIT_BACKOFF = (5.0, 15.0, 30.0, 60.0)
_MIN_RATE_LIMIT_PAUSE = 1.0

# A 200 whose body is not the JSON the API promises. Seen from OpenRouter in the
# middle of a run: the body was 3.4 kB of newlines - the keep-alive padding a
# queued request is sent while it waits for a provider - and nothing after it.
# json.loads raises inside the SDK's response parsing, and JSONDecodeError is a
# ValueError rather than an APIError, so none of the clauses below caught it and
# a two-server cell ended in a traceback at step 3 of 100.
#
# The SDK's own retries do not cover this either: they happen around the request,
# and the parse is after it. So this asks again, which is the right answer - the
# request produced no reply, so there is nothing to salvage and nothing to stay
# consistent with. Short waits: unlike a 429 the gateway is not asking for time,
# it just answered badly, and a padded body means the request was already queued
# for a while.
_GARBLED_BACKOFF = (2.0, 5.0)
_MAX_GARBLED_RETRIES = 2


class InferenceClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        read_timeout: float | None = None,
        headers: dict[str, str] | None = None,
        label: str = "The service",
        usage_accounting: bool = False,
        key_help: str = "",
        rate_limit_budget: float | None = None,
        max_retries: int = 2,
    ):
        self.base_url = base_url
        # Named in credential errors, because "DigitalOcean rejected the key" is
        # a confusing thing to read when the key was OpenRouter's.
        self.label = label
        # What to do about a 401 on this gateway, printed with the rejection. It
        # matters most where no key was sent at all: a self-hosted server that
        # turns out to want one has to say which variable to put it in.
        self.key_help = key_help
        # Ask for the charged amount per reply where the gateway reports it.
        self.usage_accounting = usage_accounting
        # Per request, not per run: a cell that is rate-limited at every step is
        # rate-limited, and the step limit is what stops it. Never negative, which
        # is the same as zero in effect and would read as a budget in the failure.
        self.rate_limit_budget = max(rate_limit_budget if rate_limit_budget is not None
                                     else rate_limit_wait(), 0.0)
        self.read_timeout = read_timeout if read_timeout is not None else stall_timeout()
        # A per-read timeout rather than one deadline for the whole request: a
        # long reply that keeps streaming never trips it, but a model that
        # accepts the request and then sends nothing (router:* has been seen to
        # do this) fails in a couple of minutes instead of hanging for ten.
        # Timeout comes from openai rather than httpx directly: the SDK sits on
        # httpx in 1.x/2.x and httpx2 in 3.x, and the type has to match its own.
        timeout = Timeout(self.read_timeout, connect=15.0, write=30.0, pool=15.0)
        # The SDK's own retries cover a dropped connection and a 5xx, which is
        # worth keeping; they also fire on a 429, and there is no way to ask for
        # one without the other. So by the time the 429 clause below runs, the
        # quick attempts have already been made and failed - which is why the
        # wait schedule starts at seconds rather than milliseconds. Settable so a
        # test can watch this class's waiting on its own.
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
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
        effort: str | None = None,
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
        params.update(self._extensions(effort))

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
        rejection = (
            f"{self.label} rejected the credential (401). Check that the key is "
            "active and copied in full."
        )
        return f"{rejection}\n\n{self.key_help}" if self.key_help else rejection

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
        effort: str | None = None,
        on_note: Callable[[str], None] | None = None,
    ) -> Completion:
        """One whole reply, not streamed - for agent loops that parse the text."""
        params: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(self._extensions(effort))

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

    def _extensions(self, effort: str | None = None) -> dict[str, Any]:
        """The body fields that are gateway extensions, not OpenAI parameters.

        extra_body rather than named parameters: they are not part of the OpenAI
        schema, so the SDK would refuse them as keywords and pass them through
        here. Both in one dict, because a second `extra_body` would replace the
        first rather than join it - a run that asked for thinking would stop
        asking what it cost.
        """
        extra: dict[str, Any] = {}
        if self.usage_accounting:
            extra.update(COST_ACCOUNTING)
        extra.update(reasoning_body(effort))
        return {"extra_body": extra} if extra else {}

    def _create(self, params: dict[str, Any], on_note: Callable[[str], None] | None):
        """Send the request, retrying once per parameter the model rejects.

        And once for a model the server unloaded between steps - see _UNLOADED_MARKS -
        as often as the budget allows for a rate limit, which is a wait rather than
        an answer (see _RATE_LIMIT_BACKOFF), and a couple of times for a reply that
        was not a reply (see _GARBLED_BACKOFF).
        """
        attempt = dict(params)
        already_fixed: set[str] = set()
        waits, waited = 0, 0.0
        garbled = 0

        # One attempt per fix there could be, one for a model the server had put
        # away, and one more to send the request that finally has none left: a model
        # that refuses everything droppable still gets asked the question. Rate
        # limits and unreadable replies get their own allowances on top, so waiting
        # out a busy minute does not spend the budget for finding a parameter the
        # model will accept.
        for _ in range(len(_DROPPABLE_PARAMS) + len(_DROPPABLE_EXTRAS) + 3
                       + self._rate_limit_attempts + _MAX_GARBLED_RETRIES):
            try:
                return self._client.chat.completions.create(**attempt)
            except AuthenticationError as exc:
                raise InferenceError(self._rejected()) from exc
            except RateLimitError as exc:
                # Before APIStatusError, which is its parent: order decides which
                # clause a 429 lands in.
                pause = self._rate_limit_pause(exc, waits, waited)
                if pause is None:
                    raise InferenceError(
                        _rate_limited(exc, attempt.get("model"), waits, waited,
                                      self.rate_limit_budget)
                    ) from exc
                waits, waited = waits + 1, waited + pause
                if on_note:
                    on_note(f"rate-limited by the gateway - waiting {pause:.0f}s "
                            f"and asking again (attempt {waits + 1})")
                time.sleep(pause)
            except BadRequestError as exc:
                message = str(exc)
                if _unloaded(message) and "wait:unloaded" not in already_fixed:
                    already_fixed.add("wait:unloaded")
                    if on_note:
                        on_note(_FIX_NOTES["wait:unloaded"])
                    time.sleep(_RELOAD_WAIT)
                    continue  # the same request, to a server that has to load it first
                fix = _diagnose_bad_request(message, attempt, already_fixed)
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
            except (APIResponseValidationError, JSONDecodeError) as exc:
                # A 200 that did not carry a reply. Last, because it is the only
                # clause here that catches something which is not an openai error:
                # anything the SDK recognises should have been classified above.
                if garbled >= _MAX_GARBLED_RETRIES:
                    raise InferenceError(
                        _garbled(exc, attempt.get("model"), self.label, garbled)
                    ) from exc
                pause = _GARBLED_BACKOFF[min(garbled, len(_GARBLED_BACKOFF) - 1)]
                garbled += 1
                if on_note:
                    on_note(f"the gateway sent {_bad_body(exc)} instead of a reply - "
                            f"waiting {pause:.0f}s and asking again (attempt {garbled + 1})")
                time.sleep(pause)

        raise InferenceError("Gave up after repeatedly adjusting request parameters.")

    @property
    def _rate_limit_attempts(self) -> int:
        """How many 429s this budget could pay for, plus the one it cannot.

        Every wait costs at least _MIN_RATE_LIMIT_PAUSE of the budget, so this
        bounds the loop in _create without being a second opinion on how long a
        request may wait: the budget decides that, and this follows from it.
        """
        return int(self.rate_limit_budget // _MIN_RATE_LIMIT_PAUSE) + 1

    def _rate_limit_pause(self, exc: RateLimitError, waits: int, waited: float) -> float | None:
        """How long to wait before asking again, or None to give up now.

        None rather than a clamped wait when the gateway asks for longer than the
        budget: waiting 40 of the 60 seconds it said it needs and asking anyway
        earns a second 429, and reporting "waited 120s" for a limit that wanted
        ten minutes reads as patience that would have worked.
        """
        remaining = self.rate_limit_budget - waited
        if remaining <= 0:
            return None
        pause = _wanted_pause(exc, waits)
        return pause if pause <= remaining else None


_FIX_NOTES = {
    "drop:temperature": "this model does not accept temperature - sent without it",
    "drop:top_p": "this model does not accept top_p - sent without it",
    "drop:stream_options": "this model does not report token usage while streaming",
    "drop:usage": "this gateway does not report what a reply cost - "
                  "the cost line will be worked out from published rates",
    "drop:reasoning": "this model cannot be asked to think harder - sent without an effort",
    "rename:max_tokens": "this model wants max_completion_tokens - renamed automatically",
    "wait:unloaded": "the server had put the model away - asking again, which loads it",
}


def _unloaded(message: str) -> bool:
    """Whether a 400 says the server no longer has the model in memory."""
    lowered = message.lower()
    return any(mark in lowered for mark in _UNLOADED_MARKS)


def _diagnose_bad_request(message: str, params: dict[str, Any], already_fixed: set[str]) -> str | None:
    """Pick a parameter to drop or rename based on the model's complaint."""
    lowered = message.lower()

    if "max_completion_tokens" in lowered and "max_tokens" in params:
        candidate = "rename:max_tokens"
        if candidate not in already_fixed:
            return candidate

    # Both extensions are extensions, so a gateway that never heard of one is
    # entitled to refuse the request outright. Losing the exact cost figure, or
    # the thinking, is worth a retry; losing the reply is not.
    for name in _DROPPABLE_EXTRAS:
        if name in lowered and params.get("extra_body", {}).get(name) is not None:
            candidate = f"drop:{name}"
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
    if action == "drop" and name in _DROPPABLE_EXTRAS:
        # The other extension stays: a gateway that refuses one has said nothing
        # about the other.
        extra = {key: value for key, value in updated.get("extra_body", {}).items()
                 if key != name}
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


def _retry_after(exc: APIStatusError) -> float | None:
    """Seconds the gateway asked to be left alone for, if it said.

    Both spellings, because they come from different houses: `retry-after` is the
    HTTP header and is whole seconds or a date, `retry-after-ms` is what several
    gateways send instead when the wait is sub-second. A date is honoured as a
    duration from now rather than an instant, because the two clocks need not
    agree and only the difference matters.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    raw_ms = headers.get("retry-after-ms") or ""
    try:
        if raw_ms.strip():
            return max(float(raw_ms) / 1000.0, 0.0)
    except (AttributeError, ValueError):
        pass
    raw = headers.get("retry-after") or ""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return max(float(raw), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    now = datetime.now(timezone.utc) if when.tzinfo else datetime.now()
    return max((when - now).total_seconds(), 0.0)


def _wanted_pause(exc: RateLimitError, waits: int) -> float:
    """How long the next wait would be: the gateway's figure, else our schedule.

    Shared by the decision and the message about it, so a run that gives up can
    say what it would have had to wait rather than just that it did not.
    """
    asked = _retry_after(exc)
    if asked is None:
        asked = _RATE_LIMIT_BACKOFF[min(waits, len(_RATE_LIMIT_BACKOFF) - 1)]
    # A gateway that says "retry immediately" is taken at its word, but not to the
    # point of a hot loop: one second is the floor.
    return max(asked, _MIN_RATE_LIMIT_PAUSE)


def _rate_limited(exc: RateLimitError, model: Any, waits: int,
                  waited: float, budget: float) -> str:
    """The one line that ends a run the gateway would not serve.

    It says what was waited against what was allowed, because "API error 429"
    alone left a reader unable to tell a key with no quota from a busy minute -
    and those two want opposite actions. Never "no time left" when there was
    some: a run given ten seconds and asked for sixty was not out of patience,
    it was never going to have enough.

    The budget is the only reason a request gives up, so this is the only line: it
    always names the wait that would have been needed and the knob that pays for
    it, and raising that knob always buys the wait. Both spellings of the knob,
    because the flag overrides the variable: a run driven by --rate-limit-wait
    would not notice DO_INFERENCE_RATE_LIMIT_WAIT being raised on its advice.
    """
    detail = _readable(exc)
    spent = (f"after {waits} wait(s) totalling {waited:.0f}s of a {budget:.0f}s budget"
             if waits else f"and the rate-limit budget is {budget:.0f}s")
    return (f"{detail} ({model} wants another {_wanted_pause(exc, waits):.0f}s, "
            f"{spent} - raise --rate-limit-wait, or "
            f"$DO_INFERENCE_RATE_LIMIT_WAIT, to wait longer)")


def _bad_body(exc: Exception) -> str:
    """What the gateway sent instead of a reply, in a few words.

    JSONDecodeError carries the text it was handed, so the common case can be
    named for what it is. "not JSON" would leave a reader guessing at a bad base
    URL or a captive proxy; "3388 bytes of whitespace" says the request reached
    the gateway, was queued behind the padding it sends while waiting, and then
    lost whatever was supposed to follow.
    """
    body = getattr(exc, "doc", None)
    if not isinstance(body, str):
        first = (str(exc).strip().splitlines() or [""])[0]
        return f"a reply it could not read ({first[:120]})" if first else "an unreadable reply"
    if not body.strip():
        return f"{len(body)} bytes of whitespace and nothing else"
    return f"a body that is not JSON ({body.strip()[:120]})"


def _garbled(exc: Exception, model: Any, label: str, retries: int) -> str:
    """The one line that ends a run the gateway answered but did not reply to.

    Names the count, because asking three times and getting padding three times
    is a gateway that is not serving this model at all - which is a different
    thing to do about it than one bad response.
    """
    return (f"{label} answered {model} with {_bad_body(exc)}, "
            f"{retries + 1} times. Nothing was generated to keep.")


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
