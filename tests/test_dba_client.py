"""The inference client over a real HTTP connection, against the local stub.

test_dba_providers.py checks the decisions made before a request goes out and
test_dba_openrouter_wire.py checks what OpenRouter receives. This checks what the
client does with what comes back from a DigitalOcean-shaped service: a whole
non-streamed reply (which is what the harness asks for - it parses replies, it
does not display them), a stream, the reasoning field under either of its two
names, the served model a router alias reports, cached prompt tokens, and the
retry that drops a parameter the model refuses - including a model that refuses
two of them, which is the case the retry budget is sized for.

It also checks the two ways a gateway ends a run for a reason that has nothing to
do with the run. A rate limit that clears must not cost the cell its remaining
steps, and one that does not clear must say what it would have taken rather than
imply the waiting was tried and failed. And a 200 whose body is not a reply must
be asked again and then reported as an API failure - never raised as the
JSONDecodeError it arrives as, which is a traceback and a lost run.

    uv run python run_tests.py client

run_tests.py starts mock_do_server.py for this; on its own it needs the stub
already listening on 127.0.0.1:8899.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from do_dba.inference.client import InferenceClient, InferenceError, _retry_after
from do_dba.inference.pricing import PRICES, PriceBook

BASE = "http://127.0.0.1:8899/v1"


def check(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


class FakeLimit:
    """Just enough of a RateLimitError for the header reading: a response with
    headers on it. Lower-case keys because that is how httpx presents them, and
    what the parser asks for."""

    def __init__(self, **headers: str):
        self.response = SimpleNamespace(headers=dict(headers))


def main() -> int:
    failures: list[str] = []
    client = InferenceClient(api_key="test-key", base_url=BASE, label="DigitalOcean")

    # ----------------------------------------------------------------- models
    records = client.list_models()
    ids = [record["id"] for record in records]
    check(failures, "anthropic-claude-opus-5" in ids, f"the model list did not arrive: {ids}")
    check(failures, all("id" in record for record in records), "a record came back without an id")

    # ------------------------------------------- one whole reply, no streaming
    reply = client.complete(
        model="anthropic-claude-opus-5",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.2,
    )
    check(failures, "You said: What is 2+2?" in reply.text, f"the reply text is wrong: {reply.text!r}")
    check(failures, reply.model == "anthropic-claude-opus-5", f"the model was rewritten: {reply.model!r}")
    check(failures, reply.finish_reason == "stop", f"finish_reason lost: {reply.finish_reason!r}")
    check(failures, reply.usage.get("prompt_tokens") == 42, f"usage was not read: {reply.usage}")
    check(failures, reply.usage.get("cached_tokens") == 0,
          "cached_tokens should be zero when the service reports no cache")

    # A reasoning model's scratchpad is kept apart from its answer: the harness
    # parses the answer for a protocol block, and thinking there would break it.
    thought = client.complete(model="reasoner-1", messages=[{"role": "user", "content": "hm"}])
    check(failures, thought.reasoning.strip() == "Let me think about that.",
          f"reasoning did not come through: {thought.reasoning!r}")
    check(failures, "Let me think" not in thought.text, "reasoning leaked into the reply text")

    # ------------------------------------------------------- a parameter fix
    notes: list[str] = []
    picky = client.complete(
        model="picky-model",
        messages=[{"role": "user", "content": "ping"}],
        temperature=0.7,
        on_note=notes.append,
    )
    check(failures, "You said: ping" in picky.text,
          f"the retry without temperature did not succeed: {picky.text!r}")
    check(failures, any("temperature" in note for note in notes),
          f"the operator was not told what was dropped: {notes}")

    # Two refusals in one request, one complaint at a time - which is the case the
    # retry budget has to be big enough for. An effort is not a parameter but a
    # body extension, so dropping it edits extra_body rather than the request.
    fussed: list[str] = []
    both = client.complete(
        model="picky-model",
        messages=[{"role": "user", "content": "ping"}],
        temperature=0.7,
        effort="high",
        on_note=fussed.append,
    )
    check(failures, "You said: ping" in both.text,
          f"a model that refuses two things never got asked: {both.text!r}")
    check(failures, len(fussed) == 2 and any("think harder" in note for note in fussed),
          f"the effort was not dropped and reported: {fussed}")

    # And a model that accepts it is sent it: the stub echoes the prompt, so what
    # proves the ask travelled is that nothing was dropped on the way.
    quiet: list[str] = []
    asked = client.complete(model="reasoner-1", messages=[{"role": "user", "content": "hm"}],
                            effort="low", on_note=quiet.append)
    check(failures, not quiet and asked.reasoning.strip() == "Let me think about that.",
          f"asking a reasoning model to think was not clean: {quiet}")

    # ------------------------------------------------------- a rate limit
    # max_retries=0 so this watches the client's own waiting rather than the SDK's
    # immediate retries, which would swallow the stub's refusals before the code
    # under test saw one. The stub asks for a second, twice.
    slowed: list[str] = []
    patient = InferenceClient(api_key="test-key", base_url=BASE, label="DigitalOcean",
                              rate_limit_budget=10.0, max_retries=0)
    waited_out = patient.complete(
        model="busy-model",
        messages=[{"role": "user", "content": "ping after a wait"}],
        on_note=slowed.append,
    )
    check(failures, "You said: ping after a wait" in waited_out.text,
          f"a limit that cleared still ended the request: {waited_out.text!r}")
    check(failures, len(slowed) == 2 and all("rate-limited" in note for note in slowed),
          f"the waiting was not reported as it happened: {slowed}")
    check(failures, all("waiting 1s" in note for note in slowed),
          f"Retry-After was not honoured over the schedule: {slowed}")

    # A budget of zero is the old behaviour, kept deliberately: fail on the first
    # 429 without waiting. Nothing should be reported as waited.
    impatient: list[str] = []
    try:
        InferenceClient(api_key="test-key", base_url=BASE, rate_limit_budget=0.0,
                        max_retries=0).complete(
            model="overloaded-model",
            messages=[{"role": "user", "content": "ping"}],
            on_note=impatient.append,
        )
        failures.append("a 429 with no budget to wait it out was not an error")
    except InferenceError as exc:
        check(failures, "DO_INFERENCE_RATE_LIMIT_WAIT" in str(exc),
              f"the failure did not say which knob controls it: {exc}")
        check(failures, "budget is 0s" in str(exc),
              f"the failure did not say what it was allowed: {exc}")
        check(failures, not impatient, f"nothing was waited, but it said it had: {impatient}")

    # A budget smaller than the first wait gives up without spending it: waiting 3
    # of the 5 seconds needed earns a second 429 and a report of patience that was
    # never going to work. The message says what it would have taken.
    try:
        InferenceClient(api_key="test-key", base_url=BASE, rate_limit_budget=3.0,
                        max_retries=0).complete(
            model="overloaded-model", messages=[{"role": "user", "content": "ping"}])
        failures.append("a limit longer than the budget was not an error")
    except InferenceError as exc:
        check(failures, "wants another 5s" in str(exc),
              f"the wait it needed was not named: {exc}")
        check(failures, "0 wait" not in str(exc) and "totalling" not in str(exc),
              f"it claimed to have waited when it did not: {exc}")
        check(failures, "--rate-limit-wait" in str(exc),
              f"the failure named no way to buy the patience it wanted: {exc}")

    # The budget is the only thing that ends the waiting. There used to be a cap of
    # six waits beside it, which silently made every budget over ~230s the same as
    # 230s - a run raising the variable the failure told it to raise would wait no
    # longer than before. So a request that has already waited nine times is still
    # allowed a tenth while its budget has room for one.
    patient_enough = InferenceClient(api_key="test-key", base_url=BASE,
                                     rate_limit_budget=600.0, max_retries=0)
    asked_for_thirty = FakeLimit(**{"retry-after": "30"})
    tenth = patient_enough._rate_limit_pause(asked_for_thirty, waits=9, waited=270.0)
    check(failures, tenth == 30.0, f"a budget with 330s left refused a 30s wait: {tenth}")
    spent = patient_enough._rate_limit_pause(asked_for_thirty, waits=9, waited=580.0)
    check(failures, spent is None,
          f"a wait longer than the budget's remainder was taken anyway: {spent}")
    # And the retry loop is sized from the budget, or a budget it cannot spend is a
    # budget that ends the request with "gave up adjusting parameters" instead.
    check(failures, patient_enough._rate_limit_attempts > 600,
          f"600s of budget bought {patient_enough._rate_limit_attempts} attempt(s)")
    check(failures, InferenceClient(api_key="test-key", base_url=BASE,
                                    rate_limit_budget=0.0)._rate_limit_attempts == 1,
          "no budget should still leave the one attempt that fails")

    # Retry-After arrives in three shapes and one of them is a date. A duration
    # from now, not an instant, because the two clocks need not agree.
    check(failures, _retry_after(FakeLimit(**{"retry-after": "30"})) == 30.0,
          "Retry-After in seconds was not read")
    check(failures, _retry_after(FakeLimit(**{"retry-after-ms": "1500"})) == 1.5,
          "retry-after-ms was not read")
    later = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60))
    dated = _retry_after(FakeLimit(**{"retry-after": later}))
    check(failures, dated is not None and 55.0 <= dated <= 61.0,
          f"an HTTP-date Retry-After was not turned into a wait: {dated}")
    check(failures, _retry_after(FakeLimit()) is None,
          "a 429 with no Retry-After should fall back to the schedule")
    check(failures, _retry_after(FakeLimit(**{"retry-after": "soon"})) is None,
          "an unparseable Retry-After should fall back to the schedule")

    # ------------------------------------------------- a reply that is not one
    # A 200 with a body of newlines: OpenRouter's keep-alive padding for a queued
    # request, with the reply that should have followed it missing. It used to end
    # the run in a traceback, because JSONDecodeError is a ValueError and nothing
    # here was looking for one. The stub sends padding twice, then answers.
    padded: list[str] = []
    recovered = InferenceClient(api_key="test-key", base_url=BASE, max_retries=0).complete(
        model="padded-model",
        messages=[{"role": "user", "content": "ping after padding"}],
        on_note=padded.append,
    )
    check(failures, "You said: ping after padding" in recovered.text,
          f"a body of padding was not retried past: {recovered.text!r}")
    check(failures, len(padded) == 2, f"the two bad bodies were not both reported: {padded}")
    check(failures, all("whitespace" in note for note in padded),
          f"padding was not named for what it is: {padded}")

    # A gateway that answers badly every time ends the run - but as an API failure
    # with a readable line, not a traceback, and saying how many times it asked.
    try:
        InferenceClient(api_key="test-key", base_url=BASE, label="OpenRouter",
                        max_retries=0).complete(
            model="garbled-model", messages=[{"role": "user", "content": "ping"}])
        failures.append("a body that never parsed was accepted as a reply")
    except InferenceError as exc:
        check(failures, "OpenRouter" in str(exc) and "garbled-model" in str(exc),
              f"the failure did not say who answered badly: {exc}")
        check(failures, "3 times" in str(exc), f"the failure did not say how often it asked: {exc}")
        check(failures, "no healthy provider" in str(exc),
              f"what the gateway actually sent was not shown: {exc}")

    # ---------------------------------------------------------------- streaming
    chunks = list(client.stream_chat(
        model="anthropic-claude-opus-5",
        messages=[{"role": "user", "content": "CACHED please"}],
        temperature=0.2,
    ))
    streamed = "".join(chunk.text for chunk in chunks)
    check(failures, "You said: CACHED please" in streamed, f"the stream did not assemble: {streamed!r}")
    check(failures, len([c for c in chunks if c.text]) > 1, "the reply arrived in one piece, not a stream")
    usage = next((chunk.usage for chunk in chunks if chunk.usage), {})
    check(failures, usage.get("cached_tokens") == 21,
          f"cached prompt tokens were not read: {usage}")
    reasoned = [c.reasoning for c in client.stream_chat(
        model="reasoner-1", messages=[{"role": "user", "content": "hm"}]) if c.reasoning]
    check(failures, "".join(reasoned).strip() == "Let me think about that.",
          f"streamed reasoning did not come through: {reasoned}")

    # ------------------------------------------------- the model that ran
    # A router alias picks a model per request and bills as that one, so the cost
    # has to follow what the frames say served the request, not what was asked for.
    routed = list(client.stream_chat(
        model="router:general",
        messages=[{"role": "user", "content": "BIG prompt"}],
    ))
    served = next((chunk.model for chunk in routed if chunk.model), "")
    check(failures, served == "llama-4-maverick", f"the served model was not reported: {served!r}")
    prices = PriceBook(prices=dict(PRICES), warning=None)
    check(failures, prices.cost("router:general", 250_000, 4) is None,
          "the alias itself must have no price")
    check(failures, prices.cost(served, 250_000, 4) is not None,
          "the model that actually ran should be priceable")

    big = next((chunk.usage for chunk in routed if chunk.usage), {})
    check(failures, big.get("prompt_tokens") == 250_000, f"the 250K prompt was not reported: {big}")

    # ------------------------------------------------------------ a bad key
    # Named for the gateway that refused it: "DigitalOcean rejected the key" is a
    # confusing thing to read when the credential in use was OpenRouter's.
    try:
        InferenceClient(api_key="expired-key", base_url=BASE, label="DigitalOcean").list_models()
        failures.append("a refused credential was accepted")
    except InferenceError as exc:
        check(failures, "DigitalOcean rejected" in str(exc),
              f"a 401 was not reported as a credential problem: {exc}")

    # An unreachable service must be one line, not a traceback. Port 1 is closed
    # everywhere and refuses immediately, so this does not wait on a timeout.
    try:
        InferenceClient(api_key="test-key", base_url="http://127.0.0.1:1/v1").list_models()
        failures.append("an unreachable service was accepted")
    except InferenceError as exc:
        check(failures, "Could not reach" in str(exc), f"a connection failure read badly: {exc}")

    print("FAILURES" if failures else "all checks passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
