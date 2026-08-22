"""The inference client over a real HTTP connection, against the local stub.

test_dba_providers.py checks the decisions made before a request goes out and
test_dba_openrouter_wire.py checks what OpenRouter receives. This checks what the
client does with what comes back from a DigitalOcean-shaped service: a whole
non-streamed reply (which is what the harness asks for - it parses replies, it
does not display them), a stream, the reasoning field under either of its two
names, the served model a router alias reports, cached prompt tokens, and the
retry that drops a parameter the model refuses.

    uv run python run_tests.py client

run_tests.py starts mock_do_server.py for this; on its own it needs the stub
already listening on 127.0.0.1:8899.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

from do_dba.inference.client import InferenceClient, InferenceError
from do_dba.inference.pricing import PRICES, PriceBook

BASE = "http://127.0.0.1:8899/v1"


def check(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


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
