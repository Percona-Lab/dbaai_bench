"""The OpenRouter path over a real HTTP connection, against a local stub.

Everything else about OpenRouter support is unit-tested, but the part that only
shows up on the wire - the key going out as a bearer token, the attribution
header, slash-shaped ids surviving the catalog, rates from the gateway's own
model list reaching the cost line, and the request asking what the reply cost so
the run's total is the billed one - is not. This stands up a server that answers
like OpenRouter, points the harness at it with OPENROUTER_BASE_URL, and checks
what actually arrived.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # the suites sit in tests/, the harness above it
# Ahead of anything installed on purpose: the point is to test this tree.
sys.path.insert(0, str(PROJECT))

RUNS = PROJECT / "_scratch" / "dba_wire_runs"
MODEL = "anthropic/claude-sonnet-4.5"
# What the stub says it charged, chosen to be nothing like tokens x published
# rate (1200 in / 40 out at $3/$15 per M is $0.0042), so the assertions can tell
# which of the two figures the harness kept.
CHARGED = 0.0123
STEP = "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: nothing needed doing"

MODELS = [
    {
        "id": "anthropic/claude-sonnet-4.5",
        "context_length": 1000000,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "0.000003", "completion": "0.000015"},
    },
    {
        "id": "openai/gpt-5.1",
        "context_length": 400000,
        "architecture": {"output_modalities": ["text"]},
        "pricing": {"prompt": "0.00000125", "completion": "0.00001"},
    },
    {
        "id": "black-forest-labs/flux-1.1-pro",
        "architecture": {"output_modalities": ["image"]},
        "pricing": {"prompt": "0", "completion": "0"},
    },
]

REQUESTS: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _record(self, body: str = "") -> None:
        REQUESTS.append({
            "path": self.path,
            "method": self.command,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "body": body,
        })

    def _send(self, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self._record()
        if self.path.endswith("/models"):
            self._send({"data": MODELS})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode()
        self._record(body)
        sent = json.loads(body or "{}")
        # As OpenRouter does it: the charged amount is reported only when the
        # request asked for it, so a test that sees a cost has proved the ask
        # travelled rather than that the stub is generous.
        usage = {"prompt_tokens": 1200, "completion_tokens": 40}
        if (sent.get("usage") or {}).get("include"):
            usage["cost"] = CHARGED
        # "ping" is the client-level check; anything else is the agent, which
        # needs a reply in the step protocol rather than a greeting.
        asked = json.dumps(sent.get("messages") or [])
        self._send({
            "id": "gen-1",
            "model": MODEL,
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": "pong" if "ping" in asked else STEP},
                         "finish_reason": "stop"}],
            "usage": usage,
        })

    def log_message(self, *args) -> None:
        pass  # the test prints its own account


def check(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    shutil.rmtree(RUNS, ignore_errors=True)  # yesterday's transcript must not answer today
    failures: list[str] = []
    # Threaded: the client keeps its connection alive, and a single-threaded
    # server would then refuse to serve the next process until it closed.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    base = f"http://127.0.0.1:{server.server_port}/v1"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"stub on {base}", flush=True)

    env = dict(os.environ)
    env["OPENROUTER_API_KEY"] = "sk-or-v1-stub-key"
    env["OPENROUTER_BASE_URL"] = base
    # A real DO key in .env must not be what makes this pass.
    env["DBA_PROVIDER"] = "openrouter"

    try:
        # ---------------------------------------------- the CLI, end to end
        listing = subprocess.run(
            [sys.executable, "dba.py", "--list-models", "--no-color"],
            cwd=PROJECT, env=env, capture_output=True, text=True, timeout=60,
        )
        check(failures, listing.returncode == 0,
              f"--list-models exited {listing.returncode}: {listing.stderr[-500:]}")
        out = listing.stdout
        check(failures, "anthropic/claude-sonnet-4.5" in out, "a slash-shaped id did not survive")
        check(failures, "$3.00/$15.00 per M" in out,
              f"the gateway's own rate did not reach the listing:\n{out}")
        check(failures, "$1.25/$10.00 per M" in out, "a sub-dollar rate converted wrongly")
        check(failures, "flux" not in out, "an image-only model was listed as chat")
        check(failures, "price unpublished" not in out,
              "every stub model has a rate, so nothing should be unpriced")

        print("listing done", flush=True)
        models_calls = [r for r in REQUESTS if r["path"].endswith("/models")]
        check(failures, len(models_calls) == 1, f"expected one /models call, saw {len(models_calls)}")
        if models_calls:
            headers = models_calls[0]["headers"]
            check(failures, headers.get("authorization") == "Bearer sk-or-v1-stub-key",
                  f"the key did not arrive as a bearer token: {headers.get('authorization')!r}")
            check(failures, headers.get("x-title") == "do-dba",
                  f"the attribution header did not arrive: {headers.get('x-title')!r}")

        # A model that only emits images must be refused by name, too.
        refused = subprocess.run(
            [sys.executable, "dba.py", "--host", "198.51.100.9", "--task", "x",
             "-m", "flux", "--no-color"],
            cwd=PROJECT, env=env, capture_output=True, text=True, timeout=60,
        )
        check(failures, refused.returncode == 2, f"a non-chat model was accepted ({refused.returncode})")
        check(failures, "not a chat model" in refused.stdout + refused.stderr,
              f"the refusal was not explained:\n{refused.stdout[-400:]}")

        print("refusal done", flush=True)
        # ------------------------------------- a completion through the client
        from do_dba.inference import providers
        from do_dba.inference.client import InferenceClient

        os.environ.update({"OPENROUTER_API_KEY": env["OPENROUTER_API_KEY"],
                           "OPENROUTER_BASE_URL": base})
        provider = providers.get("or")
        client = InferenceClient(api_key=provider.api_key(), base_url=provider.base(),
                                 headers=provider.headers, label=provider.label,
                                 usage_accounting=provider.usage_accounting)
        reply = client.complete(
            model=MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.2,
            effort="high",
        )
        check(failures, reply.text.strip() == "pong", f"the reply did not come back: {reply.text!r}")
        check(failures, reply.usage.get("prompt_tokens") == 1200, "usage was not read")
        check(failures, reply.cost == CHARGED, f"the charged amount was not read: {reply.cost!r}")
        check(failures, reply.id == "gen-1",
              f"the gateway's id for the reply was not kept: {reply.id!r}")

        posts = [r for r in REQUESTS if r["method"] == "POST"]
        check(failures, len(posts) == 1, f"expected one completion call, saw {len(posts)}")
        if posts:
            check(failures, posts[0]["headers"].get("x-title") == "do-dba",
                  "the attribution header is missing on completions")
            sent = json.loads(posts[0]["body"])
            check(failures, sent.get("model") == MODEL,
                  f"the model id was rewritten: {sent.get('model')!r}")
            # The one line that makes the cost line reconcilable with the bill.
            check(failures, sent.get("usage") == {"include": True},
                  f"the request did not ask what the reply cost: {sent.get('usage')!r}")
            # And the ask for thinking, in the same body: both are extensions
            # travelling through one extra_body, so this is where a second one
            # replacing the first would show up rather than in a unit test.
            check(failures, sent.get("reasoning") == {"effort": "high"},
                  f"the effort did not reach the wire: {sent.get('reasoning')!r}")

        print("completion done", flush=True)
        # ------------------------------- a whole run, priced by the gateway
        # The end of the chain: what OpenRouter says it charged has to be what the
        # report and the transcript say, in place of tokens x published rate.
        from do_dba.agent import DBAAgent, Limits
        from do_dba.fleet import Fleet
        from do_dba.inference.pricing import PriceBook, from_records
        from do_dba.report import HostInfo, RunRecord
        from do_dba.secrets import SecretStore
        from fake_droplet import FakeDroplet

        droplet = FakeDroplet()
        fleet = Fleet.of(droplet, name="fake.droplet")
        fleet.survey()
        store = SecretStore()
        prices = PriceBook(prices=from_records(MODELS), warning=None)
        task = "Check that ssh is running."
        record = RunRecord(
            directory=RUNS / "billed", task=task,
            hosts=[HostInfo(name=name, label=label, facts=facts)
                   for name, label, facts in fleet.host_lines()],
            model=MODEL, mode="auto", dry_run=False, provider="openrouter",
            redact=store.redact,
        )
        agent = DBAAgent(
            client=client, model=MODEL, fleet=fleet, task=task, record=record, store=store,
            prices=prices, emit=lambda kind, message: None,
            approve=lambda action, detail, reason: True, mode="auto",
            # As cli.py sizes them: this model's listing says a 1M window, so the
            # reply cap is the 16K ceiling rather than a share of it.
            limits=Limits.for_window(1_000_000, max_steps=3, command_timeout=30.0),
        )
        outcome = agent.run()
        report_text = record.write_report().read_text(encoding="utf-8")
        estimate = prices.cost(MODEL, record.prompt_tokens, record.completion_tokens)

        check(failures, outcome.status == "done", f"the stub run ended {outcome.status}")
        check(failures, abs(outcome.cost - CHARGED) < 1e-12,
              f"the run's cost is {outcome.cost}, want the charged {CHARGED}")
        check(failures, estimate is not None and abs(estimate - CHARGED) > 1e-9,
              "the published rate agrees with the charged figure, so this proves nothing")
        check(failures, (record.billed_replies, record.estimated_replies) == (1, 0),
              f"the reply was counted as {record.billed_replies} billed, "
              f"{record.estimated_replies} estimated")
        check(failures, "billed by the gateway" in report_text,
              "the report does not say the cost came from the gateway")
        logged = [json.loads(line) for line in
                  (record.directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
        # Per reply, with the gateway's own id for it: that is what makes a run's
        # total checkable line by line against OpenRouter's activity page.
        billed_events = [event for event in logged if event["kind"] == "usage"]
        check(failures, len(billed_events) == 1 and billed_events[0]["reply"] == "gen-1"
              and billed_events[0]["cost_source"] == "gateway"
              and abs(billed_events[0]["cost"] - CHARGED) < 1e-12,
              f"the per-reply cost was not logged as the gateway's: {billed_events}")

        run_posts = [r for r in REQUESTS if r["method"] == "POST"][1:]
        check(failures, run_posts and all(json.loads(r["body"]).get("max_tokens") == 16384
                                          for r in run_posts),
              f"the reply cap did not reach the wire: "
              f"{[json.loads(r['body']).get('max_tokens') for r in run_posts]}")

        # The agent was built without an effort, so its own requests must carry no
        # reasoning block at all: an unasked-for effort would change what every
        # existing run costs, and "off" is the absence of the field rather than a
        # word meaning none.
        agent_posts = [json.loads(r["body"]) for r in REQUESTS if r["method"] == "POST"][1:]
        check(failures, agent_posts and all("reasoning" not in sent for sent in agent_posts),
              f"a run that asked for no effort sent one anyway: {len(agent_posts)} request(s)")

        # A bad key must be reported as OpenRouter's rejection, not DigitalOcean's.
        check(failures, "OpenRouter rejected" in client._rejected(),
              f"credential errors name the wrong service: {client._rejected()}")
    finally:
        server.shutdown()

    print("FAILURES" if failures else "all checks passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
