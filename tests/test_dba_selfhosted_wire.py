"""The self-hosted path over a real HTTP connection, against a local stub.

What only shows up on the wire, and is what a server somebody runs themselves
does differently from a hosted gateway: an endpoint given as a bare host still
reaching /v1, a request going out with no credential of the operator's in it, a
model list with no prices surviving the catalog, the context length arriving from
a second endpoint outside /v1 - and the listing still working when that endpoint
is not there, because most self-hosted servers do not have it.

The stub answers like LM Studio, which is what the Mac Studio in the default
endpoint runs: `owned_by` names the machine, `usage` comes back without a `cost`,
unknown request fields are ignored rather than refused, and /api/v0/models says
what /v1/models cannot.
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

RUNS = PROJECT / "_scratch" / "dba_selfhosted_runs"
MODEL = "qwen/qwen3.8-27b"
STEP = "ACTION: done\nVERIFY: systemctl is-active ssh\nSUMMARY: nothing needed doing"

MODELS = [
    {"id": "openai/gpt-oss-20b", "object": "model", "owned_by": "organization_owner"},
    {"id": MODEL, "object": "model", "owned_by": "organization_owner"},
    {"id": "text-embedding-nomic-embed-text-v1.5", "object": "model",
     "owned_by": "organization_owner"},
]

# LM Studio's own listing, which is where a context length can come from at all.
# The loaded model is deliberately loaded short of what its weights allow - 32K of
# a possible 256K - because that is the number a request will actually hit, and a
# harness that reported the larger one would be describing the file on disk.
DETAILS = [
    {"id": MODEL, "type": "vlm", "state": "loaded", "publisher": "qwen",
     "quantization": "8bit", "max_context_length": 262144, "loaded_context_length": 32768},
    {"id": "openai/gpt-oss-20b", "type": "llm", "state": "not-loaded",
     "max_context_length": 131072},
    # Named after its architecture rather than its job, so only `type` says it is
    # not a chat model. The one below it is: the same box, a different quantisation
    # of the same weights, kept on disk and not offered on /v1/models - so it must
    # not turn up in the listing either.
    {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings",
     "state": "not-loaded", "max_context_length": 2048},
    {"id": "qwen/qwen3.8-27b-q4", "type": "vlm", "state": "not-loaded",
     "max_context_length": 262144},
]

REQUESTS: list[dict] = []
# Flipped partway through: a server without LM Studio's endpoint answers 404 there,
# which has to cost the listing nothing but the two columns it would have filled.
SERVE_DETAILS = [True]
# A message to refuse the next completion with, or "" to answer it. LM Studio
# unloads on an idle timer, and a recorded live run against the Mac Studio was ended
# by exactly that at step 4 - so a 400 saying so has to be sent again rather than
# the run being given up on, and a 400 saying anything else must not be.
REFUSE_ONCE = [""]


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
        # Deliberately strict: the two paths LM Studio serves and nothing else. An
        # endpoint given as a bare host has to be completed by the harness, and a
        # stub that answered anything would not prove it was. /api/v0/models sits
        # beside /v1 rather than under it, which is the other thing being proved.
        if self.path == "/v1/models":
            self._send({"data": MODELS, "object": "list"})
        elif self.path == "/api/v0/models" and SERVE_DETAILS[0]:
            self._send({"data": DETAILS, "object": "list"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode()
        self._record(body)
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if REFUSE_ONCE[0]:
            # Shaped as the Mac Studio shapes it: a 400 with a message and nothing
            # else to go on - no code, no retry-after, nothing to tell it apart from
            # a request the model refuses on its merits except the words.
            raw = json.dumps({"error": REFUSE_ONCE[0]}).encode()
            REFUSE_ONCE[0] = ""
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        sent = json.loads(body or "{}")
        asked = json.dumps(sent.get("messages") or [])
        # As LM Studio does it: token counts, and no word about money, because
        # nobody is being charged. A reasoning model on such a server puts its
        # scratchpad in reasoning_content, which the client reads.
        self._send({
            "id": "chatcmpl-local-1",
            "model": MODEL,
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": "pong" if "ping" in asked else STEP,
                                     "reasoning_content": ""},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240},
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
    host = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"stub on {host} (given to the harness without /v1)", flush=True)

    env = dict(os.environ)
    env["DBA_PROVIDER"] = "selfhosted"
    # The bare host, as the server's own interface shows it.
    env["DBA_SELFHOSTED_BASE_URL"] = host
    # Whatever is in .env must not be what makes this pass, in either direction:
    # no key of the operator's should be needed, and none should be sent.
    for name in ("DBA_SELFHOSTED_KEY", "SELFHOSTED_API_KEY"):
        env.pop(name, None)

    try:
        # ---------------------------------------------- the CLI, end to end
        listing = subprocess.run(
            [sys.executable, "dba.py", "--list-models", "--no-color"],
            cwd=PROJECT, env=env, capture_output=True, text=True, timeout=60,
        )
        check(failures, listing.returncode == 0,
              f"--list-models exited {listing.returncode}: {listing.stderr[-500:]}")
        out = listing.stdout
        check(failures, MODEL in out, f"the loaded model was not listed:\n{out}")
        check(failures, "gpt-oss-20b" in out, "the second loaded model was not listed")
        check(failures, "nomic-embed" not in out, "an embedding model was listed as chat")
        # Not "price unpublished": there is no price to publish, and a run on this
        # box is not on anybody's bill.
        check(failures, "no per-token bill" in out and "price unpublished" not in out,
              f"the listing priced a self-hosted server as unknown:\n{out}")

        # What the second endpoint is for. The loaded length wins over the larger
        # maximum, the one model in memory is marked and the cold one is not, and a
        # model only /api/v0/models knows about is not offered - the chat endpoint
        # would not accept it.
        listed = {line.split()[0]: line for line in out.splitlines() if line.strip()}
        check(failures, "32K ctx" in listed.get(MODEL, "") and "262K" not in listed.get(MODEL, ""),
              f"the loaded context length did not reach the listing: {listed.get(MODEL)!r}")
        check(failures, listed.get(MODEL, "").rstrip().endswith("loaded"),
              f"the model in memory was not marked: {listed.get(MODEL)!r}")
        check(failures, "131K ctx" in listed.get("openai/gpt-oss-20b", ""),
              f"a cold model's window was not read: {listed.get('openai/gpt-oss-20b')!r}")
        check(failures, not listed.get("openai/gpt-oss-20b", "").rstrip().endswith("loaded"),
              "a model that is not in memory was marked as loaded")
        check(failures, "qwen3.8-27b-q4" not in out,
              f"a model only the detail endpoint knows about was offered:\n{out}")

        print("listing done", flush=True)
        gets = [r for r in REQUESTS if r["method"] == "GET"]
        check(failures, [r["path"] for r in gets] == ["/v1/models", "/api/v0/models"],
              f"the two listings were not fetched as expected: {[r['path'] for r in gets]}")
        for request in gets:
            sent_auth = request["headers"].get("authorization")
            check(failures, sent_auth == "Bearer self-hosted",
                  f"a server with no keys was sent {sent_auth!r} on {request['path']}")

        # ------------------------------- the same server without that endpoint
        # vLLM, llama.cpp and Ollama have no /api/v0/models. The listing has to
        # come back exactly as it did before this feature existed.
        SERVE_DETAILS[0] = False
        plain = subprocess.run(
            [sys.executable, "dba.py", "--list-models", "--no-color"],
            cwd=PROJECT, env=env, capture_output=True, text=True, timeout=60,
        )
        SERVE_DETAILS[0] = True
        check(failures, plain.returncode == 0,
              f"a 404 on the detail endpoint broke the listing ({plain.returncode}): "
              f"{plain.stderr[-500:]}")
        check(failures, MODEL in plain.stdout and "gpt-oss-20b" in plain.stdout,
              f"the models were lost with the detail endpoint:\n{plain.stdout}")
        check(failures, "ctx" not in plain.stdout and "loaded" not in plain.stdout,
              f"a server that said nothing was reported as having said something:\n{plain.stdout}")

        # Several models loaded and none named: the operator chooses, and the
        # refusal has to say so before anything is opened to a server.
        unnamed = subprocess.run(
            [sys.executable, "dba.py", "--host", "198.51.100.9", "--task", "x", "--no-color"],
            cwd=PROJECT, env=env, capture_output=True, text=True, timeout=60,
        )
        said = unnamed.stdout + unnamed.stderr
        check(failures, unnamed.returncode == 2, f"a model was picked for the operator ({unnamed.returncode})")
        check(failures, "pins no default" in said, f"the refusal was not explained:\n{said[-400:]}")

        print("refusal done", flush=True)
        # ------------------------------------- a completion through the client
        from do_dba.inference import providers
        from do_dba.inference.client import InferenceClient, InferenceError

        os.environ["DBA_SELFHOSTED_BASE_URL"] = host
        for name in ("DBA_SELFHOSTED_KEY", "SELFHOSTED_API_KEY"):
            os.environ.pop(name, None)
        provider = providers.get("local")
        check(failures, provider.base() == f"{host}/v1",
              f"the endpoint was not completed: {provider.base()}")
        client = InferenceClient(api_key=provider.api_key(), base_url=provider.base(),
                                 headers=provider.headers, label=provider.label,
                                 usage_accounting=provider.usage_accounting,
                                 key_help=provider.key_help,
                                 read_timeout=provider.read_timeout())
        check(failures, client.read_timeout >= 900.0,
              f"a cold server was given {client.read_timeout}s to load its weights")
        reply = client.complete(
            model=MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0.2,
        )
        check(failures, reply.text.strip() == "pong", f"the reply did not come back: {reply.text!r}")
        check(failures, reply.usage.get("prompt_tokens") == 1200, "usage was not read")
        # No cost field on this gateway, and inventing one would be worse than
        # the zero the provider already knows about.
        check(failures, reply.cost is None, f"a cost was read where none was sent: {reply.cost!r}")

        posts = [r for r in REQUESTS if r["method"] == "POST"]
        check(failures, [r["path"] for r in posts] == ["/v1/chat/completions"],
              f"the completion went to {[r['path'] for r in posts]}")
        if posts:
            sent = json.loads(posts[0]["body"])
            check(failures, sent.get("model") == MODEL,
                  f"the model id was rewritten: {sent.get('model')!r}")
            # The cost report is OpenRouter's extension and this gateway was never
            # asked for it, so nothing extra should be in the body at all.
            check(failures, "usage" not in sent,
                  "a gateway that reports no costs was asked what the reply cost")

        # A server behind a proxy that does want a key has to say which variable
        # to put it in, since the run got this far having sent none of its own.
        check(failures, "Self-hosted rejected" in client._rejected()
              and "DBA_SELFHOSTED_KEY" in client._rejected(),
              f"a 401 here is not actionable: {client._rejected()}")

        print("completion done", flush=True)
        # ------------------------------ a whole run, on a box that sends no bill
        from do_dba.agent import DBAAgent, Limits
        from do_dba.fleet import Fleet
        from do_dba.inference.pricing import Price, PriceBook
        from do_dba.report import HostInfo, RunRecord
        from do_dba.secrets import SecretStore
        from fake_droplet import FakeDroplet

        droplet = FakeDroplet()
        fleet = Fleet.of(droplet, name="fake.droplet")
        fleet.survey()
        store = SecretStore()
        # What cli.py does for an unmetered provider: every model it serves at
        # zero, so the run reports $0.00 rather than "cost n/a" on every reply.
        prices = PriceBook(prices={model["id"]: Price(0.0, 0.0) for model in MODELS}, warning=None)
        task = "Check that ssh is running."
        record = RunRecord(
            directory=RUNS / "unmetered", task=task,
            hosts=[HostInfo(name=name, label=label, facts=facts)
                   for name, label, facts in fleet.host_lines()],
            model=MODEL, mode="auto", dry_run=False, provider=provider.label,
            metered=provider.metered, redact=store.redact,
        )
        agent = DBAAgent(
            client=client, model=MODEL, fleet=fleet, task=task, record=record, store=store,
            prices=prices, emit=lambda kind, message: None,
            approve=lambda action, detail, reason: True, mode="auto",
            # As cli.py sizes them, from what /api/v0/models said this model was
            # loaded with: 32K, so the cap is an eighth of it rather than the 16K
            # ceiling a large window would allow.
            limits=Limits.for_window(32768, max_steps=3, command_timeout=30.0),
        )
        outcome = agent.run()
        report_text = record.write_report().read_text(encoding="utf-8")

        check(failures, outcome.status == "done", f"the stub run ended {outcome.status}")
        check(failures, outcome.cost == 0.0 and outcome.cost_complete,
              f"a run on your own hardware cost {outcome.cost} and was "
              f"{'complete' if outcome.cost_complete else 'incomplete'}")
        check(failures, record.prompt_tokens == 1200 and record.completion_tokens == 40,
              "tokens are still worth counting on a server that charges nothing")
        check(failures, "$0.00" in report_text and "no per-token bill" in report_text,
              f"the report does not say the run was free:\n{report_text[:600]}")
        check(failures, "published rates" not in report_text,
              "a zero from a self-hosted box was reported as a rate-table estimate")
        run_posts = [r for r in REQUESTS if r["method"] == "POST"][1:]
        check(failures, run_posts and all(json.loads(r["body"]).get("max_tokens") == 4096
                                          for r in run_posts),
              f"the reply cap did not reach the wire: "
              f"{[json.loads(r['body']).get('max_tokens') for r in run_posts]}")
        logged = [json.loads(line) for line in
                  (record.directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
        usage_events = [event for event in logged if event["kind"] == "usage"]
        check(failures, len(usage_events) == 1 and usage_events[0]["cost"] == 0
              and usage_events[0]["reply"] == "chatcmpl-local-1",
              f"the per-reply line is wrong: {usage_events}")

        # ------------------------------ the model put away in the middle of a run
        # A hosted gateway does not do this; a box in an office does, and the run it
        # ends is one that has already installed half a database server. The same
        # request is sent again, which is what makes the server load the weights
        # back, and the operator is told why the step took longer.
        print("unload retry", flush=True)
        notes: list[str] = []
        before = len([r for r in REQUESTS if r["method"] == "POST"])
        REFUSE_ONCE[0] = "Model unloaded."
        recovered = client.complete(
            model=MODEL, messages=[{"role": "user", "content": "ping"}],
            on_note=notes.append,
        )
        after = len([r for r in REQUESTS if r["method"] == "POST"])
        check(failures, recovered.text.strip() == "pong",
              f"the run was given up on when the model was unloaded: {recovered.text!r}")
        check(failures, after - before == 2,
              f"the unloaded model was asked {after - before} times, want 2")
        check(failures, any("put the model away" in note for note in notes),
              f"the operator was not told why the step stalled: {notes}")
        # And a 400 that says something else is still a refusal, asked once: a
        # server that means it must not be pestered, and the run has to end with
        # what it said rather than with a retry that reads as a hang.
        REFUSE_ONCE[0] = "The model does not accept images."
        before = len([r for r in REQUESTS if r["method"] == "POST"])
        try:
            client.complete(model=MODEL, messages=[{"role": "user", "content": "ping"}])
            check(failures, False, "a 400 that was not about loading was retried into a reply")
        except InferenceError as exc:
            check(failures, "does not accept images" in str(exc),
                  f"the refusal did not reach the operator: {exc}")
        check(failures, len([r for r in REQUESTS if r["method"] == "POST"]) - before == 1,
              "a server that refused a request on its merits was asked again")
    finally:
        server.shutdown()

    print("FAILURES" if failures else "all checks passed")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
