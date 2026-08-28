"""Minimal stand-in for https://inference.do-ai.run/v1, for testing the client.

Behaviours exercised:
  * GET  /v1/models            -> catalog listing
  * POST /v1/chat/completions  -> SSE stream + trailing usage chunk
  * the same with stream unset -> one whole JSON reply, which is what the harness
                                  asks for: it parses a reply, it does not show one
  * model "picky-model"        -> 400 on temperature and on reasoning, one at a
                                  time, which is what a model that refuses more
                                  than one parameter does (tests param fallback)
  * model "busy-model"         -> 429 with Retry-After twice, then serves the reply,
                                  which is a per-minute limit clearing (tests the wait)
  * model "overloaded-model"   -> 429 every time, and no Retry-After, so the client
                                  has only its own backoff schedule to go on
  * model "padded-model"       -> 200 with a body of newlines twice, then serves the
                                  reply: the keep-alive padding a queued request gets,
                                  with the reply that should have followed it missing
  * model "garbled-model"      -> 200 with prose in the body every time, as a proxy
                                  or an upstream error page does
  * model "reasoner-1"         -> emits reasoning_content
  * model "router:general"     -> reports a different served model, as a router does
  * prompt containing "CACHED" -> usage reports prompt_tokens_details.cached_tokens
  * prompt containing "BIG"    -> usage reports a 250K prompt (crosses a price tier)
  * missing/blank auth         -> 401
  * key "expired-key"          -> 401, the way a revoked key is refused
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = [
    {"id": "anthropic-claude-opus-5", "object": "model", "owned_by": "anthropic", "context_window": 1000000},
    {"id": "openai-gpt-5.5", "object": "model", "owned_by": "openai", "context_window": 1000000},
    {"id": "anthropic-claude-4.5-sonnet", "object": "model", "owned_by": "anthropic", "context_window": 1000000},
    {"id": "router:general", "object": "model", "owned_by": "digitalocean"},
    {"id": "llama-4-maverick", "object": "model", "owned_by": "meta", "context_window": 128000},
    {"id": "picky-model", "object": "model", "owned_by": "test"},
    {"id": "busy-model", "object": "model", "owned_by": "test"},
    {"id": "overloaded-model", "object": "model", "owned_by": "test"},
    {"id": "padded-model", "object": "model", "owned_by": "test"},
    {"id": "garbled-model", "object": "model", "owned_by": "test"},
    {"id": "reasoner-1", "object": "model", "owned_by": "test"},
    {"id": "openai-gpt-image-1", "object": "model", "owned_by": "openai"},
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # How many times a model has been asked each question. Keyed by the prompt as
    # well as the model, because a retry sends the same request again: every test
    # gets its own refusals without needing the server restarted between them.
    calls: dict[tuple[str, str], int] = {}
    calls_lock = threading.Lock()  # the server is threaded; counting is not atomic
    BUSY_REFUSALS = 2
    PADDED_REFUSALS = 2

    def log_message(self, *args):
        pass

    def _seen(self, model, prompt):
        """How many times this exact request has arrived, this one included."""
        with Handler.calls_lock:
            count = Handler.calls.get((model, prompt), 0) + 1
            Handler.calls[(model, prompt)] = count
            return count

    def _authed(self):
        header = self.headers.get("Authorization", "")
        key = header[7:].strip() if header.startswith("Bearer ") else ""
        # "expired-key" is how a well-formed but refused credential is tested: an
        # empty one never gets this far, because a blank header value is rejected
        # by the http library before the request is sent.
        if key and key != "expired-key":
            return True
        self._json(401, {"error": {"message": "invalid model access key"}})
        return False

    def _json(self, status, payload, headers=None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _too_many(self, retry_after=None):
        """A 429, with the gateway's own figure for the wait where it has one."""
        return self._json(
            429,
            {"error": {"message": "Rate limit exceeded: free-models-per-min",
                       "type": "rate_limit_exceeded"}},
            {"Retry-After": str(retry_after)} if retry_after is not None else None,
        )

    def _not_json(self, body):
        """A 200 whose body is not the JSON the API promises.

        Content-Type still says application/json, which is what makes this hard to
        catch: nothing in the response says anything is wrong until the client tries
        to parse it, by which point the SDK's own retries are behind it.
        """
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/") != "/v1/models":
            return self._json(404, {"error": {"message": "not found"}})
        if not self._authed():
            return
        self._json(200, {"object": "list", "data": MODELS})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        if not self._authed():
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        model = body.get("model", "")

        # One complaint per reply, as a real service gives: the client is expected
        # to drop what it was told about and ask again, not to guess the rest.
        if model == "picky-model":
            for unwanted in ("temperature", "reasoning"):
                if unwanted in body:
                    return self._json(400, {"error": {"message":
                        f"Unsupported value: '{unwanted}' is not supported with this model."}})

        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        prompt = ""
        for message in body.get("messages", []):
            if message.get("role") == "user":
                prompt = str(message.get("content", ""))

        # A limit that clears, and one that does not. The second sends no
        # Retry-After, which is the common case: most gateways return a bare 429
        # and leave the caller to guess how long a minute is.
        if model == "overloaded-model":
            return self._too_many()
        if model == "busy-model" and self._seen(model, prompt) <= Handler.BUSY_REFUSALS:
            return self._too_many(retry_after=1)

        # And two ways of answering 200 without answering. The padding is what a
        # queued OpenRouter request is sent while it waits; a body of nothing but
        # padding is what arrives when the reply it was covering for never comes.
        if model == "garbled-model":
            return self._not_json("upstream error: no healthy provider\n")
        if model == "padded-model" and self._seen(model, prompt) <= Handler.PADDED_REFUSALS:
            return self._not_json("\n" * 617)
        # A router alias bills as whatever model it picked, and says so in the
        # "model" field of every frame it sends back.
        served = "llama-4-maverick" if model.startswith("router:") else model
        echo = f"Reply from {served}. You said: {prompt[:60]}"
        pieces = [echo[i : i + 12] for i in range(0, len(echo), 12)]

        if not body.get("stream"):
            return self._json(200, self._whole_reply(model, served, echo, prompt, len(pieces)))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(delta, finish=None):
            frame = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": served,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.flush()

        emit({"role": "assistant", "content": ""})
        if model == "reasoner-1":
            for thought in ["Let me ", "think about ", "that. "]:
                emit({"reasoning_content": thought})
        for piece in pieces:
            emit({"content": piece})
        emit({}, finish="stop")

        if want_usage:
            usage_frame = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": served,
                "choices": [],
                "usage": self._usage(prompt, len(pieces)),
            }
            self.wfile.write(f"data: {json.dumps(usage_frame)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _whole_reply(self, model, served, text, prompt, completion_tokens):
        """A non-streamed completion, shaped like the real endpoint's."""
        message = {"role": "assistant", "content": text}
        if model == "reasoner-1":
            message["reasoning_content"] = "Let me think about that. "
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1,
            "model": served,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": self._usage(prompt, completion_tokens),
        }

    def _usage(self, prompt, completion_tokens):
        prompt_tokens = 250_000 if "BIG" in prompt else 42
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if "CACHED" in prompt:
            usage["prompt_tokens_details"] = {"cached_tokens": prompt_tokens // 2}
        return usage


if __name__ == "__main__":
    # Threaded because these are keep-alive HTTP/1.1 connections and a test may
    # hold more than one: a single-threaded server sits in the first connection's
    # read waiting for a request that is never coming, and a second client's
    # connection is never accepted at all. That deadlock reads as a model that
    # accepted the request and then said nothing, which is a real failure the
    # harness has a timeout for - so it takes the whole timeout to report.
    ThreadingHTTPServer(("127.0.0.1", 8899), Handler).serve_forever()
