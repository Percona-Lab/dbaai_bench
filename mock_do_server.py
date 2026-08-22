"""Minimal stand-in for https://inference.do-ai.run/v1, for testing the client.

Behaviours exercised:
  * GET  /v1/models            -> catalog listing
  * POST /v1/chat/completions  -> SSE stream + trailing usage chunk
  * the same with stream unset -> one whole JSON reply, which is what the harness
                                  asks for: it parses a reply, it does not show one
  * model "picky-model"        -> 400 on temperature (tests param fallback)
  * model "reasoner-1"         -> emits reasoning_content
  * model "router:general"     -> reports a different served model, as a router does
  * prompt containing "CACHED" -> usage reports prompt_tokens_details.cached_tokens
  * prompt containing "BIG"    -> usage reports a 250K prompt (crosses a price tier)
  * missing/blank auth         -> 401
  * key "expired-key"          -> 401, the way a revoked key is refused
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

MODELS = [
    {"id": "anthropic-claude-opus-5", "object": "model", "owned_by": "anthropic", "context_window": 1000000},
    {"id": "openai-gpt-5.5", "object": "model", "owned_by": "openai", "context_window": 1000000},
    {"id": "anthropic-claude-4.5-sonnet", "object": "model", "owned_by": "anthropic", "context_window": 1000000},
    {"id": "router:general", "object": "model", "owned_by": "digitalocean"},
    {"id": "llama-4-maverick", "object": "model", "owned_by": "meta", "context_window": 128000},
    {"id": "picky-model", "object": "model", "owned_by": "test"},
    {"id": "reasoner-1", "object": "model", "owned_by": "test"},
    {"id": "openai-gpt-image-1", "object": "model", "owned_by": "openai"},
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

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

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

        if model == "picky-model" and "temperature" in body:
            return self._json(
                400,
                {"error": {"message": "Unsupported value: 'temperature' is not supported with this model."}},
            )

        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        prompt = ""
        for message in body.get("messages", []):
            if message.get("role") == "user":
                prompt = str(message.get("content", ""))
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
    HTTPServer(("127.0.0.1", 8899), Handler).serve_forever()
