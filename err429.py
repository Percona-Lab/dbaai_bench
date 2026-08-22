import json
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self,*a): pass
    def do_POST(self):
        body=json.dumps({"error":{"message":"rate limit exceeded, retry in 30s"}}).encode()
        self.send_response(429); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
HTTPServer(("127.0.0.1",8897),H).serve_forever()
