from http.server import BaseHTTPRequestHandler, HTTPServer
import time
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self,*a): pass
    def do_POST(self):
        # Accept, open an SSE response, then never send an event.
        self.send_response(200)
        self.send_header("Content-Type","text/event-stream")
        self.send_header("Transfer-Encoding","chunked")
        self.end_headers()
        time.sleep(300)
HTTPServer(("127.0.0.1",8896),H).serve_forever()
