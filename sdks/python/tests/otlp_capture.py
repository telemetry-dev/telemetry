"""Local OTLP capture server used by the wire-level tests."""

from __future__ import annotations

import gzip
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class CapturedRequest:
    path: str
    headers: dict[str, str]
    body: bytes

    def decompressed(self) -> bytes:
        if self.headers.get("content-encoding") == "gzip":
            return gzip.decompress(self.body)
        return self.body


@dataclass
class CaptureServer:
    url: str
    requests: list[CapturedRequest]

    def requests_for(self, path: str) -> list[CapturedRequest]:
        return [request for request in self.requests if request.path == path]


@contextmanager
def otlp_capture_server() -> Generator[CaptureServer]:
    captured: list[CapturedRequest] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length)
            captured.append(
                CapturedRequest(
                    path=self.path,
                    headers={key.lower(): value for key, value in self.headers.items()},
                    body=body,
                )
            )
            self.send_response(200)
            self.send_header("content-type", "application/x-protobuf")
            self.send_header("content-length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield CaptureServer(url=f"http://{host}:{port}", requests=captured)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
