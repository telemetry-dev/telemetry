"""Local ingest server used by the wire-level tests: it captures OTLP posts and answers prompt
fetches."""

from __future__ import annotations

import gzip
import json
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
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
    # The HTTP status of each response. To make an ingest outage, set an error status such as 503.
    status: int = 200
    # The (status, JSON body) replies to GET requests, in order; the last one repeats.
    replies: list[tuple[int, object]] = field(default_factory=lambda: [])
    # Seconds that the server waits before it answers a GET request.
    delay: float = 0
    # The location header of GET replies, for a redirect status.
    location: str | None = None

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
                    # self.path collapses a leading "//" (gh-87389); keep the raw target.
                    path=self.requestline.split(" ")[1],
                    headers={key.lower(): value for key, value in self.headers.items()},
                    body=body,
                )
            )
            self.send_response(capture.status)
            self.send_header("content-type", "application/x-protobuf")
            self.send_header("content-length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            captured.append(
                CapturedRequest(
                    path=self.requestline.split(" ")[1],
                    headers={key.lower(): value for key, value in self.headers.items()},
                    body=b"",
                )
            )
            time.sleep(capture.delay)
            status, reply = (
                capture.replies[0] if len(capture.replies) == 1 else capture.replies.pop(0)
            )
            body = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            if capture.location is not None:
                self.send_header("location", capture.location)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    host, port = server.server_address[0], server.server_address[1]
    capture = CaptureServer(url=f"http://{host}:{port}", requests=captured)
    thread.start()
    try:
        yield capture
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
