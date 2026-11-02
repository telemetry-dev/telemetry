"""Local ingest server used by the wire-level tests: it captures OTLP posts and answers prompt
fetches."""

from __future__ import annotations

import gzip
import json
import secrets
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
    # The URL has a random path. Other local processes can send requests to the port; the server
    # answers a request outside the path with 404 and leaves it out of requests and replies.
    url: str
    # The requests under url, with the path below it.
    requests: list[CapturedRequest]
    # The HTTP status of each response. To make an ingest outage, set an error status such as 503.
    status: int = 200
    # The HTTP status of POST responses to one path, in place of status.
    path_status: dict[str, int] = field(default_factory=lambda: {})
    # The (status, JSON body) replies to GET requests, in order; the last one repeats.
    replies: list[tuple[int, object]] = field(default_factory=lambda: [])
    # Seconds that the server waits before it answers a GET request. The server reads it before the
    # request shows in requests, so that a test that sees the request can change it for the next.
    delay: float = 0
    # Seconds that the server waits before it answers a POST request.
    post_delay: float = 0
    # The location header of GET replies, for a redirect status.
    location: str | None = None
    # The Retry-After header of GET replies.
    retry_after: str | None = None

    def requests_for(self, path: str) -> list[CapturedRequest]:
        return [request for request in self.requests if request.path == path]


@contextmanager
def otlp_capture_server() -> Generator[CaptureServer]:
    captured: list[CapturedRequest] = []
    # Concurrent GETs take the next reply under this lock, so two cannot both pop the last ones.
    replies_lock = threading.Lock()
    prefix = f"/{secrets.token_hex(8)}"

    class Handler(BaseHTTPRequestHandler):
        def own_path(self) -> str | None:
            if not self.path.startswith(f"{prefix}/"):
                self.send_response(404)
                self.send_header("content-length", "0")
                self.end_headers()
                # The body of a POST is still unread.
                self.close_connection = True
                return None
            return self.path.removeprefix(prefix)

        def do_POST(self) -> None:
            path = self.own_path()
            if path is None:
                return
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length)
            request = CapturedRequest(
                path=path,
                headers={key.lower(): value for key, value in self.headers.items()},
                body=body,
            )
            captured.append(request)
            time.sleep(capture.post_delay)
            # Not captured[-1]: requests to other paths can arrive on other threads meanwhile.
            self.send_response(capture.path_status.get(request.path, capture.status))
            self.send_header("content-type", "application/x-protobuf")
            self.send_header("content-length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            path = self.own_path()
            if path is None:
                return
            delay = capture.delay
            captured.append(
                CapturedRequest(
                    path=path,
                    headers={key.lower(): value for key, value in self.headers.items()},
                    body=b"",
                )
            )
            time.sleep(delay)
            with replies_lock:
                status, reply = (
                    capture.replies[0] if len(capture.replies) == 1 else capture.replies.pop(0)
                )
            body = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            if capture.location is not None:
                self.send_header("location", capture.location)
            if capture.retry_after is not None:
                self.send_header("retry-after", capture.retry_after)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    host, port = server.server_address[0], server.server_address[1]
    capture = CaptureServer(url=f"http://{host}:{port}{prefix}", requests=captured)
    thread.start()
    try:
        yield capture
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
