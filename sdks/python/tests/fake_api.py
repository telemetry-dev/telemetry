"""The Evals routes of telemetry.dev on a local HTTP server: pages of a dataset version with a
cursor, an experiment that takes item posts of at most 100 and skips repeats, and its result."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlsplit

EXPERIMENT_ID = "6f9619ff-8b86-4011-b42d-00c04fc964ff"
EXPERIMENT_URL = f"https://telemetry.dev/o/org/p/project/evals/experiments/{EXPERIMENT_ID}"


def item_id(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012d}"


def dataset_item(number: int, split: str | None = None) -> dict[str, Any]:
    return {
        "id": item_id(number),
        "number": number,
        "input": {"question": f"q{number}"},
        "expectedOutput": f"a{number}",
        "metadata": {"source": "test"},
        "tags": ["golden"],
        "split": split,
    }


def result(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": EXPERIMENT_ID,
        "name": "support-golden-v3",
        "status": "completed",
        "checkStatus": None,
        "itemsTotal": 2,
        "itemsDone": 2,
        "itemsFailed": 0,
        "error": None,
        "scores": [],
        "url": EXPERIMENT_URL,
    }
    body.update(overrides)
    return body


@dataclass
class ApiRequest:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: Any

    @property
    def route(self) -> str:
        return f"{self.method} {self.path}"


# (status, JSON body, headers), or None to answer as the API does.
Reply = tuple[int, object, dict[str, str]]


@dataclass(frozen=True)
class Drop:
    """Closes the connection with no answer, as a network failure does."""


@dataclass(frozen=True)
class Late:
    """Answers after a wait, as a server that is slower than the timeout of the client."""

    seconds: float
    reply: Reply


@dataclass
class FakeApi:
    url: str
    items: list[dict[str, Any]] = field(default_factory=lambda: [dataset_item(1), dataset_item(2)])
    page_size: int = 100
    results: list[dict[str, Any]] = field(default_factory=lambda: [result()])
    override: Callable[[ApiRequest], Reply | Drop | Late | None] | None = None
    requests: list[ApiRequest] = field(default_factory=lambda: [])
    stored: set[str] = field(default_factory=lambda: set[str]())
    lock: threading.Lock = field(default_factory=threading.Lock)

    def routes(self) -> list[str]:
        return [request.route for request in self.requests]

    def posts(self) -> list[list[dict[str, Any]]]:
        return [
            request.body["items"] for request in self.requests if request.path.endswith("/items")
        ]

    def answer(self, request: ApiRequest) -> Reply | Drop | Late:
        with self.lock:
            self.requests.append(request)
            overridden = self.override(request) if self.override else None
            if overridden is not None:
                return overridden
            return self._route(request)

    def _route(self, request: ApiRequest) -> Reply:
        if request.route == "GET /v1/datasets/support-golden":
            split = request.query.get("split")
            start = int(request.query.get("cursor", "0"))
            matching = [item for item in self.items if split is None or item["split"] == split]
            more = start + self.page_size < len(matching)
            return (
                200,
                {
                    "dataset": {
                        "id": "dataset-1",
                        "name": "support-golden",
                        "version": int(request.query.get("version", "3")),
                    },
                    "items": matching[start : start + self.page_size],
                    "nextCursor": str(start + self.page_size) if more else None,
                },
                {},
            )
        if request.route == "POST /v1/experiments":
            split = request.body.get("split")
            count = len([item for item in self.items if split is None or item["split"] == split])
            return (
                201,
                {
                    "id": EXPERIMENT_ID,
                    "name": request.body.get("name", "support-golden-v3"),
                    "datasetVersion": request.body.get("datasetVersion"),
                    "itemsTotal": count * request.body.get("runsPerItem", 1),
                    "url": EXPERIMENT_URL,
                },
                {},
            )
        if request.route == f"POST /v1/experiments/{EXPERIMENT_ID}/items":
            runs = request.body["items"]
            if len(runs) > 100:
                return 400, {"error": {"code": "invalid_request", "message": "at most 100"}}, {}
            keys = [f"{run['itemId']}:{run.get('runIndex', 0)}" for run in runs]
            accepted = len([key for key in keys if key not in self.stored])
            self.stored.update(keys)
            return 200, {"accepted": accepted, "duplicates": len(runs) - accepted}, {}
        if request.route == f"POST /v1/experiments/{EXPERIMENT_ID}/complete":
            return 200, {"id": EXPERIMENT_ID, "status": "running"}, {}
        if request.route == f"GET /v1/experiments/{EXPERIMENT_ID}":
            return 200, self.results.pop(0) if len(self.results) > 1 else self.results[0], {}
        return 404, {"error": {"code": "not_found", "message": request.route}}, {}


@contextmanager
def fake_api(**options: Any) -> Generator[FakeApi]:
    api: FakeApi | None = None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._answer()

        def do_POST(self) -> None:
            self._answer()

        def _answer(self) -> None:
            assert api is not None
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            target = urlsplit(self.path)
            reply = api.answer(
                ApiRequest(
                    method=self.command,
                    path=target.path,
                    query=dict(parse_qsl(target.query)),
                    headers={key.lower(): value for key, value in self.headers.items()},
                    body=json.loads(raw) if raw else None,
                )
            )
            if isinstance(reply, Drop):
                self.close_connection = True
                return
            if isinstance(reply, Late):
                time.sleep(reply.seconds)
                reply = reply.reply
            status, body, headers = reply
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            pass

    class Server(ThreadingHTTPServer):
        def handle_error(self, request: Any, client_address: Any) -> None:
            # The answer of a Late reply can go to a client that has stopped waiting.
            if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
                super().handle_error(request, client_address)

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    host, port = server.server_address[0], server.server_address[1]
    api = FakeApi(url=f"http://{host}:{port}", **options)
    thread.start()
    try:
        yield api
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
