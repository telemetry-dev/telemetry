from __future__ import annotations

import asyncio
import contextvars
import copy
import dataclasses
import gc
import importlib
import itertools
import json
import os
import pickle
import re
import secrets
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import weakref
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import FrameType, SimpleNamespace
from typing import Any, NoReturn, TypeAlias
from urllib.parse import parse_qsl, urlsplit

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace

import telemetry_dev
import telemetry_dev._prompts as prompts_module
from telemetry_dev import (
    ChatPrompt,
    PromptError,
    TextPrompt,
    get_prompt,
    get_prompt_async,
    prompt_attributes_for,
    propagate_attributes,
    start_span,
    with_prompt,
)
from telemetry_dev._serialize import utf16_length

from .conftest import MakeClient
from .otlp_capture import CapturedRequest, CaptureServer, otlp_capture_server

# Python 3.12+ gives a warning for fork() in a process with threads, and the capture server has one.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
needs_fork = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


def reply(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "support/reply",
        "type": "chat",
        "label": "production",
        "version": 12,
        "messages": [
            {"role": "system", "content": "Use a {{tone}} tone."},
            {"role": "placeholder", "name": "history"},
            {"role": "user", "content": "{{ customer_message }}"},
        ],
        "text": None,
        "model": {"provider": "openai", "name": "gpt-4o"},
        "parameters": {"temperature": 0.3},
        "tools": [],
        "responseFormat": {"type": "text"},
        "rollout": None,
    }
    body.update(overrides)
    return body


def api_error(status: int, code: str, message: str) -> tuple[int, object]:
    return status, {"error": {"code": code, "message": message}}


@contextmanager
def prompt_server() -> Generator[CaptureServer]:
    """A capture server whose URL has a random path. Another local process can send requests to its
    port; ``own_requests`` leaves them out."""
    with otlp_capture_server() as capture:
        capture.url += f"/{secrets.token_hex(8)}"
        yield capture


def own_requests(server: CaptureServer) -> list[CapturedRequest]:
    """The requests to the URL of ``server``, with the paths that the SDK asked for."""
    prefix = urlsplit(server.url).path
    return [
        dataclasses.replace(request, path=request.path.removeprefix(prefix))
        for request in server.requests
        if request.path.startswith(f"{prefix}/")
    ]


@pytest.fixture
def server() -> Generator[CaptureServer]:
    with prompt_server() as capture:
        capture.replies = [(200, reply())]
        yield capture


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(prompts_module, "monotonic", lambda: now[0])
    return now


@pytest.fixture(autouse=True)
def empty_cache() -> None:
    # A fetch that runs on after its test must not answer a later test on a reused port.
    prompts_module._cache.clear()  # pyright: ignore[reportPrivateUsage]


def chat_prompt(*messages: dict[str, Any], name: str = "support/reply") -> ChatPrompt:
    return ChatPrompt(
        name=name,
        version=12,
        label="production",
        messages=list(messages),
        model=None,
        parameters={},
        tools=[],
        response_format={"type": "text"},
        is_fallback=False,
    )


def text_prompt(text: str) -> TextPrompt:
    return TextPrompt(
        name="support/reply",
        version=12,
        label="production",
        text=text,
        model=None,
        parameters={},
        tools=[],
        response_format={"type": "text"},
        is_fallback=False,
    )


def query_of(path: str) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(path).query))


def wait_for(check: Callable[[], bool]) -> None:
    """Wait for a fetch in the background to end."""
    deadline = time.monotonic() + 5
    while not check():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def in_fork(child: Callable[[], str], fork: Callable[[], int] | None = None) -> str:
    """Run ``child`` in a process that ``fork`` (by default ``os.fork``) makes, and return its
    result, or the repr of its exception or of an exception in a fork hook of the prompt module.
    The parent kills a child that does not end in 10 s, and the test fails."""
    # Python ignores an exception in a fork hook and only gives it to sys.unraisablehook. The
    # hooks of OpenTelemetry raise after an earlier test discards its batch processor.
    hook_errors: list[BaseException | None] = []

    def record(unraisable: sys.UnraisableHookArgs) -> None:
        frames = traceback.extract_tb(unraisable.exc_traceback)
        if any(frame.filename == prompts_module.__file__ for frame in frames):
            hook_errors.append(unraisable.exc_value)

    previous_hook = sys.unraisablehook
    parent = os.getpid()
    read_end, write_end = os.pipe()
    sys.unraisablehook = record
    try:
        pid = (fork or os.fork)()
    finally:
        sys.unraisablehook = previous_hook
    # Not pid == 0: a fork function that forks more than one time can return a pid in a child.
    if os.getpid() != parent:
        try:
            os.close(read_end)
            try:
                result = repr(hook_errors[0]) if hook_errors else child()
            except BaseException as exc:
                result = repr(exc)
            os.write(write_end, result.encode())
        finally:
            # The child must not return into pytest.
            os._exit(0)
    os.close(write_end)
    with os.fdopen(read_end, "rb") as reader:
        deadline = time.monotonic() + 10
        while os.waitpid(pid, os.WNOHANG) == (0, 0):
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                pytest.fail("The forked process did not end in 10 s")
            time.sleep(0.01)
        result = reader.read().decode()
    assert hook_errors == []
    return result


def version_of(server: CaptureServer) -> int | None:
    return get_prompt("support/reply", api_key="td_live_a", base_url=server.url).version


@pytest.mark.parametrize(
    ("options", "query"),
    [
        ({}, {"label": "production"}),
        ({"label": "staging"}, {"label": "staging"}),
        ({"version": 13}, {"version": "13"}),
    ],
)
def test_get_prompt_sends_query_with_key_and_sdk_header(
    server: CaptureServer, options: dict[str, Any], query: dict[str, str]
) -> None:
    get_prompt("support/reply", api_key="td_live_a", base_url=f"{server.url}/", **options)

    [request] = own_requests(server)
    assert urlsplit(request.path).path == "/v1/prompts"
    assert query_of(request.path) == {"name": "support/reply", **query}
    assert request.headers["authorization"] == "Bearer td_live_a"
    assert request.headers["x-telemetry-dev-sdk"] == "telemetry-dev"
    assert request.headers["user-agent"] == f"telemetry-dev-python/{telemetry_dev.__version__}"


def test_get_prompt_reads_the_initialized_client_then_the_environment(
    server: CaptureServer, make: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    make(api_key="td_live_client", base_url=server.url)
    assert get_prompt("support/reply").version == 12
    assert own_requests(server)[0].headers["authorization"] == "Bearer td_live_client"

    telemetry_dev.shutdown()
    telemetry_dev.init(api_key="td_live_disabled", enabled=False, disable_atexit=True)
    monkeypatch.setenv("TELEMETRY_DEV_API_KEY", "td_live_env")
    monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", server.url)
    get_prompt("support/reply", label="staging")
    assert own_requests(server)[1].headers["authorization"] == "Bearer td_live_env"


def test_get_prompt_reads_the_host_that_telemetry_goes_to(server: CaptureServer) -> None:
    telemetry_dev.init(
        api_key="td_live_a", base_url=f"{server.url}/", export_mode="immediate", disable_atexit=True
    )
    with telemetry_dev.start_span("work"):
        pass
    telemetry_dev.flush()

    get_prompt("support/reply")

    assert [urlsplit(request.path).path for request in own_requests(server)] == [
        "/v1/traces",
        "/v1/prompts",
    ]


@pytest.mark.parametrize("env_base_url", [None, ""])
@pytest.mark.parametrize("initialized", [False, True])
def test_without_a_base_url_get_prompt_reads_the_default_host(
    make: MakeClient, monkeypatch: pytest.MonkeyPatch, initialized: bool, env_base_url: str | None
) -> None:
    urls: list[str] = []

    # The default base URL is a real host: the opener records the request and does not send it.
    def offline(request: urllib.request.Request, timeout: float) -> NoReturn:
        urls.append(request.full_url)
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(prompts_module, "_opener", SimpleNamespace(open=offline))
    if env_base_url is not None:
        monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", env_base_url)
    if initialized:
        make()

    with pytest.raises(PromptError):
        get_prompt("support/reply", api_key="td_live_a")

    assert urls == ["https://api.telemetry.dev/v1/prompts?name=support%2Freply&label=production"]


def test_get_prompt_keeps_a_prompt_for_the_ttl_then_refreshes_it_in_the_background(
    server: CaptureServer, clock: list[float]
) -> None:
    server.replies = [(200, reply()), (200, reply(version=13))]

    def version() -> int | None:
        return get_prompt("support/reply", api_key="td_live_a", base_url=server.url).version

    assert version() == 12
    clock[0] += 59
    assert version() == 12
    assert len(own_requests(server)) == 1

    # After the TTL, calls get the copy at once while one refresh runs.
    server.delay = 0.5
    clock[0] += 1
    assert [version() for _ in range(3)] == [12, 12, 12]
    wait_for(lambda: version() == 13)
    assert len(own_requests(server)) == 2

    # A different key does not share the copy.
    server.delay = 0
    get_prompt("support/reply", api_key="td_live_b", base_url=server.url)
    assert len(own_requests(server)) == 3


def test_concurrent_calls_share_one_fetch(server: CaptureServer) -> None:
    server.delay = 0.2
    versions: list[int | None] = []

    def fetch() -> None:
        versions.append(
            get_prompt("support/reply", api_key="td_live_a", base_url=server.url).version
        )

    threads = [threading.Thread(target=fetch) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(own_requests(server)) == 1
    assert versions == [12] * 5


def test_a_failed_refresh_returns_the_last_copy_and_reports_the_error(
    server: CaptureServer, make: MakeClient, clock: list[float]
) -> None:
    errors: list[BaseException] = []
    make(api_key="td_live_a", base_url=server.url, on_error=errors.append)
    server.replies = [
        (200, reply()),
        api_error(503, "prompts_unavailable", "Prompts are unavailable"),
        (200, {"name": "support/reply"}),
    ]

    assert get_prompt("support/reply").version == 12
    clock[0] += 60
    assert get_prompt("support/reply").version == 12
    wait_for(lambda: len(errors) == 1)
    assert isinstance(errors[0], PromptError)
    assert (errors[0].code, errors[0].status) == ("prompts_unavailable", 503)

    # The stale copy counts as fresh for another TTL, so calls do not each start a fetch.
    get_prompt("support/reply")
    assert len(own_requests(server)) == 2

    clock[0] += 60
    assert get_prompt("support/reply").version == 12
    wait_for(lambda: len(errors) == 2)
    assert isinstance(errors[1], PromptError)
    assert errors[1].code == "invalid_response"


def test_with_no_copy_a_failed_fetch_returns_the_fallback_and_so_do_calls_for_10_seconds(
    server: CaptureServer, clock: list[float]
) -> None:
    server.replies = [api_error(404, "label_not_found", "support/reply has no production label")]

    def fallback_prompt(cache_ttl_seconds: float = 60) -> ChatPrompt:
        return get_prompt(
            "support/reply",
            api_key="td_live_a",
            base_url=server.url,
            cache_ttl_seconds=cache_ttl_seconds,
            fallback=[{"role": "user", "content": "Answer {{customer_message}}"}],
        )

    prompt = fallback_prompt()
    assert (prompt.name, prompt.version, prompt.label, prompt.model, prompt.is_fallback) == (
        "support/reply",
        None,
        "production",
        None,
        True,
    )
    assert prompt.compile(customer_message="hi") == [{"role": "user", "content": "Answer hi"}]

    clock[0] += 9
    assert fallback_prompt().is_fallback
    assert len(own_requests(server)) == 1

    clock[0] += 1
    assert fallback_prompt().is_fallback
    assert len(own_requests(server)) == 2

    # A TTL under 10 seconds bounds the wait too.
    fallback_prompt(cache_ttl_seconds=0)
    assert len(own_requests(server)) == 3


@pytest.mark.parametrize(
    ("status", "retry_after", "wait"),
    [
        (429, "120", 120),
        (503, "120", 120),
        # An HTTP date 120 seconds from now; the test makes it when it runs.
        (429, "date", 120),
        (429, "86400", 3600),
        # Not a Retry-After that get_prompt reads: the 10 second error wait applies.
        (429, "soon", 10),
        (429, "Thu, 01 Jan 2099 00:00:00", 10),
        (429, "01 Jan 2099 00:00:00 GMT", 10),
        (500, "120", 10),
    ],
)
def test_with_no_copy_get_prompt_does_not_fetch_before_the_retry_after(
    server: CaptureServer, clock: list[float], status: int, retry_after: str, wait: float
) -> None:
    server.replies = [api_error(status, "rate_limited", "Too many prompt requests")]
    server.retry_after = (
        format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)
        if retry_after == "date"
        else retry_after
    )

    def fallback_prompt() -> ChatPrompt:
        return get_prompt(
            "support/reply",
            api_key="td_live_a",
            base_url=server.url,
            fallback=[{"role": "user", "content": "Hi"}],
        )

    assert fallback_prompt().is_fallback
    # The HTTP date counts from when the server answered, a moment before this.
    clock[0] += wait - 2
    assert fallback_prompt().is_fallback
    assert len(own_requests(server)) == 1

    clock[0] += 2
    fallback_prompt()
    assert len(own_requests(server)) == 2


def test_a_refresh_that_gets_a_retry_after_keeps_the_copy_until_then(
    server: CaptureServer, clock: list[float]
) -> None:
    server.replies = [
        (200, reply()),
        api_error(429, "rate_limited", "Too many prompt requests"),
        (200, reply(version=13)),
    ]
    server.retry_after = "300"

    def version() -> int | None:
        return get_prompt("support/reply", api_key="td_live_a", base_url=server.url).version

    assert version() == 12
    clock[0] += 60
    assert version() == 12
    wait_for(
        lambda: all(
            entry.pending is None
            for entry in prompts_module._cache.values()  # pyright: ignore[reportPrivateUsage]
        )
    )
    assert len(own_requests(server)) == 2

    # The TTL ends after 60 seconds, but the server asked for 300.
    clock[0] += 299
    assert version() == 12
    wait_for(
        lambda: all(
            entry.pending is None
            for entry in prompts_module._cache.values()  # pyright: ignore[reportPrivateUsage]
        )
    )
    assert len(own_requests(server)) == 2

    clock[0] += 1
    version()
    wait_for(lambda: version() == 13)
    assert len(own_requests(server)) == 3


def test_with_no_copy_and_no_fallback_get_prompt_raises_a_prompt_error(
    server: CaptureServer,
) -> None:
    server.replies = [api_error(404, "label_not_found", "support/reply has no production label")]

    with pytest.raises(PromptError) as raised:
        get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    assert (raised.value.code, raised.value.status, raised.value.message) == (
        "label_not_found",
        404,
        "support/reply has no production label",
    )

    for api_key in (None, ""):
        with pytest.raises(PromptError) as missing:
            get_prompt("support/reply", api_key=api_key, base_url=server.url)
        assert missing.value.code == "missing_api_key"

    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    with pytest.raises(PromptError) as offline:
        get_prompt("support/reply", api_key="td_live_a", base_url=f"http://127.0.0.1:{port}")
    assert offline.value.code == "network_error"


def test_a_reply_without_a_label_is_an_invalid_response(server: CaptureServer) -> None:
    unlabeled = reply()
    del unlabeled["label"]
    server.replies = [(200, unlabeled)]

    with pytest.raises(PromptError) as raised:
        get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    assert raised.value.code == "invalid_response"

    fallback = get_prompt(
        "support/reply",
        label="staging",
        api_key="td_live_a",
        base_url=server.url,
        fallback=[{"role": "user", "content": "Hi"}],
    )
    assert fallback.is_fallback


def test_a_redirect_fails_and_does_not_send_the_key(server: CaptureServer) -> None:
    with prompt_server() as other:
        other.replies = [(200, reply())]
        server.replies = [(302, {})]
        server.location = f"{other.url}/v1/prompts?name=support%2Freply&label=production"

        with pytest.raises(PromptError) as raised:
            get_prompt("support/reply", api_key="td_live_a", base_url=server.url)

        assert (raised.value.code, raised.value.status) == ("http_302", 302)
        assert own_requests(other) == []


@pytest.mark.parametrize("slow", ["reply", "address lookup"])
def test_a_fetch_that_takes_too_long_fails_with_timeout(
    server: CaptureServer, monkeypatch: pytest.MonkeyPatch, slow: str
) -> None:
    monkeypatch.setattr(prompts_module, "_FETCH_TIMEOUT_S", 0.2)
    if slow == "reply":
        server.delay = 1
    else:
        # The timeout of urllib does not cover the address lookup.
        lookup = socket.getaddrinfo

        def slow_lookup(*args: Any, **kwargs: Any) -> Any:
            time.sleep(1)
            return lookup(*args, **kwargs)

        monkeypatch.setattr(socket, "getaddrinfo", slow_lookup)

    started = time.monotonic()
    with pytest.raises(PromptError) as raised:
        get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    assert raised.value.code == "timeout"
    assert time.monotonic() - started < 0.8


def test_a_fetch_stops_reading_a_trickled_reply_at_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts_module, "_FETCH_TIMEOUT_S", 0.2)
    disconnected = threading.Event()

    # Each byte comes within the socket timeout, so only a deadline for the whole fetch ends it.
    class Trickle(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("content-length", "1000")
            self.end_headers()
            try:
                for _ in range(60):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                disconnected.set()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    # server_close then waits for the reply thread, so that it does not write after the test.
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        with pytest.raises(PromptError) as raised:
            get_prompt(
                "support/reply",
                api_key="td_" + "live_a",
                base_url=f"http://127.0.0.1:{server.server_address[1]}",
            )
        assert raised.value.code == "timeout"
        # The fetch closed the connection, rather than reading for the 3 s of the reply.
        assert disconnected.wait(1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_a_fetch_starts_no_connection_after_its_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prompts_module, "_FETCH_TIMEOUT_S", 0.2)
    timeouts: list[float | None] = []

    # Two addresses whose connect waits for the whole timeout, as an unreachable host does.
    def lookup(*args: Any, **kwargs: Any) -> list[Any]:
        address = (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.0.2.1", 80))
        return [address, address]

    class Unreachable:
        def __init__(self, *args: Any) -> None:
            self.timeout: float | None = None

        def settimeout(self, timeout: float | None) -> None:
            self.timeout = timeout

        def connect(self, address: Any) -> NoReturn:
            timeouts.append(self.timeout)
            time.sleep(self.timeout or 0)
            raise TimeoutError

        def close(self) -> None:
            pass

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setattr(socket, "socket", Unreachable)
    with pytest.raises(PromptError) as raised:
        get_prompt("support/reply", api_key="td_" + "live_a", base_url="http://prompts.test")
    assert raised.value.code == "timeout"
    time.sleep(0.3)
    assert len(timeouts) == 1
    assert timeouts[0] is not None and timeouts[0] <= 0.2


@needs_fork
@pytest.mark.parametrize("reused_pid", [False, True])
def test_a_forked_process_fetches_the_prompt_that_the_parent_was_fetching(
    server: CaptureServer, monkeypatch: pytest.MonkeyPatch, reused_pid: bool
) -> None:
    # This puts a limit on the wait of a child that waits for the fetch of the parent.
    monkeypatch.setattr(prompts_module, "_FETCH_TIMEOUT_S", 3)
    server.delay = 1
    versions: list[int | None] = []
    parent = threading.Thread(target=lambda: versions.append(version_of(server)))
    parent.start()
    wait_for(lambda: len(own_requests(server)) == 1)
    server.delay = 0
    parent_pid = os.getpid()

    def child() -> str:
        if reused_pid:
            # A reused pid: the child gets the id of the process that started the fetch.
            monkeypatch.setattr(os, "getpid", lambda: parent_pid)
        return str(version_of(server))

    assert in_fork(child) == "12"
    parent.join()
    assert versions == [12]
    # The fetch of the parent, and the fetch of the child.
    assert len(own_requests(server)) == 2


def test_a_call_does_not_wait_for_a_fetch_that_another_process_started(
    server: CaptureServer,
) -> None:
    server.delay = 1
    first = threading.Thread(target=version_of, args=(server,))
    first.start()
    wait_for(lambda: len(own_requests(server)) == 1)
    server.delay = 0
    # A fork that runs no fork hooks, as a fork in C code, copies the fetch of the parent but not
    # the thread that runs it. Here this process is the child: the fetch gets the id of its parent.
    (entry,) = prompts_module._cache.values()  # pyright: ignore[reportPrivateUsage]
    entry.pending_pid = os.getppid()

    assert version_of(server) == 12
    assert len(own_requests(server)) == 2
    first.join()


def test_a_call_does_not_wait_for_a_cache_lock_that_another_process_held(
    server: CaptureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    held, done = threading.Event(), threading.Event()
    released_at: list[float] = []

    def hold() -> None:
        with prompts_module._cache_lock():  # pyright: ignore[reportPrivateUsage]
            held.set()
            done.wait(5)
            released_at.append(time.monotonic())

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait()
    # A fork that runs no fork hooks, as a fork in C code, copies the lock while a thread of the
    # parent holds it. Here this process is the child: it gets a new process ID.
    pid = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: pid + 1)
    try:
        assert version_of(server) == 12
        returned_at = time.monotonic()
    finally:
        done.set()
        holder.join()
    assert returned_at < released_at[0]


@needs_fork
def test_a_forked_process_refreshes_the_copy_that_the_parent_was_refreshing(
    server: CaptureServer, clock: list[float]
) -> None:
    assert version_of(server) == 12
    clock[0] += 60
    server.replies = [(200, reply(version=13))]
    server.delay = 1
    assert version_of(server) == 12
    wait_for(lambda: len(own_requests(server)) == 2)
    server.delay = 0

    def child() -> str:
        kept = version_of(server)
        wait_for(lambda: version_of(server) == 13)
        return str(kept)

    # The child returns the copy of the parent. Then its refresh gets version 13.
    assert in_fork(child) == "12"


@needs_fork
@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_fork_waits_for_a_thread_that_holds_the_cache_lock(server: CaptureServer) -> None:
    held = threading.Event()
    released_at: list[float] = []
    forked_at: list[float] = []
    errors: list[BaseException] = []

    def hold() -> None:
        try:
            with prompts_module._cache_lock():  # pyright: ignore[reportPrivateUsage]
                held.set()
                time.sleep(0.2)
                released_at.append(time.monotonic())
        except BaseException as exc:
            errors.append(exc)

    def fork() -> int:
        pid = os.fork()
        if pid != 0:
            forked_at.append(time.monotonic())
        return pid

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait()
    # The child does not get a lock that a thread of the parent holds.
    assert in_fork(lambda: str(version_of(server)), fork) == "12"
    holder.join()
    assert errors == []
    assert forked_at[0] > released_at[0]


@needs_fork
def test_a_fork_in_the_thread_that_holds_the_cache_lock_returns(
    server: CaptureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A process that still holds the lock after the fork waits for its fetch until the timeout.
    monkeypatch.setattr(prompts_module, "_FETCH_TIMEOUT_S", 3)

    def fork_in_the_lock() -> int:
        with prompts_module._cache_lock():  # pyright: ignore[reportPrivateUsage]
            return os.fork()

    def fork_and_fetch() -> str:
        child = in_fork(lambda: str(version_of(server)), fork_in_the_lock)
        return f"{child} {version_of(server)}"

    # The fork runs in a child, so that a fork that waits for itself stops only that child.
    assert in_fork(fork_and_fetch) == "12 12"


@needs_fork
def test_a_fork_in_a_call_that_starts_a_fetch_returns_the_prompt(
    server: CaptureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A process that waits for a fetch that it does not have waits until the timeout.
    monkeypatch.setattr(prompts_module, "_FETCH_TIMEOUT_S", 3)
    pids: list[int] = []
    results: list[str] = []

    class ForkingThread(threading.Thread):
        def start(self) -> None:
            # A fork in the call, as in a signal handler, after the call made its fetch and before
            # the fetch thread starts. The child hook drops the fetch of the entry.
            if not pids:
                pids.append(os.fork())
            super().start()

    def fork() -> int:
        with monkeypatch.context() as patch:
            # Only the threads of the prompt module: a fork hook of OpenTelemetry starts a thread.
            patch.setattr(
                prompts_module,
                "threading",
                SimpleNamespace(**{**vars(threading), "Thread": ForkingThread}),
            )
            try:
                results.append(str(version_of(server)))
            except Exception as exc:
                # The child must return the pid to in_fork, and not raise into pytest.
                results.append(repr(exc))
        return pids[0]

    # The parent and the child each start the fetch thread, and each call returns its result.
    assert in_fork(lambda: results[0], fork) == "12"
    assert results == ["12"]


@needs_fork
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs signal.setitimer")
def test_a_fork_in_a_signal_handler_during_the_wait_for_a_fetch_fetches_again_in_the_child(
    server: CaptureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A child that waits for the fetch of the parent gets no answer in 3 s.
    monkeypatch.setattr(prompts_module, "_FETCH_TIMEOUT_S", 3)
    server.delay = 1
    pids: list[int] = []
    results: list[str] = []
    forked_in: list[str | None] = []

    def fork_in_the_wait(signum: int, frame: FrameType | None) -> None:
        server.delay = 0
        forked_in.append(frame and frame.f_code.co_name)
        pids.append(os.fork())

    def fork() -> int:
        previous = signal.signal(signal.SIGALRM, fork_in_the_wait)
        # Between two 0.1 s steps of the wait.
        signal.setitimer(signal.ITIMER_REAL, 0.35)
        started = time.monotonic()
        try:
            version = version_of(server)
            results.append(f"{version} in under 2 s: {time.monotonic() - started < 2}")
        except Exception as exc:
            # The child must return the pid to in_fork, and not raise into pytest.
            results.append(repr(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        return pids[0]

    assert in_fork(lambda: results[0], fork) == "12 in under 2 s: True"
    assert results == ["12 in under 2 s: True"]
    # The signal came in the wait of threading.Event, not before it.
    assert forked_in == ["wait"]
    # The fetch of the parent, and the fetch of the child.
    assert len(own_requests(server)) == 2


@needs_fork
def test_a_fork_after_a_reload_of_the_prompt_module_returns(
    server: CaptureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = dict(prompts_module.__dict__)
    importlib.reload(prompts_module)
    try:
        # The reload sets the timeout again. A process that still holds the lock after the fork
        # waits for its fetch until the timeout.
        monkeypatch.setattr(prompts_module, "_FETCH_TIMEOUT_S", 3)
        assert in_fork(lambda: str(version_of(server))) == "12"
        assert version_of(server) == 12
    finally:
        # The reload gave the module new classes and a new cache.
        prompts_module.__dict__.update(saved)


@needs_fork
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs signal.setitimer")
@pytest.mark.parametrize("before_hook", ["stopped by a signal", "not run"])
def test_the_fork_hooks_handle_a_lock_that_the_before_hook_did_not_get(
    server: CaptureServer, before_hook: str
) -> None:
    class Interrupted(Exception):
        pass

    lock = prompts_module._cache_lock()  # pyright: ignore[reportPrivateUsage]
    held, done = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def hold() -> None:
        try:
            with lock:
                held.set()
                # The fork must end while this thread holds the lock. If no signal stops the
                # before hook, the hook waits for this thread, and this assert fails after 5 s.
                assert done.wait(5)
        except BaseException as exc:
            errors.append(exc)

    def interrupt(signum: int, frame: FrameType | None) -> NoReturn:
        raise Interrupted

    def fork_in_a_signal() -> int:
        # SIGALRM raises while the before hook waits for the lock. os.fork() gives the error to
        # sys.unraisablehook, forks, and then runs the after hooks.
        in_fork_hook = sys.unraisablehook

        def record(unraisable: sys.UnraisableHookArgs) -> None:
            if not isinstance(unraisable.exc_value, Interrupted):
                in_fork_hook(unraisable)

        previous = signal.signal(signal.SIGALRM, interrupt)
        sys.unraisablehook = record
        signal.setitimer(signal.ITIMER_REAL, 0.1)
        try:
            return os.fork()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            sys.unraisablehook = in_fork_hook
            signal.signal(signal.SIGALRM, previous)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait()
    try:
        if before_hook == "stopped by a signal":
            # In the child, the holder is a thread of the parent that does not exist there.
            assert in_fork(lambda: str(version_of(server)), fork_in_a_signal) == "12"
        else:
            # uWSGI with py-call-osafterfork forks in C and runs only the child hook.
            prompts_module._reset_in_child()  # pyright: ignore[reportPrivateUsage]
            child_lock = prompts_module._cache_lock()  # pyright: ignore[reportPrivateUsage]
            assert child_lock.acquire(blocking=False)
            child_lock.release()
    finally:
        done.set()
        holder.join()
    assert errors == []


def test_get_prompt_rejects_a_prompt_of_the_other_type_and_bad_options(
    server: CaptureServer,
) -> None:
    server.replies = [(200, reply(type="text", messages=[], text="Hi {{name}}"))]
    url = server.url

    with pytest.raises(PromptError) as raised:
        get_prompt("support/reply", api_key="td_live_a", base_url=url)
    assert raised.value.code == "type_mismatch"
    text = get_prompt("support/reply", type="text", api_key="td_live_a", base_url=url)
    assert text.compile(name=7) == "Hi 7"

    with pytest.raises(TypeError):
        get_prompt("support/reply", label="a", version=2, api_key="td_live_a", base_url=url)
    with pytest.raises(ValueError):
        get_prompt("support/reply", version=0, api_key="td_live_a", base_url=url)


def test_one_session_keeps_one_version_of_a_gradual_rollout(
    server: CaptureServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rollout(percent: int) -> tuple[int, object]:
        return 200, reply(
            rollout={
                "percent": percent,
                "version": 13,
                "messages": [{"role": "user", "content": "v13"}],
                "text": None,
                "model": {"provider": "openai", "name": "gpt-4o-mini"},
                "parameters": {},
                "tools": [],
                "responseFormat": {"type": "text"},
            }
        )

    server.replies = [rollout(15), rollout(16), rollout(16), rollout(16)]
    keys = itertools.count()

    def fetch() -> ChatPrompt:
        # A new key for each call skips the cache.
        return get_prompt("support/reply", api_key=f"td_live_{next(keys)}", base_url=server.url)

    # FNV-1a("support/reply:production:session-123") = 777154315, so the bucket is 15. The
    # TypeScript SDK test asserts the same vector.
    with propagate_attributes(session_id="session-123"):
        assert fetch().version == 12
        chosen = fetch()
    assert chosen.version == 13
    assert chosen.model == {"provider": "openai", "name": "gpt-4o-mini"}
    assert chosen.compile() == [{"role": "user", "content": "v13"}]

    # Without a session, each call draws a bucket.
    draws = iter([15, 16])

    def draw(_stop: int) -> int:
        return next(draws)

    monkeypatch.setattr(prompts_module.random, "randrange", draw)
    assert fetch().version == 13
    assert fetch().version == 12


def test_compile_fills_variables_and_expands_placeholders(server: CaptureServer) -> None:
    prompt = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    history = [
        {"role": "user", "content": "Hi {{tone}}"},
        {"role": "assistant", "content": "Hello"},
    ]

    assert prompt.compile(
        {"tone": "friendly", "customer_message": "ignored"},
        customer_message="Where is #4?",
        history=history,
    ) == [
        {"role": "system", "content": "Use a friendly tone."},
        *history,
        {"role": "user", "content": "Where is #4?"},
    ]

    # Missing values become empty text, other values JSON.
    assert prompt.compile(customer_message=42) == [
        {"role": "system", "content": "Use a  tone."},
        {"role": "user", "content": "42"},
    ]
    assert prompt.compile(tone=None, customer_message=True)[1] == {
        "role": "user",
        "content": "true",
    }

    with pytest.raises(TypeError, match="The history variable must be a list of messages"):
        prompt.compile(history="not a list")

    # The cached copy does not change when an app edits a prompt.
    prompt.messages[0]["content"] = "changed"
    again = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    assert again.messages[0]["content"] == "Use a {{tone}} tone."


def test_with_prompt_stamps_model_call_spans_in_scope_and_no_others(
    server: CaptureServer, make: MakeClient
) -> None:
    memory = make()
    prompt = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)

    with start_span("before", type="generation"):
        pass
    outside = start_span("outside")
    outside_context = trace.set_span_in_context(outside.span)

    with with_prompt(prompt):
        prompt.compile(customer_message="Where is #4?", tone="friendly")
        with start_span("chat gpt-4o", type="generation"):
            pass
        start_span(
            "image", type="generation", attributes={"gen_ai.operation.name": "generate_content"}
        ).end()
        with start_span("tool", type="tool"):
            pass
        start_span("plain").end()
        start_span("embed", type="embedding").end()
        start_span("explicit parent", type="generation", parent=outside_context).end()
        start_span("list operation", attributes={"gen_ai.operation.name": ["chat"]}).end()

    outside.end()
    with start_span("after", type="generation"):
        pass

    by_name = {
        span.name: dict(span.attributes or {}) for span in memory.span_exporter.get_finished_spans()
    }
    for name in ("chat gpt-4o", "image", "explicit parent"):
        assert by_name[name]["td.prompt.name"] == "support/reply"
        assert by_name[name]["td.prompt.version"] == 12
        assert by_name[name]["td.prompt.label"] == "production"
        assert (
            by_name[name]["td.prompt.variables"]
            == '{"customer_message":"Where is #4?","tone":"friendly"}'
        )
    for name in ("before", "outside", "tool", "plain", "embed", "list operation", "after"):
        assert "td.prompt.name" not in by_name[name]


@pytest.mark.parametrize("where", ["before", "inside"])
def test_each_thread_records_its_own_variables_when_it_compiles_a_shared_prompt(
    server: CaptureServer, make: MakeClient, where: str
) -> None:
    memory = make()
    prompt = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)

    def task(name: str, customer_message: str, ready: threading.Event, go: threading.Event) -> None:
        if where == "before":
            prompt.compile(customer_message=customer_message)
        with with_prompt(prompt):
            if where == "inside":
                prompt.compile(customer_message=customer_message)
            ready.set()
            go.wait()
            start_span(name, type="generation").end()

    # A compiles and waits in with_prompt. Then B compiles and calls the model, and A calls it last.
    a_ready, b_done, go_now = threading.Event(), threading.Event(), threading.Event()
    go_now.set()
    a = threading.Thread(target=task, args=("a", "from A", a_ready, b_done))
    a.start()
    a_ready.wait()
    task("b", "from B", threading.Event(), go_now)
    b_done.set()
    a.join()

    variables: dict[str, Any] = {
        span.name: dict(span.attributes or {}).get("td.prompt.variables")
        for span in memory.span_exporter.get_finished_spans()
    }
    assert variables == {
        "a": '{"customer_message":"from A"}',
        "b": '{"customer_message":"from B"}',
    }


async def test_each_task_in_one_with_prompt_records_its_own_variables(
    server: CaptureServer, make: MakeClient
) -> None:
    memory = make()
    prompt = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    other = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    a_compiled, b_called = asyncio.Event(), asyncio.Event()

    async def task_a() -> None:
        prompt.compile(customer_message="from A")
        # A compile in a nested with_prompt does not replace A's variables of the outer one.
        with with_prompt(other):
            other.compile(customer_message="nested")
        a_compiled.set()
        await b_called.wait()
        start_span("a", type="generation").end()

    async def task_b() -> None:
        await a_compiled.wait()
        prompt.compile(customer_message="from B")
        start_span("b", type="generation").end()
        b_called.set()

    # A compiles and waits in the shared scope. Then B compiles and calls the model, and A calls
    # it last.
    with with_prompt(prompt):
        await asyncio.gather(task_a(), task_b())

    variables: dict[str, Any] = {
        span.name: dict(span.attributes or {}).get("td.prompt.variables")
        for span in memory.span_exporter.get_finished_spans()
    }
    assert variables == {
        "a": '{"customer_message":"from A"}',
        "b": '{"customer_message":"from B"}',
    }


def test_with_prompt_can_end_in_another_context(server: CaptureServer, make: MakeClient) -> None:
    make()
    prompt = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)

    # A generator that a framework closes in a different Context than the one it started in.
    def stream() -> Generator[None]:
        with with_prompt(prompt):
            yield

    steps = stream()
    contextvars.copy_context().run(next, steps)
    contextvars.copy_context().run(steps.close)


def test_variables_are_omitted_when_input_capture_is_off(
    server: CaptureServer, make: MakeClient
) -> None:
    masked: list[str] = []

    def mask(value: Any, context: telemetry_dev.MaskContext) -> Any:
        masked.append(context.key)
        return value

    memory = make(mask=mask)
    server.replies = [(200, reply(label=None, version=7))]
    prompt = get_prompt("support/reply", version=7, api_key="td_live_a", base_url=server.url)
    prompt.compile(tone="SECRET")

    with with_prompt(prompt):
        start_span("captured", type="generation").end()
        start_span("private", type="generation", capture_input=False).end()

    captured, private = (
        dict(s.attributes or {}) for s in memory.span_exporter.get_finished_spans()
    )
    assert captured["td.prompt.version"] == 7
    assert captured["td.prompt.variables"] == '{"tone":"SECRET"}'
    assert "td.prompt.label" not in captured
    assert "td.prompt.variables" in masked
    assert private["td.prompt.name"] == "support/reply"
    assert "td.prompt.variables" not in private

    telemetry_dev.shutdown()
    off = make(capture_input=False)
    with with_prompt(prompt):
        start_span("off", type="generation").end()
    [span] = off.span_exporter.get_finished_spans()
    assert (span.attributes or {})["td.prompt.version"] == 7
    assert "td.prompt.variables" not in (span.attributes or {})


DESK_MESSAGES = [
    {"role": "system", "content": "Answer as the {{team}} desk."},
    {"role": "user", "content": "Order {{order}}: where is it?"},
]


def test_prompt_attributes_for_names_the_compile_of_a_message_dict_and_a_copy_names_nothing(
    make: MakeClient,
) -> None:
    make()
    compiled = chat_prompt(*DESK_MESSAGES).compile(team="billing", order="#4")
    attributes = {
        "td.prompt.name": "support/reply",
        "td.prompt.version": 12,
        "td.prompt.label": "production",
        "td.prompt.variables": '{"team":"billing","order":"#4"}',
    }

    assert prompt_attributes_for(compiled) == attributes
    assert prompt_attributes_for(tuple(compiled)) == attributes
    assert prompt_attributes_for(compiled[0]) == attributes
    assert json.dumps(compiled[0]) == '{"role": "system", "content": "Answer as the billing desk."}'

    copies: list[object] = [
        [dict(message) for message in compiled],
        [message.copy() for message in compiled],
        [copy.copy(message) for message in compiled],
        copy.deepcopy(compiled),
        pickle.loads(pickle.dumps(compiled)),
        json.loads(json.dumps(compiled)),
        [message["content"] for message in compiled],
    ]
    for messages in copies:
        assert prompt_attributes_for(messages) == {}

    summary = text_prompt("Summarize {{doc}}.").compile(doc="the policy")
    assert prompt_attributes_for([summary]) == {}


def test_an_unrelated_message_with_the_text_of_a_compiled_message_is_not_tagged(
    make: MakeClient,
) -> None:
    make()
    chat_prompt({"role": "system", "content": "You are a helpful assistant."}).compile()

    unrelated = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    assert prompt_attributes_for(unrelated) == {}


def test_two_tasks_with_one_text_and_different_other_variables_each_record_their_own(
    make: MakeClient,
) -> None:
    memory = make()
    prompt = chat_prompt(*DESK_MESSAGES)

    # Both compiles make the system text "Answer as the billing desk.", and each task sends only it.
    system_a = prompt.compile(team="billing", order="#A")[0]
    system_b = prompt.compile(team="billing", order="#B")[0]
    start_span("a", type="generation", attributes=prompt_attributes_for([system_a])).end()
    start_span("b", type="generation", attributes=prompt_attributes_for([system_b])).end()

    assert {
        span.name: (span.attributes or {})["td.prompt.variables"]
        for span in memory.span_exporter.get_finished_spans()
    } == {
        "a": '{"team":"billing","order":"#A"}',
        "b": '{"team":"billing","order":"#B"}',
    }


def test_a_rollout_with_two_versions_of_the_same_messages_records_the_version_of_each_call(
    server: CaptureServer, make: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    make()
    server.replies = [
        (
            200,
            reply(
                messages=DESK_MESSAGES,
                rollout={
                    "percent": 16,
                    "version": 13,
                    "messages": DESK_MESSAGES,
                    "text": None,
                    "model": {"provider": "openai", "name": "gpt-4o-mini"},
                    "parameters": {},
                    "tools": [],
                    "responseFormat": {"type": "text"},
                },
            ),
        )
    ]
    draws = iter([15, 16])

    def draw(_stop: int) -> int:
        return next(draws)

    monkeypatch.setattr(prompts_module.random, "randrange", draw)
    rollout = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    production = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    from_rollout = rollout.compile(team="billing", order="#4")
    from_production = production.compile(team="billing", order="#4")

    assert prompt_attributes_for(from_rollout)["td.prompt.version"] == 13
    assert prompt_attributes_for(from_production)["td.prompt.version"] == 12


def test_the_newest_compiled_message_wins_also_after_170_messages_of_history(
    make: MakeClient,
) -> None:
    make()
    prompt = chat_prompt(
        {"role": "system", "content": "Answer as the {{team}} desk."},
        {"role": "placeholder", "name": "history"},
        {"role": "user", "content": "{{question}}"},
    )
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"message {index}"}
        for index in range(170)
    ]

    first_turn = prompt.compile(team="billing", question="Where is #4?", history=history)
    second_turn = prompt.compile(team="returns", question="Can I return it?")
    chat = [*first_turn, {"role": "assistant", "content": "On its way."}, *second_turn[1:]]

    assert '"team":"billing"' in str(prompt_attributes_for(first_turn)["td.prompt.variables"])
    assert '"team":"billing"' in str(
        prompt_attributes_for([first_turn[0], *history])["td.prompt.variables"]
    )
    assert (
        prompt_attributes_for(chat)["td.prompt.variables"]
        == '{"team":"returns","question":"Can I return it?"}'
    )


def test_a_message_that_a_placeholder_brings_from_an_older_compile_does_not_win(
    make: MakeClient,
) -> None:
    make()
    router = chat_prompt({"role": "user", "content": "{{question}}"}, name="router")
    answer = chat_prompt(
        {"role": "system", "content": "Answer for {{team}}."},
        {"role": "placeholder", "name": "history"},
        name="answer",
    )
    chat = chat_prompt({"role": "user", "content": "{{question}}"}, name="chat")
    summary = chat_prompt(
        {"role": "system", "content": "Summarize the chat."},
        {"role": "placeholder", "name": "history"},
        name="summary",
    )

    question = router.compile(question="Where is #4?")
    routed = answer.compile(team="billing", history=question)
    # The last 1000 messages are all from chat; the summary message is the first.
    conversation = [chat.compile(question=f"question {index}")[0] for index in range(1500)]
    summarized = summary.compile(history=conversation)

    assert prompt_attributes_for(routed) == {
        "td.prompt.name": "answer",
        "td.prompt.version": 12,
        "td.prompt.label": "production",
        "td.prompt.variables": (
            '{"team":"billing","history":[{"role":"user","content":"Where is #4?"}]}'
        ),
    }
    assert prompt_attributes_for(summarized)["td.prompt.name"] == "summary"


def test_prompt_attributes_for_cuts_variables_past_the_attribute_size_limit(
    make: MakeClient,
) -> None:
    make(max_attribute_length=60)
    compiled = chat_prompt(*DESK_MESSAGES).compile(team="billing", order="#" * 100)

    assert prompt_attributes_for(compiled)["td.prompt.variables"] == (
        '{"team":"billing","order":"' + "#" * 17 + '...[truncated]"}'
    )


def test_prompt_attributes_for_looks_at_the_first_and_the_last_1000_messages(
    make: MakeClient,
) -> None:
    make()
    system = chat_prompt(*DESK_MESSAGES).compile(team="billing", order="#4")[0]

    def around(before: int, after: int) -> list[dict[str, Any]]:
        return [*[{"role": "user"}] * before, system, *[{"role": "user"}] * after]

    assert prompt_attributes_for(around(1000, 999)) != {}
    assert prompt_attributes_for(around(999, 1001)) != {}
    assert prompt_attributes_for(tuple(around(1001, 999))) != {}
    assert prompt_attributes_for(around(1000, 1000)) == {}


def test_compile_records_its_variables_not_later_changes_of_the_mapping_or_of_its_lists(
    make: MakeClient,
) -> None:
    memory = make()
    prompt = chat_prompt(
        {"role": "placeholder", "name": "history"},
        {"role": "user", "content": "Order {{order}}"},
    )
    history = [{"role": "user", "content": "Hi"}]
    variables: dict[str, Any] = {"order": "#A", "history": history}

    first = prompt.compile(variables)
    variables["order"] = "#B"
    history.append({"role": "assistant", "content": "Hello"})
    second = prompt.compile(variables)
    variables["order"] = "#C"
    history.append({"role": "user", "content": "Added after compile"})
    with with_prompt(prompt):
        start_span("scope", type="generation").end()

    second_variables = (
        '{"order":"#B","history":[{"role":"user","content":"Hi"},'
        '{"role":"assistant","content":"Hello"}]}'
    )
    assert (
        prompt_attributes_for(first)["td.prompt.variables"]
        == '{"order":"#A","history":[{"role":"user","content":"Hi"}]}'
    )
    assert prompt_attributes_for(second)["td.prompt.variables"] == second_variables
    [span] = memory.span_exporter.get_finished_spans()
    assert (span.attributes or {})["td.prompt.variables"] == second_variables


def test_a_compiled_message_rebuilds_like_a_dict_and_the_rebuilt_one_has_no_tag(
    make: MakeClient,
) -> None:
    make()
    compiled = chat_prompt(*DESK_MESSAGES).compile(team="billing", order="#4")

    @dataclasses.dataclass
    class Conversation:
        messages: list[dict[str, Any]]

    rebuilt = dataclasses.asdict(Conversation(messages=compiled))["messages"]
    same_type = [type(message)(message) for message in compiled]
    plain = [
        {"role": "system", "content": "Answer as the billing desk."},
        {"role": "user", "content": "Order #4: where is it?"},
    ]

    assert rebuilt == plain
    assert same_type == plain
    assert prompt_attributes_for(rebuilt) == {}
    assert prompt_attributes_for(same_type) == {}


def test_with_prompt_wins_over_a_compiled_message_of_another_prompt_through_a_parent_and_in_updates(
    make: MakeClient,
) -> None:
    memory = make()
    scoped = chat_prompt({"role": "user", "content": "Hi"}, name="scoped/prompt")
    # Fetched by version, so it has no label that the label of the scope could hide.
    other = dataclasses.replace(
        chat_prompt({"role": "user", "content": "Cancel order {{order}}."}, name="other/prompt"),
        version=3,
        label=None,
    )
    compiled = other.compile(order="#9")

    with with_prompt(scoped):
        start_span("in scope", type="generation", attributes=prompt_attributes_for(compiled)).end()
        parent = otel_context.get_current()
    attributes = prompt_attributes_for(compiled)
    start_span("with parent", type="generation", parent=parent, attributes=attributes).end()
    with with_prompt(scoped):
        start_span("updated", type="generation").update(attributes=attributes).end()
        start_span("ended", type="generation").end(attributes=attributes)
        with start_span("current", type="generation"):
            telemetry_dev.update_current_span(attributes=attributes)

    assert attributes["td.prompt.name"] == "other/prompt"
    spans = memory.span_exporter.get_finished_spans()
    assert sorted(span.name for span in spans) == [
        "current",
        "ended",
        "in scope",
        "updated",
        "with parent",
    ]
    for span in spans:
        assert {
            key: value
            for key, value in dict(span.attributes or {}).items()
            if key.startswith("td.prompt.")
        } == {
            "td.prompt.name": "scoped/prompt",
            "td.prompt.version": 12,
            "td.prompt.label": "production",
        }


def test_prompt_variables_follow_the_input_capture_of_the_client_and_of_the_span(
    make: MakeClient,
) -> None:
    compiled = chat_prompt({"role": "user", "content": "My card is {{card}}."}).compile(
        card="4111 1111"
    )

    make(capture_input=False)
    assert prompt_attributes_for(compiled) == {
        "td.prompt.name": "support/reply",
        "td.prompt.version": 12,
        "td.prompt.label": "production",
    }

    telemetry_dev.shutdown()

    def mask(value: Any, context: telemetry_dev.MaskContext) -> Any:
        return {"card": "[masked]"} if context.key == "td.prompt.variables" else value

    memory = make(mask=mask)
    attributes = prompt_attributes_for(compiled)
    assert attributes["td.prompt.variables"] == '{"card":"[masked]"}'

    start_span("private", type="generation", capture_input=False, attributes=attributes).end()
    with start_span("updated", type="generation", capture_input=False):
        telemetry_dev.update_current_span(attributes=attributes)

    for span in memory.span_exporter.get_finished_spans():
        assert (span.attributes or {})["td.prompt.name"] == "support/reply"
        assert "td.prompt.variables" not in (span.attributes or {})


def test_prompt_attributes_for_runs_no_code_of_the_app(make: MakeClient) -> None:
    make()
    [message] = chat_prompt({"role": "user", "content": "Refund order {{order}}."}).compile(
        order="#3"
    )
    reads: list[str] = []

    class Messages(list[object]):
        def __getitem__(self, index: Any) -> Any:
            reads.append("__getitem__")
            return super().__getitem__(index)

        def __iter__(self) -> Any:
            reads.append("__iter__")
            return super().__iter__()

        def __reversed__(self) -> Any:
            reads.append("__reversed__")
            return super().__reversed__()

    assert prompt_attributes_for(Messages([message])) == {}
    assert reads == []


RAG_MESSAGES = [
    {"role": "system", "content": "Answer from {{documents}}."},
    {"role": "user", "content": "{{question}}"},
]


class Document(dict[str, str]):
    """A dict that a weak reference can point to."""


def test_the_variables_of_a_compile_are_freed_with_its_messages(make: MakeClient) -> None:
    make()
    prompt = chat_prompt(*RAG_MESSAGES)

    document = Document(content="The refund policy")
    freed = weakref.ref(document)
    messages = prompt.compile(documents=[document], question="q")
    del document
    # with_prompt keeps the variables of the last compile of a prompt object.
    prompt.compile(documents=[], question="other")
    gc.collect()
    assert freed() is not None

    assert len(messages) == 2
    del messages
    gc.collect()
    assert freed() is None


def test_with_input_capture_off_compiled_messages_keep_no_variables(make: MakeClient) -> None:
    make(capture_input=False)
    prompt = chat_prompt(*RAG_MESSAGES)
    documents: list[weakref.ref[Document]] = []
    questions: list[dict[str, Any]] = []

    for turn in range(200):
        document = Document(content=f"document {turn}")
        documents.append(weakref.ref(document))
        questions.append(prompt.compile(documents=[document], question=f"question {turn}")[-1])
    gc.collect()

    # with_prompt keeps the variables of the last compile of a prompt object.
    assert sum(1 for document in documents if document() is not None) == 1

    telemetry_dev.shutdown()
    make()
    assert prompt_attributes_for(questions) == {
        "td.prompt.name": "support/reply",
        "td.prompt.version": 12,
        "td.prompt.label": "production",
    }


CHAT_MESSAGES = (
    {"role": "system", "content": "Answer as the {{team}} desk."},
    {"role": "placeholder", "name": "history"},
    {"role": "user", "content": "{{question}}"},
)


@dataclasses.dataclass
class ChatPrompts:
    chat: ChatPrompt
    question: ChatPrompt
    rag: ChatPrompt
    reply: ChatPrompt


def _answer(turn: int) -> dict[str, Any]:
    return {"role": "assistant", "content": f"answer {turn}"}


# What a row gives for one turn of a chat: the request, the history for the next turn, and the kept
# message. The row puts `marker` in the variables of the compile that made the kept message.
ChatTurn: TypeAlias = tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]


def _new_list(
    p: ChatPrompts, history: list[dict[str, Any]], turn: int, marker: Document
) -> ChatTurn:
    request = p.chat.compile(
        team="billing", history=history, question=f"question {turn}", marker=marker
    )
    return request, [*history, request[-1], _answer(turn)], request[-1]


def _new_list_of_20(
    p: ChatPrompts, history: list[dict[str, Any]], turn: int, marker: Document
) -> ChatTurn:
    request = p.chat.compile(
        team="billing", history=history, question=f"question {turn}", marker=marker
    )
    return request, [*history, request[-1], _answer(turn)][-20:], request[-1]


def _full_list_as_copies(
    p: ChatPrompts, history: list[dict[str, Any]], turn: int, marker: Document
) -> ChatTurn:
    request = p.chat.compile(
        team="billing",
        history=[dict(message) for message in history],
        question=f"question {turn}",
        marker=marker,
    )
    return request, [*history, request[-1], _answer(turn)], request[-1]


def _question_prompt_in_a_placeholder(
    p: ChatPrompts, history: list[dict[str, Any]], turn: int, marker: Document
) -> ChatTurn:
    [kept] = p.question.compile(
        documents=f"document {turn}", question=f"question {turn}", marker=marker
    )
    request = p.rag.compile(team="billing", history=history, current=[kept])
    return request, [*history, kept, _answer(turn)], kept


def _raw_questions_and_a_reply_prompt(
    p: ChatPrompts, history: list[dict[str, Any]], turn: int, marker: Document
) -> ChatTurn:
    request = p.chat.compile(team="billing", history=history, question=f"question {turn}")
    [kept] = p.reply.compile(answer=f"answer {turn}", marker=marker)
    return request, [*history, {"role": "user", "content": f"question {turn}"}, kept], kept


@pytest.mark.parametrize(
    "run",
    [
        _new_list,
        _new_list_of_20,
        _full_list_as_copies,
        _question_prompt_in_a_placeholder,
        _raw_questions_and_a_reply_prompt,
    ],
    ids=[
        "new-list",
        "new-list-of-20",
        "full-list-as-copies",
        "question-prompt-in-a-placeholder",
        "raw-questions-and-a-reply-prompt",
    ],
)
def test_a_chat_history_keeps_the_variables_of_the_last_5_minutes(
    make: MakeClient,
    clock: list[float],
    run: Callable[[ChatPrompts, list[dict[str, Any]], int, Document], ChatTurn],
) -> None:
    make()
    prompts = ChatPrompts(
        chat=chat_prompt(*CHAT_MESSAGES),
        question=chat_prompt({"role": "user", "content": "Context: {{documents}}\n\n{{question}}"}),
        rag=chat_prompt(
            {"role": "system", "content": "Answer as the {{team}} desk."},
            {"role": "placeholder", "name": "history"},
            {"role": "placeholder", "name": "current"},
        ),
        reply=chat_prompt({"role": "assistant", "content": "{{answer}}"}),
    )
    markers: list[weakref.ref[Document]] = []
    history: list[dict[str, Any]] = []
    request: list[dict[str, Any]] = []
    older: dict[str, Any] = {}

    # One turn in 2 minutes, for 300 turns. An unused variable marks the variables of each turn.
    for turn in range(300):
        if turn > 0:
            clock[0] += 120
        marker = Document(content=f"turn {turn}")
        markers.append(weakref.ref(marker))
        request, history, kept = run(prompts, history, turn, marker)
        # The model call of the turn.
        prompt_attributes_for(request)
        if turn == 296:
            older = kept
    gc.collect()

    # Turns 297, 298, and 299: 4, 2, and 0 minutes old.
    assert sum(1 for marker in markers if marker() is not None) == 3
    assert "question 299" in str(prompt_attributes_for(request)["td.prompt.variables"])
    assert prompt_attributes_for(older) == {
        "td.prompt.name": "support/reply",
        "td.prompt.version": 12,
        "td.prompt.label": "production",
    }


def _variables_of(messages: object) -> object:
    return prompt_attributes_for(messages).get("td.prompt.variables")


REFUND_VARIABLES = '{"documents":"The refund policy","question":"Where is #4?"}'


@dataclasses.dataclass
class LivePrompts:
    answer: ChatPrompt
    guard: ChatPrompt
    judge: ChatPrompt
    other: ChatPrompt
    chat: ChatPrompt
    system: ChatPrompt
    with_system: ChatPrompt


# A row compiles one or more requests and runs other compiles before their calls. It gives the
# variables that each call records and the variables of the compile of its request.
LiveCalls: TypeAlias = list[tuple[object, str]]


def _other_compiles_then_a_guard(p: LivePrompts) -> LiveCalls:
    request = p.answer.compile(documents="The refund policy", question="Where is #4?")
    for index in range(150):
        p.other.compile(q=str(index))
    p.guard.compile(conversation=request)
    return [(_variables_of(request), REFUND_VARIABLES)]


def _a_guard_before_each_agent_step(p: LivePrompts) -> LiveCalls:
    conversation = p.answer.compile(documents="The refund policy", question="Where is #4?")
    for step in range(150):
        p.guard.compile(conversation=conversation)
        conversation.append({"role": "assistant", "content": f"step {step}"})
    return [(_variables_of(conversation), REFUND_VARIABLES)]


def _a_guard_then_a_judge_of_the_guard(p: LivePrompts) -> LiveCalls:
    request = p.answer.compile(documents="The refund policy", question="Where is #4?")
    p.judge.compile(conversation=p.guard.compile(conversation=request))
    return [(_variables_of(request), REFUND_VARIABLES)]


def _turns_in_threads_that_share_a_history(p: LivePrompts) -> LiveCalls:
    history: list[dict[str, Any]] = []
    turns = [threading.Event() for _ in range(4)]
    compiled = threading.Barrier(3, timeout=5)
    recorded: list[object] = [None] * 3

    def turn(index: int) -> None:
        turns[index].wait(5)
        request = p.chat.compile(team="billing", history=history, question=f"q{index + 1}")
        history.append(request[-1])
        turns[index + 1].set()
        # A guard call or a queue: the call comes after the other turns compiled.
        compiled.wait()
        recorded[index] = _variables_of(request)

    threads = [threading.Thread(target=turn, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    turns[0].set()
    for thread in threads:
        thread.join(5)
    return list(
        zip(
            recorded,
            [
                '{"team":"billing","history":[],"question":"q1"}',
                '{"team":"billing","history":[{"role":"user","content":"q1"}],"question":"q2"}',
                '{"team":"billing","history":[{"role":"user","content":"q1"},'
                '{"role":"user","content":"q2"}],"question":"q3"}',
            ],
            strict=True,
        )
    )


def _a_system_message_from_startup_after_a_chat_that_gets_it(p: LivePrompts) -> LiveCalls:
    startup = p.system.compile(product="Acme")
    history: list[dict[str, Any]] = []
    for turn in range(3):
        request = p.with_system.compile(
            system=startup, history=history, question=f"question {turn}"
        )
        prompt_attributes_for(request)
        history = [*history, request[-1], _answer(turn)]
    return [
        (
            _variables_of([*startup, {"role": "user", "content": "Where is #4?"}]),
            '{"product":"Acme"}',
        )
    ]


@pytest.mark.parametrize(
    "run",
    [
        _other_compiles_then_a_guard,
        _a_guard_before_each_agent_step,
        _a_guard_then_a_judge_of_the_guard,
        _turns_in_threads_that_share_a_history,
        _a_system_message_from_startup_after_a_chat_that_gets_it,
    ],
    ids=[
        "150-other-compiles-then-a-guard",
        "a-guard-before-each-of-150-agent-steps",
        "a-guard-then-a-judge-of-the-guard",
        "turns-in-threads-that-share-a-history",
        "a-system-message-from-startup-after-a-chat-that-gets-it",
    ],
)
def test_a_request_keeps_its_variables_until_its_call(
    make: MakeClient, run: Callable[[LivePrompts], LiveCalls]
) -> None:
    make()
    calls = run(
        LivePrompts(
            answer=chat_prompt(*RAG_MESSAGES),
            guard=chat_prompt(
                {"role": "system", "content": "Is this safe?"},
                {"role": "placeholder", "name": "conversation"},
                name="guard",
            ),
            judge=chat_prompt(
                {"role": "system", "content": "Check the answer of the guard."},
                {"role": "placeholder", "name": "conversation"},
                name="judge",
            ),
            other=chat_prompt({"role": "user", "content": "{{q}}"}, name="other"),
            chat=chat_prompt(*CHAT_MESSAGES),
            system=chat_prompt(
                {"role": "system", "content": "You support {{product}}."}, name="system"
            ),
            with_system=chat_prompt(
                {"role": "placeholder", "name": "system"},
                {"role": "placeholder", "name": "history"},
                {"role": "user", "content": "{{question}}"},
            ),
        )
    )

    for recorded, compiled in calls:
        assert recorded == compiled


def test_a_compile_keeps_its_variables_for_5_minutes_after_it_or_after_the_last_call_that_names_it(
    make: MakeClient, clock: list[float]
) -> None:
    make()
    chat = chat_prompt(*CHAT_MESSAGES)
    history: list[dict[str, Any]] = []
    for turn in range(5):
        request = chat.compile(team="billing", history=history, question=f"q{turn}")
        prompt_attributes_for(request)
        history = [*history, request[-1], _answer(turn)]
    # These calls send only messages of the chat, so they name the newest compile among them.
    summary = [*history, {"role": "user", "content": "Summarize the chat."}]
    earlier = history[:-2]
    start = clock[0]

    assert '"question":"q4"' in str(_variables_of(summary))
    assert '"question":"q3"' in str(_variables_of(earlier))

    clock[0] = start + 299.5
    assert '"question":"q4"' in str(_variables_of(summary))
    clock[0] = start + 300
    assert _variables_of(earlier) is None
    # The call 0.5 seconds before moved the deadline of q4.
    assert '"question":"q4"' in str(_variables_of(summary))
    clock[0] = start + 600
    assert prompt_attributes_for(summary) == {
        "td.prompt.name": "support/reply",
        "td.prompt.version": 12,
        "td.prompt.label": "production",
    }


def test_variables_past_the_attribute_size_limit_keep_every_key(
    server: CaptureServer, make: MakeClient
) -> None:
    memory = make(max_attribute_length=200)
    prompt = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
    history = [{"role": "user", "content": 'a "quoted" turn'}] * 20
    prompt.compile(
        customer="Ada", history=history, question="Where is order 42?", note="\U0001f600" * 60
    )

    with with_prompt(prompt):
        start_span("chat", type="generation").end()

    [span] = memory.span_exporter.get_finished_spans()
    attribute = dict(span.attributes or {}).get("td.prompt.variables")
    assert isinstance(attribute, str)
    variables: dict[str, str] = json.loads(attribute)
    assert utf16_length(attribute) <= 200
    assert list(variables) == ["customer", "history", "question", "note"]
    assert variables["customer"] == "Ada"
    assert variables["question"] == "Where is order 42?"
    assert re.fullmatch(r'\[\{"role":"user".*\.\.\.\[truncated\]', variables["history"])
    # The cut keeps no half of a surrogate pair.
    assert re.fullmatch(r"(\U0001f600)*(\.\.\.\[truncated\])?", variables["note"])


def test_a_value_with_no_room_for_its_marker_keeps_the_marker(
    server: CaptureServer, make: MakeClient
) -> None:
    def variables_of(max_attribute_length: int, variables: dict[str, str]) -> object:
        memory = make(max_attribute_length=max_attribute_length)
        prompt = get_prompt("support/reply", api_key="td_live_a", base_url=server.url)
        prompt.compile(**variables)
        with with_prompt(prompt):
            start_span("chat", type="generation").end()
        [span] = memory.span_exporter.get_finished_spans()
        telemetry_dev.shutdown()
        return dict(span.attributes or {}).get("td.prompt.variables")

    # Control characters escape to 6 units each, so the cut text is the marker alone.
    assert variables_of(28, {"a": "\u0001" * 50}) == '{"a":"...[truncated]"}'

    # A share of 12 units holds no marker, so the JSON is cut as a whole.
    three = {"a": "x" * 50, "b": "y" * 50, "c": "z" * 50}
    assert (
        variables_of(60, three) == json.dumps(three, separators=(",", ":"))[:46] + "...[truncated]"
    )

    # A short text of escapes gets a share of the marker's length, so it becomes the marker.
    assert variables_of(22, {"a": "\n" * 8}) == '{"a":"...[truncated]"}'


def test_a_value_with_lone_surrogates_is_cut_by_utf16_units_and_the_span_keeps_it(
    server: CaptureServer, make: MakeClient
) -> None:
    def variables_of(value: str) -> object:
        memory = make(max_attribute_length=50)
        prompt = get_prompt("support/reply", api_key="[REDACTED:api-key]", base_url=server.url)
        prompt.compile(a=value)
        with with_prompt(prompt):
            start_span("chat", type="generation").end()
        [span] = memory.span_exporter.get_finished_spans()
        telemetry_dev.shutdown()
        return dict(span.attributes or {}).get("td.prompt.variables")

    assert variables_of("\ud800" * 100) == '{"a":"...[truncated]"}'
    # A lone surrogate counts as one unit, and its JSON escape as 6.
    assert variables_of("\udc00" + "x" * 100) == '{"a":"\\udc00' + "x" * 22 + '...[truncated]"}'
    # A high surrogate right before the marker goes.
    assert variables_of("x" * 27 + "\ud800" + "y" * 100) == '{"a":"' + "x" * 27 + '...[truncated]"}'


def test_a_failure_to_fit_the_variables_drops_them_and_keeps_the_span(
    server: CaptureServer, make: MakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[BaseException] = []
    memory = make(max_attribute_length=50, on_error=errors.append)

    def broken(text: str, max_len: int) -> str:
        raise ValueError("cannot fit")

    monkeypatch.setattr(prompts_module, "fit_object_values", broken)
    prompt = get_prompt("support/reply", api_key="[REDACTED:api-key]", base_url=server.url)
    prompt.compile(a="x" * 100)
    with with_prompt(prompt):
        start_span("chat", type="generation").end()

    [span] = memory.span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes["td.prompt.name"] == "support/reply"
    assert "td.prompt.variables" not in attributes
    assert [str(error) for error in errors] == ["cannot fit"]


def test_a_fallback_prompt_records_its_name_and_label_but_no_version(
    server: CaptureServer, make: MakeClient
) -> None:
    memory = make()
    server.replies = [api_error(503, "prompts_unavailable", "Prompts are unavailable")]
    prompt = get_prompt(
        "support/reply",
        api_key="td_live_a",
        base_url=server.url,
        fallback=[{"role": "user", "content": "Hi"}],
    )

    with with_prompt(prompt):
        start_span("chat", type="generation").end()

    [span] = memory.span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes["td.prompt.name"] == "support/reply"
    assert attributes["td.prompt.label"] == "production"
    assert "td.prompt.version" not in attributes


async def test_get_prompt_async_fetches_in_a_worker_thread(server: CaptureServer) -> None:
    server.delay = 0.3
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(tick())
    prompt = await get_prompt_async("support/reply", api_key="td_live_a", base_url=server.url)
    ticker.cancel()

    assert prompt.version == 12
    assert query_of(own_requests(server)[0].path)["label"] == "production"
    # The event loop ran while the fetch waited for the slow reply. A fetch on the loop's thread
    # would block it, and the ticker would not run until the fetch ended.
    assert ticks >= 5
