from __future__ import annotations

import asyncio
import contextvars
import copy
import dataclasses
import gc
import itertools
import json
import pickle
import socket
import threading
import time
import urllib.error
import urllib.request
import weakref
from collections.abc import Callable, Generator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, NoReturn
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

from .conftest import MakeClient
from .otlp_capture import CaptureServer, otlp_capture_server


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


@pytest.fixture
def server() -> Generator[CaptureServer]:
    with otlp_capture_server() as capture:
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

    [request] = server.requests
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
    assert server.requests[0].headers["authorization"] == "Bearer td_live_client"

    telemetry_dev.shutdown()
    telemetry_dev.init(api_key="td_live_disabled", enabled=False, disable_atexit=True)
    monkeypatch.setenv("TELEMETRY_DEV_API_KEY", "td_live_env")
    monkeypatch.setenv("TELEMETRY_DEV_BASE_URL", server.url)
    get_prompt("support/reply", label="staging")
    assert server.requests[1].headers["authorization"] == "Bearer td_live_env"


def test_get_prompt_reads_the_host_that_telemetry_goes_to(server: CaptureServer) -> None:
    telemetry_dev.init(
        api_key="td_live_a", base_url=f"{server.url}/", export_mode="immediate", disable_atexit=True
    )
    with telemetry_dev.start_span("work"):
        pass
    telemetry_dev.flush()

    get_prompt("support/reply")

    assert [urlsplit(request.path).path for request in server.requests] == [
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
    assert len(server.requests) == 1

    # After the TTL, calls get the copy at once while one refresh runs.
    server.delay = 0.5
    clock[0] += 1
    assert [version() for _ in range(3)] == [12, 12, 12]
    wait_for(lambda: version() == 13)
    assert len(server.requests) == 2

    # A different key does not share the copy.
    server.delay = 0
    get_prompt("support/reply", api_key="td_live_b", base_url=server.url)
    assert len(server.requests) == 3


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

    assert len(server.requests) == 1
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
    assert len(server.requests) == 2

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
    assert len(server.requests) == 1

    clock[0] += 1
    assert fallback_prompt().is_fallback
    assert len(server.requests) == 2

    # A TTL under 10 seconds bounds the wait too.
    fallback_prompt(cache_ttl_seconds=0)
    assert len(server.requests) == 3


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
    assert len(server.requests) == 1

    clock[0] += 2
    fallback_prompt()
    assert len(server.requests) == 2


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
    assert len(server.requests) == 2

    # The TTL ends after 60 seconds, but the server asked for 300.
    clock[0] += 299
    assert version() == 12
    wait_for(
        lambda: all(
            entry.pending is None
            for entry in prompts_module._cache.values()  # pyright: ignore[reportPrivateUsage]
        )
    )
    assert len(server.requests) == 2

    clock[0] += 1
    version()
    wait_for(lambda: version() == 13)
    assert len(server.requests) == 3


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
    with otlp_capture_server() as other:
        other.replies = [(200, reply())]
        server.replies = [(302, {})]
        server.location = f"{other.url}/v1/prompts?name=support%2Freply&label=production"

        with pytest.raises(PromptError) as raised:
            get_prompt("support/reply", api_key="td_live_a", base_url=server.url)

        assert (raised.value.code, raised.value.status) == ("http_302", 302)
        assert other.requests == []


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


# How a chat keeps the compiled question and the answer of a turn for the next turn.
def _new_list(kept: Any, question: dict[str, Any], answer: dict[str, Any]) -> Any:
    return [*kept, question, answer]


def _list_trimmed_in_place(kept: Any, question: dict[str, Any], answer: dict[str, Any]) -> Any:
    kept.extend((question, answer))
    del kept[:-20]
    return kept


def _new_list_of_20(kept: Any, question: dict[str, Any], answer: dict[str, Any]) -> Any:
    return [*kept, question, answer][-20:]


def _tuple_of_20(kept: Any, question: dict[str, Any], answer: dict[str, Any]) -> Any:
    return (*kept, question, answer)[-20:]


def _last_question(kept: Any, question: dict[str, Any], answer: dict[str, Any]) -> Any:
    return question


@pytest.mark.parametrize(
    ("variable", "first", "keep"),
    [
        ("history", [], _new_list),
        ("history", [], _list_trimmed_in_place),
        ("history", [], _new_list_of_20),
        ("history", (), _tuple_of_20),
        ("previous", None, _last_question),
    ],
    ids=["new-list", "list-trimmed-in-place", "new-list-of-20", "tuple-of-20", "not-in-a-list"],
)
def test_a_chat_history_keeps_the_variables_of_its_last_two_turns(
    make: MakeClient,
    variable: str,
    first: Any,
    keep: Callable[[Any, dict[str, Any], dict[str, Any]], Any],
) -> None:
    make()
    prompt = chat_prompt(
        {"role": "system", "content": "Answer as the {{team}} desk."},
        {"role": "placeholder", "name": "history"},
        {"role": "user", "content": "{{question}}"},
    )
    turns: list[weakref.ref[Document]] = []
    kept = copy.copy(first)
    last: list[dict[str, Any]] = []
    older: dict[str, Any] = {}

    # An unused variable marks the variables of each turn.
    for turn in range(300):
        marker = Document(content=f"turn {turn}")
        turns.append(weakref.ref(marker))
        last = prompt.compile(
            {"team": "billing", "question": f"question {turn}", "marker": marker, variable: kept}
        )
        if turn == 297:
            older = last[-1]
        kept = keep(kept, last[-1], {"role": "assistant", "content": f"answer {turn}"})
    gc.collect()

    assert sum(1 for turn in turns if turn() is not None) == 2
    assert '"question":"question 299"' in str(prompt_attributes_for(last)["td.prompt.variables"])
    assert prompt_attributes_for(older) == {
        "td.prompt.name": "support/reply",
        "td.prompt.version": 12,
        "td.prompt.label": "production",
    }


def test_a_guard_prompt_keeps_the_variables_of_its_request_also_after_other_compiles(
    make: MakeClient,
) -> None:
    make()
    answer = chat_prompt(*RAG_MESSAGES)
    guard = chat_prompt(
        {"role": "system", "content": "Is this safe?"},
        {"role": "placeholder", "name": "conversation"},
        name="guard",
    )
    other = chat_prompt({"role": "user", "content": "{{q}}"}, name="other")
    variables = '{"documents":"The refund policy","question":"Where is #4?"}'

    messages = answer.compile(documents="The refund policy", question="Where is #4?")
    # Other requests of the server compile before this request runs its guard.
    for index in range(150):
        other.compile(q=str(index))
    guard.compile(conversation=messages)
    assert prompt_attributes_for(messages)["td.prompt.variables"] == variables

    # An agent loop runs the guard before each step.
    conversation = list(messages)
    for step in range(150):
        guard.compile(conversation=conversation)
        conversation.append({"role": "assistant", "content": f"step {step}"})
    assert prompt_attributes_for(conversation)["td.prompt.variables"] == variables


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
    assert query_of(server.requests[0].path)["label"] == "production"
    # The event loop ran while the fetch waited for the slow reply. A fetch on the loop's thread
    # would block it, and the ticker would not run until the fetch ended.
    assert ticks >= 5
