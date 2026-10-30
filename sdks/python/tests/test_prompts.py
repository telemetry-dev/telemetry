from __future__ import annotations

import asyncio
import contextvars
import itertools
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Generator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, NoReturn
from urllib.parse import parse_qsl, urlsplit

import pytest
from opentelemetry import trace

import telemetry_dev
import telemetry_dev._prompts as prompts_module
from telemetry_dev import (
    ChatPrompt,
    PromptError,
    get_prompt,
    get_prompt_async,
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
