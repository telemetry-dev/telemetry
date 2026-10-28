from __future__ import annotations

import importlib
import importlib.util
import json
import random
import sys
import types
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import anthropic
import pytest
import telemetry_dev
from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib.bedrock import AnthropicBedrock, AsyncAnthropicBedrock
from anthropic.lib.vertex import AnthropicVertex, AsyncAnthropicVertex
from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
from anthropic.resources.beta.messages import Messages as BetaMessages
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import MessageParam, TextBlock
from anthropic.types.beta import BetaMessageParam
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ConfigDict, Field

import telemetry_dev_anthropic
from telemetry_dev_anthropic import instrument_anthropic, uninstrument_anthropic, wrap_anthropic

if TYPE_CHECKING:
    import httpx2 as httpx
else:
    # anthropic 1.x is built on httpx2 and rejects httpx clients; 0.x uses httpx.
    _sdk_http = importlib.import_module("anthropic._base_client")
    httpx = getattr(_sdk_http, "httpx2", None) or _sdk_http.httpx

SAMPLING_KWARGS = anthropic.__version__.startswith("0.")

SyncHandler = Callable[[httpx.Request], httpx.Response]
AsyncHandler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]

MESSAGES: list[MessageParam] = [{"role": "user", "content": "Say hi"}]
BETA_MESSAGES: list[BetaMessageParam] = [{"role": "user", "content": "Say hi"}]


def only_span(env: SimpleNamespace) -> ReadableSpan:
    spans = env.span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


@pytest.mark.parametrize("async_mode", [False, True])
async def test_response_mapping_failure_returns_provider_result_and_marks_capture(
    make: Any, async_mode: bool
) -> None:
    errors: list[BaseException] = []
    memory = make(on_error=errors.append)

    class MappingFailure:
        @property
        def content(self) -> Any:
            raise RuntimeError("mapping failed")

    provider_result = MappingFailure()

    if async_mode:

        async def async_create(**_kwargs: Any) -> MappingFailure:
            return provider_result

        client: Any = object.__new__(AsyncAnthropic)
        messages: Any = object.__new__(AsyncMessages)
        messages.create = async_create
        client.messages = messages
        result = await wrap_anthropic(client).messages.create(model="claude-sonnet-4-6")
    else:

        def create(**_kwargs: Any) -> MappingFailure:
            return provider_result

        client = object.__new__(Anthropic)
        client.messages = SimpleNamespace(create=create)
        result = wrap_anthropic(client).messages.create(model="claude-sonnet-4-6")

    assert result is provider_result
    assert [str(error) for error in errors] == ["mapping failed"]
    assert attrs(only_span(memory))["telemetry.dev.capture.truncated"] is True


def request_json(request: httpx.Request) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(request.content.decode("utf-8")))


def json_response(data: Mapping[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=data, headers={"Content-Type": "application/json"})


def error_response(status_code: int, message: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"type": "error", "error": {"type": "invalid_request_error", "message": message}},
        headers={"Content-Type": "application/json"},
    )


def named_sse_body(events: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def named_sse_response(events: Sequence[Mapping[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        content=named_sse_body(events),
        headers={"Content-Type": "text/event-stream"},
    )


class FailingSyncByteStream(httpx.SyncByteStream):
    def __init__(self, first_chunk: bytes, message: str) -> None:
        self._first_chunk = first_chunk
        self._message = message

    def __iter__(self) -> Iterator[bytes]:
        yield self._first_chunk
        raise httpx.ReadError(self._message)


class FailingAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, first_chunk: bytes, message: str) -> None:
        self._first_chunk = first_chunk
        self._message = message

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first_chunk
        raise httpx.ReadError(self._message)


def failing_sync_sse_response(events: Sequence[Mapping[str, Any]], message: str) -> httpx.Response:
    return httpx.Response(
        200,
        stream=FailingSyncByteStream(named_sse_body(events), message),
        headers={"Content-Type": "text/event-stream"},
    )


def failing_async_sse_response(events: Sequence[Mapping[str, Any]], message: str) -> httpx.Response:
    return httpx.Response(
        200,
        stream=FailingAsyncByteStream(named_sse_body(events), message),
        headers={"Content-Type": "text/event-stream"},
    )


def sync_client(handler: SyncHandler) -> Anthropic:
    return Anthropic(
        api_key="test",
        base_url="https://api.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def async_client(handler: AsyncHandler) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key="test",
        base_url="https://api.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def wrapped_sync_client(handler: SyncHandler) -> Any:
    return wrap_anthropic(sync_client(handler))


def wrapped_async_client(handler: AsyncHandler) -> Any:
    return wrap_anthropic(async_client(handler))


def message_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "Telemetry works."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
            "output_tokens_details": {"thinking_tokens": 1},
        },
    }
    payload.update(overrides)
    return payload


def stream_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": message_payload(
                id="msg_stream", content=[], usage={"input_tokens": 5, "output_tokens": 0}
            ),
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


def oversized_stream_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": message_payload(id="msg_oversized", content=[]),
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        *[
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "x" * 100},
            }
            for _ in range(1100)
        ],
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]


def tool_stream_events() -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "message": message_payload(id="msg_tool_stream", content=[])},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"unit": "celsius"},
                "caller": {"type": "direct"},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"loc'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 'ation":"Paris"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_2", "name": "empty_tool"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
    ]


def thinking_stream_events() -> list[dict[str, Any]]:
    return [
        {"type": "message_start", "message": message_payload(id="msg_thinking", content=[])},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "I should answer."},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]


def citation_stream_events() -> list[dict[str, Any]]:
    citation = {
        "type": "char_location",
        "cited_text": "quoted text",
        "document_index": 0,
        "document_title": "Source",
        "start_char_index": 0,
        "end_char_index": 11,
    }
    return [
        {"type": "message_start", "message": message_payload(id="msg_cited", content=[])},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": citation},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


@pytest.mark.parametrize(
    ("start", "delta", "expected_input"),
    [
        (
            {
                "input_tokens": 40,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 200,
            },
            {"output_tokens": 12},
            40 + 1000 + 200,
        ),
        (
            {
                "input_tokens": 40,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 200,
            },
            {
                "input_tokens": 40,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 200,
                "output_tokens": 12,
            },
            40 + 1000 + 200,
        ),
        (
            {
                "input_tokens": 40,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 200,
            },
            {"input_tokens": 40, "output_tokens": 12},
            40 + 1000 + 200,
        ),
        ({"cache_read_input_tokens": 1000}, {"output_tokens": 12}, None),
    ],
    ids=[
        "cache_on_message_start",
        "delta_repeats_cumulative_counts",
        "delta_repeats_input_only",
        "no_input_count",
    ],
)
def test_streamed_input_tokens_include_cache_reads_and_writes(
    memory: SimpleNamespace,
    start: dict[str, Any],
    delta: dict[str, Any],
    expected_input: int | None,
) -> None:
    events = stream_events()
    events[0]["message"]["usage"] = start
    events[-2]["usage"] = delta

    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(events)

    stream = wrapped_sync_client(handler).messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    for _event in stream:
        pass

    a = attrs(only_span(memory))
    assert a.get("gen_ai.usage.input_tokens") == expected_input
    assert a["gen_ai.usage.output_tokens"] == 12
    assert a["gen_ai.usage.cache_read.input_tokens"] == 1000


def test_create_maps_native_messages_system_params_usage_finish_and_provider(
    memory: SimpleNamespace,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return json_response(message_payload())

    client = wrapped_sync_client(handler)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=MESSAGES,
        system="Be terse",
        stop_sequences=["END"],
        extra_body={"temperature": 0.2, "top_p": 0.9, "top_k": 40},
    )

    assert response.id == "msg_123"
    assert requests[0]["messages"] == MESSAGES
    span = only_span(memory)
    assert span.name == "chat claude-sonnet-4-6"
    a = attrs(span)
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "anthropic"
    assert a["gen_ai.request.model"] == "claude-sonnet-4-6"
    assert a["gen_ai.response.model"] == "claude-sonnet-4-6"
    assert a["gen_ai.response.id"] == "msg_123"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["end_turn"]
    assert a["gen_ai.request.temperature"] == 0.2
    assert a["gen_ai.request.top_p"] == 0.9
    assert a["gen_ai.request.top_k"] == 40
    assert a["gen_ai.request.max_tokens"] == 64
    assert list(cast(Any, a["gen_ai.request.stop_sequences"])) == ["END"]
    assert a["gen_ai.usage.input_tokens"] == 11 + 3 + 2
    assert a["gen_ai.usage.output_tokens"] == 7
    assert a["gen_ai.usage.cache_creation.input_tokens"] == 2
    assert a["gen_ai.usage.cache_read.input_tokens"] == 3
    assert a["gen_ai.usage.reasoning.output_tokens"] == 1
    assert "gen_ai.usage.total_tokens" not in a
    assert json.loads(str(a["gen_ai.input.messages"])) == MESSAGES
    assert a["gen_ai.system_instructions"] == "Be terse"
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Telemetry works."}]}
    ]


def test_create_records_only_the_fields_the_request_sends(memory: SimpleNamespace) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return json_response(message_payload())

    wrapped_sync_client(handler).messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=MESSAGES,
        system="Be terse",
        stop_sequences=anthropic.omit,
        extra_body={"max_tokens": anthropic.NOT_GIVEN, "system": anthropic.omit},
    )

    a = attrs(only_span(memory))
    assert requests[0]["max_tokens"] == a["gen_ai.request.max_tokens"] == 64
    assert "system" not in requests[0]
    assert "gen_ai.system_instructions" not in a
    assert "stop_sequences" not in requests[0]
    assert "gen_ai.request.stop_sequences" not in a


@pytest.mark.skipif(not SAMPLING_KWARGS, reason="anthropic 1.x removed sampling kwargs")
def test_create_records_the_sampling_values_the_request_sends(memory: SimpleNamespace) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return json_response(message_payload())

    messages = wrapped_sync_client(handler).messages
    messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=MESSAGES,
        temperature=0.1,
        top_p=0.9,
        extra_body={"temperature": 0.2},
    )

    a = attrs(only_span(memory))
    assert (requests[0]["temperature"], requests[0]["top_p"]) == (0.2, 0.9)
    assert (a["gen_ai.request.temperature"], a["gen_ai.request.top_p"]) == (0.2, 0.9)


def test_create_serializes_pydantic_request_content_blocks(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload())

    client = wrapped_sync_client(handler)
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": [TextBlock(type="text", text="Say hi")]}],
        system=[TextBlock(type="text", text="Be terse")],
    )

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.input.messages"])) == [
        {"role": "user", "content": [{"type": "text", "text": "Say hi"}]}
    ]
    assert json.loads(str(a["gen_ai.system_instructions"])) == [
        {"type": "text", "text": "Be terse"}
    ]


def test_create_normalizes_iterable_messages_and_system_before_capture(
    memory: SimpleNamespace,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return json_response(message_payload())

    client = wrapped_sync_client(handler)
    messages = (item for item in MESSAGES)
    system = ({"type": "text", "text": "Be terse"} for _ in range(1))
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=messages,
        system=system,
    )

    assert requests[0]["messages"] == MESSAGES
    assert requests[0]["system"] == [{"type": "text", "text": "Be terse"}]
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.input.messages"])) == MESSAGES
    assert json.loads(str(a["gen_ai.system_instructions"])) == [
        {"type": "text", "text": "Be terse"}
    ]


def test_create_normalizes_nested_iterable_content_before_capture(
    memory: SimpleNamespace,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return json_response(message_payload())

    client = wrapped_sync_client(handler)
    content = ({"type": "text", "text": "Say hi"} for _ in range(1))
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": content}],
    )

    expected = [{"role": "user", "content": [{"type": "text", "text": "Say hi"}]}]
    assert requests[0]["messages"] == expected
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.input.messages"])) == expected


def test_tool_use_response_output_is_preserved(memory: SimpleNamespace) -> None:
    tool_use = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "get_weather",
        "input": {"location": "Paris"},
    }
    tool = {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
    }
    tools: list[Any] = [
        tool,
        {"type": "custom", "name": "issue_refund", "input_schema": {"type": "object"}},
        {"type": "web_search_20250305", "name": "web_search"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload(content=[tool_use], stop_reason="tool_use"))

    client = wrapped_sync_client(handler)
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=MESSAGES,
        tools=tools,
    )

    a = attrs(only_span(memory))
    # The dashboard lists these names as the tools offered; schemas stay in the captured input.
    assert json.loads(str(a["gen_ai.tool.definitions"])) == [
        {"type": "function", "name": "get_weather"},
        {"type": "function", "name": "issue_refund"},
        {"type": "web_search_20250305", "name": "web_search"},
    ]
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [tool_use]}
    ]
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["tool_use"]
    assert json.loads(str(a["gen_ai.input.messages"])) == {"messages": MESSAGES, "tools": tools}


def test_offered_tool_names_are_recorded_with_capture_off(make: Any) -> None:
    memory = make(capture_input=False, capture_output=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload())

    wrapped_sync_client(handler).messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=MESSAGES,
        tools=[
            {
                "name": "issue_refund",
                "description": "Refund an order",
                "input_schema": {"type": "object", "properties": {"order": {"type": "string"}}},
            }
        ],
    )

    a = attrs(only_span(memory))
    assert a["gen_ai.tool.definitions"] == '[{"type":"function","name":"issue_refund"}]'
    assert "gen_ai.input.messages" not in a
    assert "gen_ai.output.messages" not in a


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic.lib.tools._tool_params") is None,
    reason="anthropic 0.x beta methods do not accept tool objects",
)
def test_beta_create_records_tool_objects_as_the_definitions_it_sends(
    memory: SimpleNamespace,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return json_response(message_payload())

    @anthropic.beta_tool
    def get_weather(city: str) -> str:
        """Look up the current weather for a city."""
        return f"sunny in {city}"

    wrapped_sync_client(handler).beta.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, tools=[get_weather]
    )

    a = attrs(only_span(memory))
    captured = json.loads(str(a["gen_ai.input.messages"]))
    assert captured["tools"] == requests[0]["tools"]
    assert captured["tools"][0]["name"] == "get_weather"
    assert json.loads(str(a["gen_ai.tool.definitions"])) == [
        {"type": "function", "name": "get_weather"}
    ]


def test_create_streaming_preserves_events_and_records_aggregate(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps: list[float] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        assert timestamp_ms is not None
        timestamps.append(timestamp_ms)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request_json(request)["stream"] is True
        return named_sse_response(stream_events())

    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    events = list(stream)

    assert [event.type for event in events] == [event["type"] for event in stream_events()]
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}
    ]
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 2
    assert a["gen_ai.response.id"] == "msg_stream"
    assert a["gen_ai.response.model"] == "claude-sonnet-4-6"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["end_turn"]
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)
    assert len(timestamps) == 2


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("capture_output", [False, True])
async def test_stream_mapping_error_reports_and_still_yields_provider_event(
    make: Any, async_mode: bool, capture_output: bool
) -> None:
    errors: list[BaseException] = []
    memory = make(capture_output=capture_output, on_error=errors.append)

    class BrokenMessage:
        @property
        def id(self) -> str:
            raise RuntimeError("broken message")

    broken = {"type": "message_start", "message": BrokenMessage()}
    # message_stop makes the stream complete, so only the mapping error can flag the span.
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "sensitive"},
        },
        broken,
        broken,
        {"type": "message_stop"},
    ]

    if async_mode:

        async def handler(request: httpx.Request) -> httpx.Response:
            return named_sse_response(stream_events())

        async def source() -> AsyncIterator[Any]:
            for event in events:
                yield event

        client = wrapped_async_client(handler)
        stream = await client.messages.create(
            model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
        )
        stream._inner = source()
        delivered = [event async for event in stream]
        await client.close()
    else:

        def sync_handler(request: httpx.Request) -> httpx.Response:
            return named_sse_response(stream_events())

        client = wrapped_sync_client(sync_handler)
        stream = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
        )
        stream._inner = iter(events)
        delivered = list(stream)
        client.close()

    assert delivered == events
    assert stream._state.blocks == (
        {0: {"type": "text", "text": "sensitive"}} if capture_output else {}
    )
    assert [str(error) for error in errors] == ["broken message"]
    span = only_span(memory)
    assert span.status.status_code == StatusCode.UNSET
    assert attrs(span)["telemetry.dev.capture.truncated"] is True


def test_stream_capture_does_not_invoke_expanding_model_dump() -> None:
    model_dump_calls = 0

    class ExpandingDelta:
        def __init__(self) -> None:
            self.type = "text_delta"
            self.text = "safe"

        def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
            nonlocal model_dump_calls
            model_dump_calls += 1
            return {"type": "text_delta", "text": "safe", "items": list(range(5_000))}

    captured = telemetry_dev_anthropic._bounded_native(ExpandingDelta())  # pyright: ignore[reportPrivateUsage]

    assert captured == {"type": "text_delta", "text": "safe"}
    assert model_dump_calls == 0


def test_streaming_response_helper_preserves_api_response(memory: SimpleNamespace) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return named_sse_response(stream_events())

    client = wrapped_sync_client(handler)
    with client.messages.with_streaming_response.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=MESSAGES,
        stream=True,
    ) as response:
        assert response.__class__.__name__ == "APIResponse"
        assert response.request_id is None

    assert requests[0]["stream"] is True
    assert memory.span_exporter.get_finished_spans() == ()


async def test_async_create_streaming_matches_sync(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps: list[float] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        assert timestamp_ms is not None
        timestamps.append(timestamp_ms)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request_json(request)["stream"] is True
        return named_sse_response(stream_events())

    client = wrapped_async_client(handler)
    stream = await client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    events = [event async for event in stream]
    await client.close()

    assert [event.type for event in events] == [event["type"] for event in stream_events()]
    assert len(timestamps) == 2
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}
    ]
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 2


@pytest.mark.parametrize("async_mode", [False, True])
async def test_stream_bounds_retained_events_without_dropping_chunks(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    source_events = oversized_stream_events()

    if async_mode:

        async def handler(request: httpx.Request) -> httpx.Response:
            return named_sse_response(source_events)

        client = wrapped_async_client(handler)
        stream = await client.messages.create(
            model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
        )
        delivered = [event async for event in stream]
        await client.close()
    else:

        def sync_handler(request: httpx.Request) -> httpx.Response:
            return named_sse_response(source_events)

        client = wrapped_sync_client(sync_handler)
        stream = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
        )
        delivered = list(stream)
    state = stream._state

    assert len(delivered) == len(source_events)
    assert state.budget.truncated is True
    assert state.budget.bytes_used <= state.budget.max_bytes
    assert len(state.blocks[0]["text"]) < 64 * 1024
    assert attrs(only_span(memory))["telemetry.dev.capture.truncated"] is True


def test_stream_revalidates_expanded_tool_json_against_the_item_limit() -> None:
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    record(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "many", "input": {}},
        },
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps([0] * 1001, separators=(",", ":")),
            },
        },
        state,
    )
    record({"type": "message_stop"}, state)

    partial = telemetry_dev_anthropic._stream_partial(state)  # pyright: ignore[reportPrivateUsage]

    # The expanded input is charged while streaming now, so the budget catches the item
    # limit before the final check. The exported output and flag are unchanged.
    assert state.budget.truncated is True
    assert partial["output"] is None
    assert partial["attributes"] == {"telemetry.dev.capture.truncated": True}


def test_stream_rejects_normalized_tool_block_over_budget_but_keeps_metadata(
    memory: SimpleNamespace,
) -> None:
    source_events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": message_payload(
                id="msg_oversized_tool",
                content=[],
                usage={"input_tokens": 5, "output_tokens": 0},
            ),
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_oversized",
                "name": "oversized_tool",
                "input": {},
                # Over the 48 KiB stream bound but under the 64 KiB core cap.
                "oversized": "x" * (49 * 1024),
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 3},
        },
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(source_events)

    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    list(stream)

    assert stream._state.budget.truncated is True
    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 3
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["tool_use"]


def test_create_streaming_preserves_citation_deltas(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(citation_stream_events())

    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    list(stream)

    citation = citation_stream_events()[3]["delta"]["citation"]
    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello", "citations": [citation]}],
        }
    ]


def test_streaming_tool_use_json_fragments_and_empty_input(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(tool_stream_events())

    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    list(stream)

    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"unit": "celsius", "location": "Paris"},
                    "caller": {"type": "direct"},
                },
                {"type": "tool_use", "id": "toolu_2", "name": "empty_tool", "input": {}},
            ],
        }
    ]


def test_streaming_thinking_delta_and_signature(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(thinking_stream_events())

    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    list(stream)

    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "I should answer.", "signature": "sig"}],
        }
    ]


def test_stream_close_ends_partial_span_once(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(stream_events())

    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    iterator = iter(stream)
    assert next(iterator).type == "message_start"
    assert next(iterator).type == "content_block_start"
    assert next(iterator).type == "content_block_delta"
    stream.close()

    span = only_span(memory)
    assert span.status.status_code == StatusCode.UNSET
    a = attrs(span)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
    ]
    assert "gen_ai.response.finish_reasons" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def test_masked_stream_close_omits_partial_output(make: Any) -> None:
    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return value

    memory = make(mask=mask)
    client = wrapped_sync_client(lambda _request: named_sse_response(stream_events()))
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    next(stream)
    next(stream)
    next(stream)
    stream.close()

    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def test_mid_stream_read_error_records_error_span(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return failing_sync_sse_response(stream_events()[:3], "sync stream broke")

    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )

    with pytest.raises(Exception) as exc_info:
        list(stream)

    assert "sync stream broke" in str(exc_info.value)
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == type(exc_info.value).__name__
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
    ]
    assert a["telemetry.dev.capture.truncated"] is True
    assert any(event.name == "exception" for event in span.events)


async def test_async_mid_stream_read_error_records_error_span(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return failing_async_sse_response(stream_events()[:3], "async stream broke")

    client = wrapped_async_client(handler)
    stream = await client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )

    with pytest.raises(Exception) as exc_info:
        _ = [event async for event in stream]
    await client.close()

    assert "async stream broke" in str(exc_info.value)
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == type(exc_info.value).__name__
    assert a["telemetry.dev.capture.truncated"] is True


def test_messages_stream_context_manager_records_helper_span(memory: SimpleNamespace) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return named_sse_response(stream_events())

    client = wrapped_sync_client(handler)
    with client.messages.stream(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    ) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()

    assert text == "Hello world"
    assert final.id == "msg_stream"
    assert requests[0]["stream"] is True
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}
    ]
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 2
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)


async def test_async_messages_stream_context_manager_records_helper_span(
    memory: SimpleNamespace,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(stream_events())

    client = wrapped_async_client(handler)
    async with client.messages.stream(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    ) as stream:
        parts = [text async for text in stream.text_stream]
        final = await stream.get_final_message()
    await client.close()

    assert "".join(parts) == "Hello world"
    assert final.id == "msg_stream"
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}
    ]


def test_messages_stream_enter_failure_records_error(memory: SimpleNamespace) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = error_response(500, "server broke")
        response.headers["retry-after-ms"] = "1"
        return response

    client = wrapped_sync_client(handler)
    with pytest.raises(anthropic.APIStatusError):
        with client.messages.stream(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES):
            pass

    # The SDK retries inside one span.
    assert len(requests) > 1
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_create_api_error_records_error_span(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response(400, "bad model")

    client = wrapped_sync_client(handler)
    with pytest.raises(Exception) as exc_info:
        client.messages.create(model="claude-bad", max_tokens=64, messages=MESSAGES)

    assert "bad model" in str(exc_info.value)
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


async def test_async_create_api_error_records_error_span(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return error_response(400, "bad model")

    client = wrapped_async_client(handler)
    with pytest.raises(Exception) as exc_info:
        await client.messages.create(model="claude-bad", max_tokens=64, messages=MESSAGES)
    await client.close()

    assert "bad model" in str(exc_info.value)
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR


async def test_global_instrumentation_is_idempotent_and_restores_originals(
    memory: SimpleNamespace,
) -> None:
    originals = (Messages.create, Messages.stream, AsyncMessages.create, AsyncMessages.stream)
    responses = [
        json_response(message_payload()),
        named_sse_response(stream_events()),
        json_response(message_payload(id="msg_async")),
        named_sse_response(stream_events()),
        json_response(message_payload(id="msg_restored")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return handler(request)

    instrument_anthropic()
    instrumented_create = Messages.create
    assert instrumented_create is not originals[0]
    instrument_anthropic()
    assert Messages.create is instrumented_create

    client = sync_client(handler)
    client.messages.create(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES)
    with client.messages.stream(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    ) as stream:
        list(stream)

    async_client_instance = async_client(async_handler)
    await async_client_instance.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    )
    async with async_client_instance.messages.stream(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    ) as stream:
        _ = [event async for event in stream]
    await async_client_instance.close()
    assert len(memory.span_exporter.get_finished_spans()) == 4

    uninstrument_anthropic()
    assert (
        Messages.create,
        Messages.stream,
        AsyncMessages.create,
        AsyncMessages.stream,
    ) == originals
    sync_client(handler).messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    )
    assert len(memory.span_exporter.get_finished_spans()) == 4


def test_uninstrument_preserves_method_installed_after_its_wrapper() -> None:
    original = Messages.create

    def later_owner(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)

    try:
        instrument_anthropic()
        Messages.create = later_owner
        uninstrument_anthropic()
        assert Messages.create is later_owner
    finally:
        Messages.create = original


def test_uninstrument_preserves_deleted_method_and_restores_other_wrappers() -> None:
    originals = (Messages.create, Messages.stream, AsyncMessages.create, AsyncMessages.stream)

    try:
        instrument_anthropic()
        del Messages.create
        uninstrument_anthropic()

        assert "create" not in vars(Messages)
        assert (Messages.stream, AsyncMessages.create, AsyncMessages.stream) == originals[1:]

        Messages.create = originals[0]
        instrument_anthropic()
        assert Messages.create is not originals[0]
        uninstrument_anthropic()
        assert (
            Messages.create,
            Messages.stream,
            AsyncMessages.create,
            AsyncMessages.stream,
        ) == originals
    finally:
        Messages.create = originals[0]
        uninstrument_anthropic()


def test_wrap_anthropic_is_idempotent(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload())

    client = sync_client(handler)
    wrapped_once = wrap_anthropic(client)
    wrapped_create = wrapped_once.messages.create
    wrapped_twice = wrap_anthropic(wrapped_once)

    assert wrapped_twice is client
    assert client.messages.create is wrapped_create
    client.messages.create(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES)
    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_wrap_anthropic_keeps_client_wrapped_after_global_uninstrument(
    memory: SimpleNamespace,
) -> None:
    responses = [json_response(message_payload(id="msg_create"))]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    instrument_anthropic()
    client = wrap_anthropic(sync_client(handler))
    uninstrument_anthropic()
    client.messages.create(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES)

    assert len(memory.span_exporter.get_finished_spans()) == 1


@pytest.mark.parametrize(
    ("client_class", "subclass", "expected"),
    [
        ("AnthropicBedrockMantle", False, "aws.bedrock"),
        ("AsyncAnthropicBedrockMantle", False, "aws.bedrock"),
        ("AnthropicBedrockMantle", True, "aws.bedrock"),
        ("AnthropicFoundry", False, "anthropic"),
        ("AnthropicAWS", False, "anthropic"),
    ],
)
def test_provider_mapping_for_other_client_classes(
    memory: SimpleNamespace, client_class: str, subclass: bool, expected: str
) -> None:
    base: type[Any] = getattr(anthropic, client_class)
    cls: type[Any] = type("CustomClient", (base,), {}) if subclass else base
    client = object.__new__(cls)

    def create(**_kwargs: Any) -> dict[str, Any]:
        return message_payload(id="msg_provider")

    def stream_stub(**_kwargs: Any) -> None:
        return None

    client.messages = SimpleNamespace(create=create, stream=stream_stub)

    wrap_anthropic(client).messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    )

    assert attrs(only_span(memory))["gen_ai.provider.name"] == expected


def test_beta_create_records_generation_span(memory: SimpleNamespace) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return json_response(message_payload(id="msg_beta"))

    client = wrapped_sync_client(handler)
    client.beta.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=MESSAGES,
        betas=["context-management-2025-06-27"],
    )

    assert requests[0].url.params.get("beta") == "true"
    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "anthropic"
    assert a["gen_ai.response.id"] == "msg_beta"
    assert a["gen_ai.usage.input_tokens"] == 11 + 3 + 2
    assert a["gen_ai.response.finish_reasons"] == ("end_turn",)


@pytest.mark.parametrize("namespace", ["stable", "beta"])
def test_parse_records_one_generation_span(memory: SimpleNamespace, namespace: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload(id=f"msg_parse_{namespace}"))

    client = wrapped_sync_client(handler)
    messages = client.messages if namespace == "stable" else client.beta.messages
    parsed = messages.parse(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES)

    assert parsed.id == f"msg_parse_{namespace}"
    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == f"msg_parse_{namespace}"
    assert a["gen_ai.usage.output_tokens"] == 7


@pytest.mark.parametrize("namespace", ["stable", "beta"])
async def test_async_parse_records_one_generation_span(
    memory: SimpleNamespace, namespace: str
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload(id=f"msg_async_parse_{namespace}"))

    client = wrapped_async_client(handler)
    messages = client.messages if namespace == "stable" else client.beta.messages
    await messages.parse(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES)
    await client.close()

    assert attrs(only_span(memory))["gen_ai.response.id"] == f"msg_async_parse_{namespace}"


def test_beta_tool_runner_records_one_generation_span_per_turn(memory: SimpleNamespace) -> None:
    responses = [
        json_response(
            message_payload(
                id="msg_tool_turn",
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "weather",
                        "input": {"city": "Accra"},
                    }
                ],
                stop_reason="tool_use",
            )
        ),
        json_response(message_payload(id="msg_final_turn")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    calls: list[str] = []

    @anthropic.beta_tool
    def weather(city: str) -> str:
        """Look up the weather.

        Args:
            city: City name.
        """
        calls.append(city)
        return "sunny"

    client = wrapped_sync_client(handler)
    final = client.beta.messages.tool_runner(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, tools=[weather]
    ).until_done()

    assert final.id == "msg_final_turn"
    assert calls == ["Accra"]
    spans = memory.span_exporter.get_finished_spans()
    assert [attrs(span)["gen_ai.response.id"] for span in spans] == [
        "msg_tool_turn",
        "msg_final_turn",
    ]


async def test_async_beta_tool_runner_records_one_generation_span_per_turn(
    memory: SimpleNamespace,
) -> None:
    responses = [
        json_response(
            message_payload(
                id="msg_async_tool_turn",
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "weather",
                        "input": {"city": "Accra"},
                    }
                ],
                stop_reason="tool_use",
            )
        ),
        json_response(message_payload(id="msg_async_final_turn")),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    calls: list[str] = []

    @anthropic.beta_async_tool
    async def weather(city: str) -> str:
        calls.append(city)
        return "sunny"

    client = wrapped_async_client(handler)
    final = await client.beta.messages.tool_runner(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, tools=[weather]
    ).until_done()
    await client.close()

    assert final.id == "msg_async_final_turn"
    assert calls == ["Accra"]
    assert [
        attrs(span)["gen_ai.response.id"] for span in memory.span_exporter.get_finished_spans()
    ] == [
        "msg_async_tool_turn",
        "msg_async_final_turn",
    ]


def test_global_instrumentation_covers_beta_and_parse_and_restores(
    memory: SimpleNamespace,
) -> None:
    patched = [
        (cls, name)
        for cls in (Messages, AsyncMessages, BetaMessages, AsyncBetaMessages)
        for name in ("create", "stream", "parse")
    ]
    originals = [getattr(cls, name) for cls, name in patched]

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload())

    instrument_anthropic()
    assert all(
        getattr(cls, name) is not original
        for (cls, name), original in zip(patched, originals, strict=True)
    )
    client = sync_client(handler)
    client.beta.messages.create(model="claude-sonnet-4-6", max_tokens=64, messages=BETA_MESSAGES)
    client.messages.parse(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES)
    assert len(memory.span_exporter.get_finished_spans()) == 2

    uninstrument_anthropic()
    assert [getattr(cls, name) for cls, name in patched] == originals
    client.beta.messages.create(model="claude-sonnet-4-6", max_tokens=64, messages=BETA_MESSAGES)
    assert len(memory.span_exporter.get_finished_spans()) == 2


def test_global_and_wrapped_beta_record_one_span_per_call(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload())

    instrument_anthropic()
    client = wrapped_sync_client(handler)
    client.beta.messages.create(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES)
    client.beta.messages.parse(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES)

    assert len(memory.span_exporter.get_finished_spans()) == 2


def no_bedrock_signing(**_kwargs: object) -> dict[str, str]:
    return {}


def provider_clients(
    monkeypatch: pytest.MonkeyPatch, handler: AsyncHandler, sync_handler: SyncHandler
) -> dict[str, tuple[Any, Any]]:
    # Bedrock signs requests with botocore; the resource classes under test do not need it.
    monkeypatch.setattr("anthropic.lib.bedrock._auth.get_auth_headers", no_bedrock_signing)

    def sync_http() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(sync_handler))

    def async_http() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return {
        "aws.bedrock": (
            AnthropicBedrock(
                aws_region="us-east-1",
                aws_access_key="a",
                aws_secret_key="b",
                http_client=sync_http(),
            ),
            AsyncAnthropicBedrock(
                aws_region="us-east-1",
                aws_access_key="a",
                aws_secret_key="b",
                http_client=async_http(),
            ),
        ),
        "gcp.vertex_ai": (
            AnthropicVertex(
                region="us-east5", project_id="p", access_token="t", http_client=sync_http()
            ),
            AsyncAnthropicVertex(
                region="us-east5", project_id="p", access_token="t", http_client=async_http()
            ),
        ),
    }


@pytest.mark.parametrize("provider", ["aws.bedrock", "gcp.vertex_ai"])
@pytest.mark.parametrize("mode", ["wrap", "instrument"])
async def test_provider_beta_create_records_response_sync_and_async(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, provider: str, mode: str
) -> None:
    def sync_handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload(id="msg_provider_sync"))

    async def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload(id="msg_provider_async"))

    if mode == "instrument":
        instrument_anthropic()
    sync_provider, async_provider = provider_clients(monkeypatch, handler, sync_handler)[provider]
    if mode == "wrap":
        sync_provider = wrap_anthropic(sync_provider)
        async_provider = wrap_anthropic(async_provider)

    sync_provider.beta.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=BETA_MESSAGES
    )
    await async_provider.beta.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=BETA_MESSAGES
    )
    await async_provider.close()

    spans = memory.span_exporter.get_finished_spans()
    assert [attrs(span)["gen_ai.response.id"] for span in spans] == [
        "msg_provider_sync",
        "msg_provider_async",
    ]
    for span in spans:
        assert attrs(span)["gen_ai.provider.name"] == provider
        assert "gen_ai.output.messages" in attrs(span)


def beta_stream_events(
    *blocks: tuple[dict[str, Any], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": message_payload(
                id="msg_beta_stream",
                model="claude-opus-5",
                content=[],
                usage={"input_tokens": 5, "output_tokens": 0},
            ),
        }
    ]
    for index, (block, deltas) in enumerate(blocks):
        events.append({"type": "content_block_start", "index": index, "content_block": block})
        events.extend(
            {"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas
        )
        events.append({"type": "content_block_stop", "index": index})
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]
    )
    return events


def streamed_beta_span(events: list[dict[str, Any]], memory: SimpleNamespace) -> dict[str, object]:
    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(events)

    client = wrapped_sync_client(handler)
    for _event in client.beta.messages.create(
        model="claude-opus-5", max_tokens=64, messages=BETA_MESSAGES, stream=True
    ):
        pass
    return attrs(only_span(memory))


@pytest.mark.parametrize("masked", [False, True])
def test_stream_output_over_the_encoded_limit_keeps_a_bounded_prefix(
    make: Any, masked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 1_000)

    def identity_mask(value: Any, _context: Any) -> Any:
        return value

    memory = make(mask=identity_mask) if masked else make()
    events = beta_stream_events(
        ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "first"}]),
        (
            {"type": "text", "text": ""},
            [{"type": "text_delta", "text": "ab"}, {"type": "text_delta", "text": '"' * 500}],
        ),
    )

    a = streamed_beta_span(events, memory)

    assert a["telemetry.dev.capture.truncated"] is True
    if masked:
        assert "gen_ai.output.messages" not in a
    else:
        content = json.loads(str(a["gen_ai.output.messages"]))[0]["content"]
        assert content == [{"type": "text", "text": "first"}, {"type": "text", "text": "ab"}]


@pytest.mark.parametrize(("stream_limit", "complete"), [(48 * 1024, True), (1_000, False)])
def test_many_small_deltas_do_not_exhaust_the_item_budget(
    make: Any, monkeypatch: pytest.MonkeyPatch, stream_limit: int, complete: bool
) -> None:
    def identity_mask(value: Any, _context: Any) -> Any:
        return value

    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", stream_limit)
    memory = make(mask=identity_mask)
    deltas = [{"type": "text_delta", "text": "Hello wor"} for _ in range(400)]
    events = beta_stream_events(({"type": "text", "text": ""}, deltas))

    a = streamed_beta_span(events, memory)

    if complete:
        assert "telemetry.dev.capture.truncated" not in a
        assert json.loads(str(a["gen_ai.output.messages"])) == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello wor" * 400}]}
        ]
    else:
        assert a["telemetry.dev.capture.truncated"] is True
        assert "gen_ai.output.messages" not in a


def test_appended_deltas_are_rejected_before_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    measured: list[int] = []
    original_dumps = json.dumps

    def counting_dumps(value: Any, *args: Any, **kwargs: Any) -> str:
        if isinstance(value, str):
            measured.append(len(value))
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(implementation.json, "dumps", counting_dumps)
    state = implementation._StreamState()
    oversized = "x" * (state.budget.remaining_bytes + 1)

    assert implementation._reserve_appended_text(0, oversized, state) is False
    assert state.budget.truncated is True
    assert implementation._reserve_appended_text(0, "x", state) is False
    assert measured == []

    fitting = implementation._StreamState()
    assert implementation._reserve_appended_text(0, "é", fitting) is True
    assert fitting.budget.bytes_used == 2
    assert fitting.block_reservations[0] == (2, 0)
    assert measured == []

    escaped = implementation._StreamState()
    assert implementation._reserve_appended_text(0, '"\n', escaped) is True
    assert escaped.budget.bytes_used == 4


def test_deltas_that_fill_the_stream_limit_keep_the_text_that_was_accepted(
    memory: SimpleNamespace,
) -> None:
    events = beta_stream_events(
        (
            {"type": "text", "text": ""},
            [
                {"type": "text_delta", "text": "x" * 49_000},
                {"type": "text_delta", "text": "y" * 127},
            ],
        )
    )

    a = streamed_beta_span(events, memory)

    output = json.loads(str(a["gen_ai.output.messages"]))
    assert a["telemetry.dev.capture.truncated"] is True
    assert output[0]["content"][0]["text"] == "x" * 49_000


@pytest.mark.parametrize(("deltas", "complete"), [(200, True), (300, False)])
def test_escape_heavy_deltas_are_charged_at_their_serialized_size(
    memory: SimpleNamespace, deltas: int, complete: bool
) -> None:
    chunk = '"\n' * 50
    events = beta_stream_events(
        ({"type": "text", "text": ""}, [{"type": "text_delta", "text": chunk}] * deltas)
    )

    a = streamed_beta_span(events, memory)

    text = json.loads(str(a["gen_ai.output.messages"]))[0]["content"][0]["text"]
    if complete:
        assert text == chunk * deltas
        assert "telemetry.dev.capture.truncated" not in a
    else:
        assert text
        assert (chunk * deltas).startswith(text)
        assert a["telemetry.dev.capture.truncated"] is True


def test_block_creating_text_delta_also_charges_the_block_it_creates() -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    state = implementation._StreamState()

    implementation._record_stream_event(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        state,
    )

    structure = telemetry_dev.CaptureBudget()
    assert structure.accept({"type": "text", "text": ""})
    assert state.budget.bytes_used >= structure.bytes_used
    assert state.budget.items_used >= structure.items_used


@pytest.mark.parametrize(
    ("tool_block", "deltas"),
    [
        (
            {"type": "tool_use", "id": "t1", "name": "calc"},
            [{"type": "input_json_delta", "partial_json": '{"x":"\\ud800"}'}],
        ),
        (
            {"type": "tool_use", "id": "t1", "name": "calc"},
            [{"type": "input_json_delta", "partial_json": '{"\\ud800":1}'}],
        ),
        ({"type": "tool_use", "id": "t1", "name": "calc", "input": {"\ud800": 1}}, []),
    ],
    ids=["streamed-value", "streamed-key", "start-key"],
)
def test_lone_surrogate_in_tool_input_keeps_the_exported_span_whole(
    memory: SimpleNamespace, tool_block: dict[str, Any], deltas: list[dict[str, Any]]
) -> None:
    events = beta_stream_events(({"type": "text", "text": "keep me"}, []), (tool_block, deltas))

    a = streamed_beta_span(events, memory)

    assert "gen_ai.output.messages" in a
    messages = str(a["gen_ai.output.messages"])
    messages.encode()
    assert "keep me" in messages
    assert "\ud800" not in messages
    assert "\ufffd" in messages


def test_lone_surrogate_in_encrypted_content_keeps_the_span_encodable(
    memory: SimpleNamespace,
) -> None:
    a = streamed_beta_span(
        beta_stream_events(
            (
                {"type": "compaction", "content": None},
                [
                    {
                        "type": "compaction_delta",
                        "content": "Summary.",
                        "encrypted_content": "enc_\ud800",
                    }
                ],
            )
        ),
        memory,
    )

    messages = str(a["gen_ai.output.messages"])
    messages.encode()  # encrypted_content is injected raw, bypassing the bounded capture
    assert "\ud800" not in messages
    assert "\ufffd" in messages


def test_lone_surrogate_in_stream_metadata_keeps_the_span_encodable(
    memory: SimpleNamespace,
) -> None:
    events = beta_stream_events(({"type": "text", "text": "hi"}, []))
    events[0]["message"]["model"] = "claude-\ud800"
    for event in events:
        if event["type"] == "message_delta":
            event["delta"]["stop_reason"] = "end_\ud800"

    a = streamed_beta_span(events, memory)

    model = str(a["gen_ai.response.model"])
    reasons = list(cast(Any, a["gen_ai.response.finish_reasons"]))
    "".join([model, *reasons]).encode()  # the exporter encodes these too
    assert "\ud800" not in model
    assert "\ufffd" in model
    assert all("\ud800" not in reason for reason in reasons)


def test_lone_surrogate_keeps_the_exported_span_whole(memory: SimpleNamespace) -> None:
    events = beta_stream_events(
        ({"type": "text", "text": "keep me"}, []),
        ({"type": "text", "text": "a\ud800"}, []),
    )

    a = streamed_beta_span(events, memory)

    messages = str(a["gen_ai.output.messages"])
    messages.encode()  # the OTLP exporter does this; a lone surrogate would raise
    assert "keep me" in messages
    assert "\ud800" not in messages
    assert "\ufffd" in messages
    assert a["gen_ai.usage.output_tokens"] == 2
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["end_turn"]


def _tool_stream(payloads: list[str], start_input: dict[str, Any] | None = None) -> Any:
    implementation = cast(Any, telemetry_dev_anthropic)
    state = implementation._StreamState()
    record = implementation._record_stream_event
    for index, payload in enumerate(payloads):
        block: dict[str, Any] = {"type": "tool_use", "id": f"t{index}", "name": "calc"}
        if start_input is not None:
            block["input"] = dict(start_input)
        record({"type": "content_block_start", "index": index, "content_block": block}, state)
        record(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": payload},
            },
            state,
        )
        record({"type": "content_block_stop", "index": index}, state)
    return state


@pytest.mark.parametrize(
    ("payload", "start_input"),
    [('{"x":1}' + " " * 40_000, None), ('{"x":1}', {"x": "a" * 40_000})],
    ids=["padded-fragments", "overwritten-start-input"],
)
def test_settled_tool_input_charge_covers_what_is_still_retained(
    payload: str, start_input: dict[str, Any] | None
) -> None:
    state = _tool_stream([payload, payload], start_input)

    retained = sum(len(raw) for raw in state.tool_json.values()) + sum(
        len(json.dumps(block)) for block in state.blocks.values()
    )

    assert retained <= state.budget.bytes_used


@pytest.mark.parametrize(
    ("elements", "truncated"), [(8, False), (1_500, True), (5_000, True), (20_000, True)]
)
def test_settled_tool_input_agrees_with_the_final_bounded_check(
    elements: int, truncated: bool
) -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    state = _tool_stream(['{"n":[' + ",".join(["1"] * elements) + "]}"])
    truncated_while_streaming = state.budget.truncated

    dropped = implementation._stream_partial(state)["output"] is None

    assert truncated_while_streaming is truncated
    assert dropped is truncated


def test_many_small_tool_blocks_are_all_kept() -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    state = _tool_stream(['{"a":1}'] * 60)
    truncated_while_streaming = state.budget.truncated

    output = implementation._stream_partial(state)["output"]

    assert truncated_while_streaming is False
    assert output is not None
    assert len(output[0]["content"]) == 60


@pytest.mark.parametrize(
    "signatures", [["sig"], ["sig", "s" * 60_000]], ids=["fits", "rejected-later"]
)
def test_signature_after_stop_keeps_the_settled_tool_input_charge(signatures: list[str]) -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    # Control characters cost six bytes each once serialized, one in a structural measure.
    state = _tool_stream([json.dumps({"x": "\x01" * 1000})])
    for signature in signatures:
        implementation._record_stream_event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature},
            },
            state,
        )
    charged = state.budget.bytes_used

    output = implementation._stream_partial(state)["output"]

    assert output is not None
    assert charged >= len(json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode())


@pytest.mark.parametrize(
    "signatures", [["sig"], ["sig", "s" * 60_000]], ids=["fits", "rejected-later"]
)
def test_tool_input_after_the_stop_stays_charged_through_signatures(
    signatures: list[str],
) -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    record = implementation._record_stream_event
    state = _tool_stream(['{"x":1}'])
    for _ in range(200):
        deltas = [{"type": "input_json_delta", "partial_json": "\x01" * 1000}] + [
            {"type": "signature_delta", "signature": signature} for signature in signatures
        ]
        for delta in deltas:
            record({"type": "content_block_delta", "index": 0, "delta": delta}, state)
        retained = sum(len(json.dumps(raw)) - 2 for raw in state.tool_json.values()) + sum(
            len(json.dumps(block)) for block in state.blocks.values()
        )
        assert retained <= state.budget.bytes_used

    assert state.budget.truncated is True


@pytest.mark.parametrize(
    "signatures", [["sig"], ["sig", "s" * 60_000]], ids=["accepted", "rejected"]
)
def test_unsettled_tool_input_stays_charged_at_its_escaped_size_through_signatures(
    signatures: list[str],
) -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    record = implementation._record_stream_event
    state = implementation._StreamState()
    raw_input = "\x01" * 1000
    record(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t1", "name": "calc"},
        },
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": raw_input},
        },
        state,
    )
    streamed = state.budget.bytes_used
    for signature in signatures:
        record(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature},
            },
            state,
        )

    # No content_block_stop, so the replacement re-measures the raw input through the core.
    assert 0 not in state.settled_tool_blocks
    assert state.tool_json[0] == raw_input
    escaped_input = len(json.dumps(raw_input)) - 2
    assert escaped_input == 6_000
    assert state.block_reservations[0][0] >= escaped_input + len(json.dumps(state.blocks[0]))
    assert state.budget.bytes_used >= streamed
    if len(signatures) > 1:
        assert "signature" not in state.blocks[0]
    else:
        assert state.blocks[0]["signature"] == "sig"


@pytest.mark.parametrize(
    "replacement",
    [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t1", "name": "calc"},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "compaction", "content": "summary"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": "c" * 60_000},
        },
    ],
    ids=["new-start", "compaction-start", "dropped"],
)
def test_block_after_a_settled_tool_is_charged_like_a_fresh_one(
    replacement: dict[str, Any],
) -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    record = implementation._record_stream_event
    after_settled = _tool_stream(['{"x":1}'])
    fresh = implementation._StreamState()
    for state in (after_settled, fresh):
        record(replacement, state)
        for delta in [
            {"type": "input_json_delta", "partial_json": "a" * 1000},
            {"type": "signature_delta", "signature": "sig"},
        ]:
            record({"type": "content_block_delta", "index": 0, "delta": delta}, state)

    retained = sum(len(raw) for raw in after_settled.tool_json.values()) + sum(
        len(json.dumps(block)) for block in after_settled.blocks.values()
    )
    assert retained <= after_settled.budget.bytes_used
    assert after_settled.block_reservations[0] == fresh.block_reservations[0]


def test_escape_heavy_text_is_charged_the_same_with_or_without_a_start_block() -> None:
    implementation = cast(Any, telemetry_dev_anthropic)

    def retained(with_start: bool) -> str:
        state = implementation._StreamState()
        record = implementation._record_stream_event
        if with_start:
            record(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                state,
            )
        for text in ["\x01" * 2_000] + ["y" * 100] * 500:
            record(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
                state,
            )
        output = implementation._stream_partial(state)["output"]
        return output[0]["content"][0]["text"] if output else ""

    with_start = retained(True)
    without_start = retained(False)

    assert with_start != ""
    assert without_start == with_start


def test_settling_a_tool_input_refunds_its_previous_reservation() -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    state = implementation._StreamState()
    record = implementation._record_stream_event

    record(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t1", "name": "calc"},
        },
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"a":1,"b":2}'},
        },
        state,
    )
    record({"type": "content_block_stop", "index": 0}, state)

    # The only block, so the envelope is charged once and there is no separator.
    expected = len(
        b'[{"role":"assistant","content":[{"type":"tool_use","id":"t1",'
        b'"name":"calc","input":{"a":1,"b":2}}]}]'
    )

    assert state.budget.bytes_used == expected
    implementation._settle_tool_input(0, state)

    assert state.budget.bytes_used == expected
    assert state.budget.truncated is False


@pytest.mark.parametrize(
    ("max_attribute_length", "redact", "expected"),
    [
        (1_000, False, "東" * 400),
        (1_000, True, "[redacted]"),
        (0, False, None),
    ],
)
def test_configured_cap_applies_after_the_mask_not_to_stream_retention(
    make: Any, max_attribute_length: int, redact: bool, expected: str | None
) -> None:
    def mask(value: Any, _context: Any) -> Any:
        if not redact:
            return value
        return [{"role": "assistant", "content": [{"type": "text", "text": "[redacted]"}]}]

    memory = make(max_attribute_length=max_attribute_length, mask=mask)
    text = ("東" * 400) if not redact else ("x" * 5_000)
    events = beta_stream_events(
        ({"type": "text", "text": ""}, [{"type": "text_delta", "text": text}])
    )

    a = streamed_beta_span(events, memory)

    assert "telemetry.dev.capture.truncated" not in a
    if expected is None:
        assert a.get("gen_ai.output.messages", "") == ""
    else:
        output = json.loads(str(a["gen_ai.output.messages"]))
        assert output[0]["content"][0]["text"] == expected


FALLBACK_BLOCK: dict[str, Any] = {
    "type": "fallback",
    "from": {"model": "claude-opus-5"},
    "to": {"model": "claude-opus-4-8"},
}
DECLINED_ITERATION: dict[str, Any] = {
    "type": "message",
    "model": "claude-opus-5",
    "input_tokens": 5,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}
SERVED_ITERATION: dict[str, Any] = {
    **DECLINED_ITERATION,
    "type": "fallback_message",
    "model": "claude-opus-4-8",
    "output_tokens": 2,
}


def with_iterations(
    events: list[dict[str, Any]], iterations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    # A fallback block marks the switch point; only a fallback_message iteration in the
    # terminal usage proves the fallback model served the response.
    for event in events:
        if event["type"] == "message_delta":
            event["usage"] = {"output_tokens": 2, "iterations": iterations}
    return events


def fallback_stream(iterations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return with_iterations(
        beta_stream_events(
            (FALLBACK_BLOCK, []),
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "Hi"}]),
        ),
        iterations,
    )


ADVISOR_ITERATION: dict[str, Any] = {
    **DECLINED_ITERATION,
    "type": "advisor_message",
    "model": "claude-haiku-4-5",
}


@pytest.mark.parametrize(("served_index", "expected"), [(1_001, "claude-opus-4-8"), (None, None)])
def test_fallback_inspection_reads_only_the_bounded_tail(
    served_index: int | None, expected: str | None
) -> None:
    class HostileIterations(list[dict[str, Any]]):
        def __iter__(self) -> Iterator[dict[str, Any]]:
            raise AssertionError("inspected an iteration outside the bounded tail")

        def __reversed__(self) -> Iterator[dict[str, Any]]:
            raise AssertionError("inspected an iteration outside the bounded tail")

        def __getitem__(self, index: Any) -> Any:
            # A slice copies the whole sequence, so a full scan of the copy would go unseen.
            if isinstance(index, slice):
                raise AssertionError("sliced the iterations")
            if isinstance(index, int) and index < 1_001:
                raise AssertionError("inspected an iteration outside the bounded tail")
            return super().__getitem__(index)

    implementation = cast(Any, telemetry_dev_anthropic)
    values = [DECLINED_ITERATION] * 2_001
    if served_index is not None:
        values[served_index] = SERVED_ITERATION
    values[2_000] = ADVISOR_ITERATION
    iterations = HostileIterations(values)
    state = implementation._StreamState()

    model = implementation._fallback_serving_model({"iterations": iterations}, state)

    assert model == expected
    assert state.metadata_truncated is True
    assert state.budget.truncated is False


def test_fallback_metadata_overflow_preserves_complete_masked_output(make: Any) -> None:
    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return value

    memory = make(mask=mask)
    iterations = [DECLINED_ITERATION] * 1_001
    iterations[999] = SERVED_ITERATION
    iterations[1_000] = ADVISOR_ITERATION

    a = streamed_beta_span(fallback_stream(iterations), memory)

    output = json.loads(str(a["gen_ai.output.messages"]))
    assert output == [
        {"role": "assistant", "content": [FALLBACK_BLOCK, {"type": "text", "text": "Hi"}]}
    ]
    assert a["telemetry.dev.capture.truncated"] is True


def test_beta_create_records_a_fallback_block_under_its_wire_keys(memory: SimpleNamespace) -> None:
    content = [FALLBACK_BLOCK, {"type": "text", "text": "Hi"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload(content=content))

    wrapped_sync_client(handler).beta.messages.create(
        model="claude-opus-5", max_tokens=64, messages=MESSAGES
    )

    output = json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"]))
    assert output[0]["content"] == content


@pytest.mark.parametrize(
    ("iterations", "expected"),
    [
        ([DECLINED_ITERATION, SERVED_ITERATION], "claude-opus-4-8"),
        ([DECLINED_ITERATION], "claude-opus-5"),
        ([DECLINED_ITERATION, ADVISOR_ITERATION], "claude-opus-5"),
        ([DECLINED_ITERATION, SERVED_ITERATION, ADVISOR_ITERATION], "claude-opus-4-8"),
        (
            [SERVED_ITERATION, {**SERVED_ITERATION, "model": "claude-sonnet-5"}, ADVISOR_ITERATION],
            "claude-sonnet-5",
        ),
    ],
    ids=[
        "fallback_served",
        "fallback_failed",
        "advisor_without_fallback",
        "advisor_after_fallback",
        "last_fallback_wins",
    ],
)
def test_beta_stream_response_model_follows_the_last_served_fallback(
    memory: SimpleNamespace, iterations: list[dict[str, Any]], expected: str
) -> None:
    a = streamed_beta_span(fallback_stream(iterations), memory)

    assert a["gen_ai.response.model"] == expected


def test_beta_stream_records_compaction_content(memory: SimpleNamespace) -> None:
    a = streamed_beta_span(
        beta_stream_events(
            (
                {"type": "compaction", "content": None},
                [
                    {
                        "type": "compaction_delta",
                        "content": "Summary so far.",
                        "encrypted_content": "enc_1",
                    }
                ],
            )
        ),
        memory,
    )

    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "content": [
                {"type": "compaction", "content": "Summary so far.", "encrypted_content": "enc_1"}
            ],
        }
    ]


def test_beta_stream_recovers_from_an_oversized_compaction_start(
    memory: SimpleNamespace,
) -> None:
    a = streamed_beta_span(
        beta_stream_events(
            (
                {"type": "compaction", "content": "x" * (60 * 1024)},
                [{"type": "compaction_delta", "content": "recovered"}],
            )
        ),
        memory,
    )

    assert "telemetry.dev.capture.truncated" not in a
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "content": [{"type": "compaction", "content": "recovered"}],
        }
    ]


def test_beta_stream_masks_recovered_compaction_when_rejected_start_had_encrypted_content(
    make: Any,
) -> None:
    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return value

    memory = make(mask=mask)
    a = streamed_beta_span(
        beta_stream_events(
            (
                {
                    "type": "compaction",
                    "content": "x" * (60 * 1024),
                    "encrypted_content": "encrypted",
                },
                [{"type": "compaction_delta", "content": "recovered"}],
            )
        ),
        memory,
    )

    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def test_beta_stream_records_mcp_tool_use_input(memory: SimpleNamespace) -> None:
    block: dict[str, Any] = {
        "type": "mcp_tool_use",
        "id": "mcptoolu_1",
        "name": "search",
        "server_name": "docs",
        "input": {},
    }
    a = streamed_beta_span(
        beta_stream_events(
            (
                block,
                [
                    {"type": "input_json_delta", "partial_json": '{"q": '},
                    {"type": "input_json_delta", "partial_json": '"otel"}'},
                ],
            )
        ),
        memory,
    )

    content = json.loads(str(a["gen_ai.output.messages"]))[0]["content"]
    assert content == [{**block, "input": {"q": "otel"}}]


def test_beta_stream_records_fallback_model_after_capture_budget_is_exhausted(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 64)
    memory = make()
    events = with_iterations(
        beta_stream_events(
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "x" * 60}]),
            (FALLBACK_BLOCK, []),
        ),
        [DECLINED_ITERATION, SERVED_ITERATION],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(events)

    stream = wrapped_sync_client(handler).beta.messages.create(
        model="claude-opus-5", max_tokens=64, messages=BETA_MESSAGES, stream=True
    )
    list(stream)

    assert stream._state.budget.truncated is True
    assert attrs(only_span(memory))["gen_ai.response.model"] == "claude-opus-4-8"


def test_beta_stream_records_output_chunks_for_compaction_content(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps: list[float] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        assert timestamp_ms is not None
        timestamps.append(timestamp_ms)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    streamed_beta_span(
        beta_stream_events(
            (
                {"type": "compaction", "content": None},
                [
                    {"type": "compaction_delta", "content": "Summary ", "encrypted_content": None},
                    {"type": "compaction_delta", "encrypted_content": "enc_ignored"},
                    {"type": "compaction_delta", "content": "so far.", "encrypted_content": "enc"},
                ],
            )
        ),
        memory,
    )

    assert len(timestamps) == 2


@pytest.mark.parametrize("provider", ["anthropic", "gcp.vertex_ai"])
@pytest.mark.parametrize("mode", ["wrap", "instrument"])
async def test_beta_stream_helper_records_one_span_sync_and_async(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, mode: str, provider: str
) -> None:
    def sync_handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(stream_events())

    async def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(stream_events())

    if mode == "instrument":
        instrument_anthropic()
    if provider == "anthropic":
        sync_provider, async_provider = sync_client(sync_handler), async_client(handler)
    else:
        sync_provider, async_provider = provider_clients(monkeypatch, handler, sync_handler)[
            provider
        ]
    if mode == "wrap":
        sync_provider = wrap_anthropic(sync_provider)
        async_provider = wrap_anthropic(async_provider)

    with sync_provider.beta.messages.stream(
        model="claude-sonnet-4-6", max_tokens=64, messages=BETA_MESSAGES
    ) as stream:
        sync_text = "".join(stream.text_stream)
    async with async_provider.beta.messages.stream(
        model="claude-sonnet-4-6", max_tokens=64, messages=BETA_MESSAGES
    ) as stream:
        async_text = "".join([text async for text in stream.text_stream])
    await async_provider.close()

    assert sync_text == async_text == "Hello world"
    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert attrs(span)["gen_ai.provider.name"] == provider
        assert attrs(span)["gen_ai.usage.output_tokens"] == 2
        assert json.loads(str(attrs(span)["gen_ai.output.messages"])) == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}
        ]


def test_missing_provider_beta_module_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # A future SDK may rename these private modules; importing must still fail open.
    monkeypatch.setitem(sys.modules, "anthropic.lib.bedrock._beta_messages", None)
    classes = telemetry_dev_anthropic._provider_beta_classes()  # pyright: ignore[reportPrivateUsage]

    assert (
        AnthropicVertex(region="us-east5", project_id="p", access_token="t").beta.messages.__class__
        in classes
    )
    assert all("bedrock" not in cls.__module__ for cls in classes)


def test_beta_stream_compaction_delta_without_start_is_a_compaction_block(
    memory: SimpleNamespace,
) -> None:
    events = beta_stream_events()
    events.insert(
        1,
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": "Summary.", "encrypted_content": "e"},
        },
    )
    a = streamed_beta_span(events, memory)

    assert json.loads(str(a["gen_ai.output.messages"]))[0]["content"] == [
        {"type": "compaction", "content": "Summary.", "encrypted_content": "e"}
    ]


def test_beta_stream_compaction_omitting_encrypted_content_keeps_the_previous_value(
    memory: SimpleNamespace,
) -> None:
    a = streamed_beta_span(
        beta_stream_events(
            (
                {"type": "compaction", "content": None},
                [
                    {
                        "type": "compaction_delta",
                        "content": "Summary",
                        "encrypted_content": "enc_1",
                    },
                    {"type": "compaction_delta", "content": "Summary."},
                ],
            )
        ),
        memory,
    )

    assert json.loads(str(a["gen_ai.output.messages"]))[0]["content"] == [
        {"type": "compaction", "content": "Summary.", "encrypted_content": "enc_1"}
    ]


def test_compaction_replacement_releases_only_its_own_budget_reservation(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    sibling = {"type": "text", "text": "hi"}
    first = {"type": "compaction", "content": "x" * 100}
    replacement = {"type": "compaction", "content": "ok"}
    final = {"type": "compaction_delta", "content": "ok"}

    stacked = telemetry_dev.CaptureBudget(max_bytes=300)
    assert stacked.accept(sibling)
    assert stacked.accept(first)
    assert stacked.accept(replacement) is False

    record({"type": "content_block_start", "index": 0, "content_block": sibling}, state)
    record(
        {"type": "content_block_start", "index": 1, "content_block": {"type": "compaction"}},
        state,
    )
    for content in ("x" * 100, "ok"):
        delta = {**final, "content": content}
        record({"type": "content_block_delta", "index": 1, "delta": delta}, state)

    expected = telemetry_dev.CaptureBudget(max_bytes=300)
    for retained in (sibling, replacement):
        assert expected.accept(retained)
    assert state.budget.truncated is False
    assert state.budget.bytes_used == expected.bytes_used
    assert state.blocks[1] == replacement


def test_stream_keeps_short_compaction_replacement_within_budget(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    memory = make()
    a = streamed_beta_span(
        beta_stream_events(
            (
                {"type": "compaction", "content": None},
                [
                    {"type": "compaction_delta", "content": "x" * 100},
                    {"type": "compaction_delta", "content": "ok"},
                ],
            )
        ),
        memory,
    )

    assert json.loads(str(a["gen_ai.output.messages"]))[0]["content"] == [
        {"type": "compaction", "content": "ok"}
    ]


def test_oversized_compaction_replacement_drops_stale_block_and_recovers(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    start = {"type": "compaction"}
    first = {"type": "compaction_delta", "content": "ok"}

    record({"type": "content_block_start", "index": 0, "content_block": start}, state)
    record({"type": "content_block_delta", "index": 0, "delta": first}, state)
    oversized = {"type": "compaction_delta", "content": "x" * 400}
    record({"type": "content_block_delta", "index": 0, "delta": oversized}, state)

    # The stale "ok" summary is dropped and its reservation released.
    assert 0 not in state.blocks
    assert state.budget.bytes_used == 0

    final = {"type": "compaction_delta", "content": "final"}
    record({"type": "content_block_delta", "index": 0, "delta": final}, state)

    expected = telemetry_dev.CaptureBudget(max_bytes=300)
    for retained in ({**start, "content": "final"},):
        assert expected.accept(retained)
    assert state.budget.truncated is False
    assert state.budget.bytes_used == expected.bytes_used
    assert state.blocks[0]["content"] == "final"
    record({"type": "message_stop"}, state)
    partial = telemetry_dev_anthropic._stream_partial(state)  # pyright: ignore[reportPrivateUsage]
    assert "attributes" not in partial


def test_multiple_rejected_compaction_replacements_still_recover(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    record(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "compaction"}},
        state,
    )
    for content in ("x" * 400, "y" * 400, "recovered"):
        record(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "compaction_delta", "content": content},
            },
            state,
        )

    assert state.budget.truncated is False
    assert state.unresolved_replacements == set()
    assert state.blocks[0]["content"] == "recovered"


def test_rejected_inherited_encrypted_content_stays_unresolved_until_replaced(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    record(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "compaction"}},
        state,
    )
    for delta in (
        {"type": "compaction_delta", "content": "ok", "encrypted_content": "enc"},
        {"type": "compaction_delta", "content": "x" * 400},
        {"type": "compaction_delta", "content": "recovered"},
    ):
        record({"type": "content_block_delta", "index": 0, "delta": delta}, state)

    assert state.blocks[0] == {"type": "compaction", "content": "recovered"}
    assert state.unresolved_encrypted_content == {0}
    record({"type": "message_stop"}, state)
    partial = telemetry_dev_anthropic._stream_partial(  # pyright: ignore[reportPrivateUsage]
        state, mask_output_when_incomplete=True
    )
    assert partial["output"] is None
    assert partial["attributes"] == {"telemetry.dev.capture.truncated": True}

    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "compaction_delta",
                "content": "resolved",
                "encrypted_content": "enc-2",
            },
        },
        state,
    )

    assert state.unresolved_encrypted_content == set()
    assert telemetry_dev_anthropic._stream_partial(state)["output"] == [  # pyright: ignore[reportPrivateUsage]
        {
            "role": "assistant",
            "content": [
                {"type": "compaction", "content": "resolved", "encrypted_content": "enc-2"}
            ],
        }
    ]


@pytest.mark.parametrize("container_type", ["mapping", "sequence"])
def test_stream_capture_bounds_native_container_traversal(container_type: str) -> None:
    reads = 0

    class HostileMapping(Mapping[str, Any]):
        def __len__(self) -> int:
            return 400

        def __iter__(self) -> Iterator[str]:
            nonlocal reads
            for index in range(400):
                reads += 1
                if reads > 1_000:
                    raise AssertionError("exceeded the traversal limit")
                yield f"key-{index}"

        def __getitem__(self, key: str) -> Any:
            return key

    class HostileSequence(Sequence[Any]):
        def __len__(self) -> int:
            return 749

        def __getitem__(self, index: int | slice) -> Any:
            nonlocal reads
            if isinstance(index, slice):
                raise AssertionError("sliced the sequence")
            if index >= len(self):
                raise IndexError
            reads += 1
            if reads > 1_000:
                raise AssertionError("exceeded the traversal limit")
            return index

    value: Any = HostileMapping() if container_type == "mapping" else HostileSequence()
    bounded = telemetry_dev_anthropic._bounded_native(value)  # pyright: ignore[reportPrivateUsage]

    assert reads <= 1_000
    assert bounded is not value
    if container_type == "mapping":
        assert bounded == {f"key-{index}": f"key-{index}" for index in range(400)}
    else:
        assert bounded == list(range(749))


@pytest.mark.parametrize("container_type", ["mapping", "sequence"])
def test_stream_capture_rejects_containers_over_the_item_limit(container_type: str) -> None:
    reads = 0

    class OversizedMapping(Mapping[str, Any]):
        def __len__(self) -> int:
            return 5_000

        def __iter__(self) -> Iterator[str]:
            nonlocal reads
            for index in range(5_000):
                reads += 1
                if reads > 1_000:
                    raise AssertionError("exceeded the traversal limit")
                yield f"key-{index}"

        def __getitem__(self, key: str) -> Any:
            return key

    class OversizedSequence(Sequence[Any]):
        def __len__(self) -> int:
            return 5_000

        def __getitem__(self, index: int | slice) -> Any:
            nonlocal reads
            if isinstance(index, slice):
                raise AssertionError("sliced the sequence")
            reads += 1
            if reads > 1_000:
                raise AssertionError("exceeded the traversal limit")
            return index

    value: Any = OversizedMapping() if container_type == "mapping" else OversizedSequence()
    bounded = telemetry_dev_anthropic._bounded_native(value)  # pyright: ignore[reportPrivateUsage]

    assert bounded is telemetry_dev_anthropic._OMIT  # pyright: ignore[reportPrivateUsage]
    assert reads <= 1_000


def test_stream_capture_rejects_large_pydantic_extras_and_ignores_subclassed_extras() -> None:
    reads = 0

    class CountingExtras(dict[str, Any]):
        def __iter__(self) -> Iterator[str]:
            nonlocal reads
            for key in super().__iter__():
                reads += 1
                if reads > 1_000:
                    raise AssertionError("materialized unbounded extras")
                yield key

        def keys(self) -> Any:
            return self.__iter__()

        def items(self) -> Any:
            def counted_items() -> Iterator[tuple[str, Any]]:
                nonlocal reads
                for item in super(CountingExtras, self).items():
                    reads += 1
                    if reads > 1_000:
                        raise AssertionError("materialized unbounded extras")
                    yield item

            return counted_items()

    class Model(BaseModel):
        model_config = ConfigDict(extra="allow")

    model = Model.model_validate({f"key-{index}": index for index in range(5_000)})
    assert "__pydantic_extra__" not in vars(model)
    implementation = cast(Any, telemetry_dev_anthropic)

    assert implementation._bounded_native(model) is implementation._OMIT

    model.__pydantic_extra__ = CountingExtras(model.__pydantic_extra__ or {})

    assert implementation._bounded_native(model) == {}
    assert reads == 0


def test_stream_capture_keeps_pydantic_extras_from_slot_storage() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(extra="allow")
        type: str

    model = Model.model_validate({"type": "text", "citations_extra": "kept"})
    assert "__pydantic_extra__" not in vars(model)

    bounded = telemetry_dev_anthropic._bounded_native(model)  # pyright: ignore[reportPrivateUsage]

    assert bounded == {"type": "text", "citations_extra": "kept"}


def test_stream_capture_alias_inspection_is_bounded(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation = cast(Any, telemetry_dev_anthropic)
    stored = implementation._stored
    alias_reads = 0

    def counted_stored(value: Any, name: str) -> Any:
        nonlocal alias_reads
        if name == "alias":
            alias_reads += 1
        return stored(value, name)

    class SparseModel:
        __pydantic_fields__ = {
            f"unused_{index}": Field(alias=f"unused-{index}") for index in range(5_000)
        } | {"from_": Field(alias="from")}

        def __init__(self) -> None:
            self.type = "fallback"
            self.from_ = {"model": "claude-opus-5"}

    monkeypatch.setattr(implementation, "_stored", counted_stored)
    events = [
        {"type": "content_block_start", "index": 0, "content_block": SparseModel()},
        {"type": "message_stop"},
    ]

    def create(**_kwargs: Any) -> Iterator[Any]:
        return iter(events)

    client: Any = object.__new__(Anthropic)
    client.messages = SimpleNamespace(create=create)
    delivered = list(wrap_anthropic(client).messages.create(model="claude-opus-5", stream=True))

    assert delivered == events
    output = json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"]))
    assert output[0]["content"] == [{"type": "fallback", "from": {"model": "claude-opus-5"}}]
    assert alias_reads <= 1_000


def _spoofed_and_aliased_storage(calls: list[str]) -> list[Any]:
    class Payload:
        def __get__(self, instance: Any, owner: Any) -> dict[str, Any]:
            calls.append("payload")
            return {"custom": {"input": "x"}, "text": "x"}

    class Spoofed(Payload):
        @property
        def __class__(self) -> type:  # type: ignore[override]
            calls.append("__class__")
            return types.MemberDescriptorType

        __name__ = "__pydantic_extra__"

    class SpoofedModel:
        __pydantic_extra__ = Spoofed()

    class Meta(type):
        __pydantic_extra__ = type.__dict__["__doc__"]
        __dict__ = type.__dict__["__doc__"]  # type: ignore[assignment]

    class Aliased(metaclass=Meta):
        __doc__ = Payload()  # type: ignore[assignment]

    return [SpoofedModel(), Aliased]


def test_stream_capture_never_runs_storage_properties() -> None:
    calls: list[str] = []

    class Hostile:
        @property
        def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
            calls.append("__dict__")
            return {"text": "x"}

        @property
        def __pydantic_extra__(self) -> dict[str, Any]:
            calls.append("__pydantic_extra__")
            return {"text": "x"}

        @property
        def model_fields_set(self) -> set[str]:
            calls.append("model_fields_set")
            return {"encrypted_content"}

    implementation = cast(Any, telemetry_dev_anthropic)
    implementation._bounded_native(Hostile())
    implementation._field_present(Hostile(), "encrypted_content")
    for hostile in _spoofed_and_aliased_storage(calls):
        implementation._bounded_native(hostile)
        implementation._field_present(hostile, "text")

    assert calls == []


def test_stream_capture_bounds_underreported_mapping_traversal() -> None:
    reads = 0

    class UnderreportedMapping(Mapping[str, Any]):
        def __len__(self) -> int:
            return 1

        def __iter__(self) -> Iterator[str]:
            nonlocal reads
            for index in range(5_000):
                reads += 1
                if reads > 1_000:
                    raise AssertionError("exceeded the traversal limit")
                yield f"key-{index}"

        def __getitem__(self, key: str) -> Any:
            return key

    bounded = telemetry_dev_anthropic._bounded_native(  # pyright: ignore[reportPrivateUsage]
        UnderreportedMapping()
    )

    assert bounded is telemetry_dev_anthropic._OMIT  # pyright: ignore[reportPrivateUsage]
    assert reads <= 1_000


def test_rejected_unseen_compaction_replacement_recovers(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]

    for content in ("x" * 400, "recovered"):
        record(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "compaction_delta", "content": content},
            },
            state,
        )

    assert state.budget.truncated is False
    assert state.unresolved_replacements == set()
    assert state.blocks[0]["content"] == "recovered"


def test_masked_beta_stream_omits_incomplete_retained_output(make: Any) -> None:
    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return value

    memory = make(mask=mask, max_attribute_length=50 * 1024)
    a = streamed_beta_span(
        beta_stream_events(
            (
                {"type": "thinking", "thinking": ""},
                [
                    {"type": "thinking_delta", "thinking": "retained"},
                    {"type": "signature_delta", "signature": "x" * (60 * 1024)},
                ],
            )
        ),
        memory,
    )

    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def test_compaction_after_other_block_truncation_keeps_fitting_replacement(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]

    record(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "compaction"}},
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": "ok"},
        },
        state,
    )
    record(
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "y" * 400},
        },
        state,
    )
    assert state.budget.truncated is True

    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": "hi"},
        },
        state,
    )

    assert state.budget.truncated is True
    assert state.blocks[0] == {"type": "compaction", "content": "hi"}


def test_signature_replacement_releases_its_previous_budget_reservation(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 350)
    # 350 bytes fits the block with the long signature (241 bytes) and the block with the
    # short one (144 bytes), but not both stacked (385 bytes).
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    start = {"type": "thinking", "thinking": ""}
    final = {"type": "signature_delta", "signature": "sig"}

    record({"type": "content_block_start", "index": 0, "content_block": start}, state)
    for signature in ("s" * 100, "sig"):
        delta = {**final, "signature": signature}
        record({"type": "content_block_delta", "index": 0, "delta": delta}, state)

    expected = telemetry_dev.CaptureBudget(max_bytes=350)
    for retained in ({**start, "signature": "sig"},):
        assert expected.accept(retained)
    assert state.budget.truncated is False
    assert state.budget.bytes_used == expected.bytes_used
    assert state.blocks[0]["signature"] == "sig"


def test_first_compaction_delta_replaces_the_start_shell_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = {"type": "compaction", "content": "x" * 50}
    expected = telemetry_dev.CaptureBudget()
    assert expected.accept(completed)
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", expected.bytes_used)
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]

    record(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "compaction"}},
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": "x" * 50},
        },
        state,
    )

    assert state.blocks[0] == completed
    assert state.budget.bytes_used == state.budget.max_bytes == expected.bytes_used
    assert state.budget.truncated is False


def test_replacement_budget_invariants_hold_for_random_streams(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 600)
    make()
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    rng = random.Random(1729)

    def measure(value: dict[str, Any]) -> int:
        budget = telemetry_dev.CaptureBudget()
        assert budget.accept(value)
        return budget.bytes_used

    def random_event(index: int, kind: str) -> dict[str, Any]:
        if kind == "compaction":
            delta: dict[str, Any] = {"type": "compaction_delta"}
            if rng.random() < 0.9:
                delta["content"] = None if rng.random() < 0.1 else "c" * rng.randrange(120)
            if rng.random() < 0.5:
                delta["encrypted_content"] = "e" * rng.randrange(40)
        elif kind == "thinking" and rng.random() < 0.5:
            delta = {"type": "signature_delta", "signature": "s" * rng.randrange(120)}
        elif kind == "thinking":
            delta = {"type": "thinking_delta", "thinking": "t" * rng.randrange(20)}
        else:
            delta = {"type": "text_delta", "text": "x" * rng.randrange(20)}
        return {"type": "content_block_delta", "index": index, "delta": delta}

    starts = {
        "compaction": {"type": "compaction"},
        "thinking": {"type": "thinking", "thinking": ""},
        "text": {"type": "text", "text": ""},
    }
    for _ in range(1000):
        state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
        kinds = [rng.choice(list(starts)) for _ in range(rng.randrange(1, 4))]
        for index, kind in enumerate(kinds):
            record(
                {"type": "content_block_start", "index": index, "content_block": starts[kind]},
                state,
            )
        for _ in range(rng.randrange(1, 12)):
            index = rng.randrange(len(kinds))
            event = random_event(index, kinds[index])
            delta = event["delta"]
            replacing = delta["type"] in ("compaction_delta", "signature_delta")
            was_truncated = state.budget.truncated
            held = state.block_reservations.get(index, (0, 0))[0]
            others = state.budget.bytes_used - held
            before = (dict(state.blocks.get(index, {})), state.budget.bytes_used)
            candidate = telemetry_dev_anthropic._replaced_block(index, delta, state)  # pyright: ignore[reportPrivateUsage]

            record(event, state)

            reserved = sum(bytes_held for bytes_held, _ in state.block_reservations.values())
            assert reserved == state.budget.bytes_used
            if not replacing or was_truncated or candidate is None:
                continue
            fits = measure(candidate) <= state.budget.max_bytes - others
            compaction = delta["type"] == "compaction_delta"
            assert (state.blocks.get(index) == candidate) == fits
            output = telemetry_dev_anthropic._stream_output(state)  # pyright: ignore[reportPrivateUsage]
            emitted: list[Any] = output[0]["content"] if output else []
            assert len(emitted) == len(state.blocks)
            if fits:
                continue
            if not before[0] and index not in state.unresolved_replacements:
                assert state.budget.truncated is True
                continue
            assert state.budget.truncated is False
            if compaction:
                # A rejected summary drops the stale block and frees everything it held.
                assert index not in state.blocks
                assert index not in state.block_reservations
                assert state.budget.bytes_used == others
            else:
                # A rejected signature keeps the thinking block without the stale signature.
                kept = {key: value for key, value in before[0].items() if key != "signature"}
                assert state.blocks.get(index) == kept


def test_stream_drops_compaction_superseded_by_a_rejected_final_replacement(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 400)
    memory = make()
    a = streamed_beta_span(
        beta_stream_events(
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "Hi"}]),
            (
                {"type": "compaction", "content": None},
                [
                    {"type": "compaction_delta", "content": "ok"},
                    {"type": "compaction_delta", "content": "x" * 400},
                ],
            ),
        ),
        memory,
    )

    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}
    ]
    assert a["telemetry.dev.capture.truncated"] is True


def test_rejected_signature_keeps_thinking_and_drops_the_stale_signature(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 800)
    # 800 bytes fits the thinking text (698 bytes) and the block with the short signature
    # (636 bytes), but not the block with the oversized signature (1233 bytes).
    memory = make()
    a = streamed_beta_span(
        beta_stream_events(
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "Hi"}]),
            (
                {"type": "thinking", "thinking": ""},
                [
                    {"type": "thinking_delta", "thinking": "t" * 300},
                    {"type": "signature_delta", "signature": "sig"},
                    {"type": "signature_delta", "signature": "s" * 600},
                ],
            ),
        ),
        memory,
    )

    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hi"},
                {"type": "thinking", "thinking": "t" * 300},
            ],
        }
    ]


def test_rejected_signature_stays_unresolved_until_a_replacement_is_accepted(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    record(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        state,
    )
    for signature in ("sig", "x" * 400):
        record(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature},
            },
            state,
        )

    assert state.unresolved_replacements == {0}
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "recovered"},
        },
        state,
    )
    assert state.unresolved_replacements == set()
    assert state.blocks[0]["signature"] == "recovered"


def test_rejected_signature_releases_budget_for_sibling_and_recovers(make: Any) -> None:
    make(max_attribute_length=50 * 1024)
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    record(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        state,
    )
    for signature in ("s" * (40 * 1024), "x" * (60 * 1024)):
        record(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": signature},
            },
            state,
        )
    record(
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "t" * (10 * 1024)},
        },
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "recovered"},
        },
        state,
    )

    assert state.budget.truncated is False
    assert state.unresolved_replacements == set()
    assert state.blocks[0]["signature"] == "recovered"
    assert state.blocks[1]["text"] == "t" * (10 * 1024)
    assert state.budget.bytes_used == sum(
        held_bytes for held_bytes, _ in state.block_reservations.values()
    )


def test_rejected_unseen_replacements_use_sticky_bounded_truncation(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 100)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]

    for index in range(2000):
        record(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "signature_delta", "signature": "x" * 200},
            },
            state,
        )

    assert state.budget.truncated is True
    assert state.unresolved_replacements == set()
    assert state.blocks == {}


def test_rejected_compaction_indexes_are_bounded_before_sticky_truncation(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 100)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]

    for index in range(1100):
        record(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "compaction"},
            },
            state,
        )
        record(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "compaction_delta", "content": "x" * 200},
            },
            state,
        )

    assert state.budget.truncated is True
    assert len(state.unresolved_replacements) <= 1000
    assert len(state.blocks) <= 1000
    assert len(state.block_reservations) <= 1000


def test_rejected_compaction_releases_budget_for_later_blocks(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 400)
    memory = make()
    a = streamed_beta_span(
        beta_stream_events(
            (
                {"type": "compaction", "content": None},
                [
                    {"type": "compaction_delta", "content": "x" * 150},
                    {"type": "compaction_delta", "content": "x" * 400},
                ],
            ),
            ({"type": "text", "text": ""}, [{"type": "text_delta", "text": "y" * 200}]),
        ),
        memory,
    )

    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "y" * 200}]}
    ]


def test_rejected_compactions_after_truncation_leave_no_state(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    record({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}, state)
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "z" * 500},
        },
        state,
    )
    assert state.budget.truncated is True
    retained = (
        dict(state.blocks),
        dict(state.block_reservations),
        state.budget.bytes_used,
        set(state.unresolved_replacements),
    )

    for index in range(1, 2001):
        delta = {"type": "compaction_delta", "content": "c"}
        record({"type": "content_block_delta", "index": index, "delta": delta}, state)

    assert (
        state.blocks,
        state.block_reservations,
        state.budget.bytes_used,
        state.unresolved_replacements,
    ) == retained

    record(
        {
            "type": "content_block_delta",
            "index": 2_001,
            "delta": {"type": "compaction_delta", "content": ["c"] * 1_001},
        },
        state,
    )

    assert (
        state.blocks,
        state.block_reservations,
        state.budget.bytes_used,
        state.unresolved_replacements,
    ) == retained


def test_oversized_compaction_replacement_keeps_inherited_encrypted_content_unresolved(
    make: Any,
) -> None:
    make(max_attribute_length=50 * 1024)
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    record(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "compaction"}},
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "compaction_delta",
                "content": "before",
                "encrypted_content": "enc",
            },
        },
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": ["x"] * 1_001},
        },
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "compaction_delta", "content": "recovered"},
        },
        state,
    )

    assert state.blocks[0] == {"type": "compaction", "content": "recovered"}
    assert state.unresolved_replacements == set()
    assert state.unresolved_encrypted_content == {0}
    record({"type": "message_stop"}, state)
    partial = telemetry_dev_anthropic._stream_partial(state)  # pyright: ignore[reportPrivateUsage]
    assert partial["attributes"] == {"telemetry.dev.capture.truncated": True}
    masked = telemetry_dev_anthropic._stream_partial(  # pyright: ignore[reportPrivateUsage]
        state, mask_output_when_incomplete=True
    )
    assert masked["output"] is None


@pytest.mark.parametrize("signature", ["sig", "s" * 200])
def test_signature_at_a_tool_index_reserves_the_raw_tool_input_it_keeps(
    make: Any, signature: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 600)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    start: dict[str, Any] = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}
    partial_json = json.dumps({"query": "q" * 200})

    record({"type": "content_block_start", "index": 0, "content_block": start}, state)
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
        state,
    )
    held = state.budget.bytes_used
    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": signature},
        },
        state,
    )

    # The signed block is measured with the raw input it keeps; one that does not fit leaves
    # the unsigned block and its input on their existing reservation.
    signed = {**start, "signature": signature}
    expected = telemetry_dev.CaptureBudget(max_bytes=600)
    fits = expected.accept((signed, partial_json))
    assert fits == (signature == "sig")
    retained = signed if fits else start
    assert state.budget.bytes_used == (expected.bytes_used if fits else held)
    assert state.budget.truncated is False
    assert telemetry_dev_anthropic._stream_output(state) == [  # pyright: ignore[reportPrivateUsage]
        {"role": "assistant", "content": [{**retained, "input": {"query": "q" * 200}}]}
    ]


def test_dropped_compaction_at_a_tool_index_leaves_no_tool_input_for_later_blocks(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]

    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"a": 1}'},
        },
        state,
    )
    oversized = {"type": "compaction_delta", "content": "x" * 400}
    record({"type": "content_block_delta", "index": 0, "delta": oversized}, state)
    record(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        state,
    )

    assert telemetry_dev_anthropic._stream_output(state) == [  # pyright: ignore[reportPrivateUsage]
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}
    ]


def test_signature_after_truncation_remeasures_the_stripped_block(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_anthropic, "_STREAM_CAPTURE_MAX_BYTES", 300)
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    thinking = {"type": "thinking", "thinking": ""}

    record({"type": "content_block_start", "index": 0, "content_block": thinking}, state)
    for delta in (
        {"type": "thinking_delta", "thinking": "t" * 20},
        {"type": "signature_delta", "signature": "sig"},
    ):
        record({"type": "content_block_delta", "index": 0, "delta": delta}, state)
    record({"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}, state)
    record(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "z" * 500},
        },
        state,
    )
    assert state.budget.truncated is True
    retained_bytes = state.budget.bytes_used

    record(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "new"},
        },
        state,
    )

    assert state.budget.bytes_used < retained_bytes
    assert state.budget.bytes_used == sum(
        held_bytes for held_bytes, _ in state.block_reservations.values()
    )
    assert state.unresolved_replacements == {0}
    assert state.blocks[0] == {"type": "thinking", "thinking": "t" * 20}


@pytest.mark.parametrize("spare_items", [0, -1])
def test_compaction_replacement_respects_the_item_limit(make: Any, spare_items: int) -> None:
    make()
    state = telemetry_dev_anthropic._StreamState()  # pyright: ignore[reportPrivateUsage]
    record = telemetry_dev_anthropic._record_stream_event  # pyright: ignore[reportPrivateUsage]
    sibling = {"type": "text", "text": "hi"}
    replaced = {"type": "compaction", "content": "ok"}

    def items(value: object) -> int:
        budget = telemetry_dev.CaptureBudget()
        assert budget.accept(value)
        return budget.items_used

    # Exactly enough items for the sibling and the completed block, so the replacement only
    # fits once the start shell's items are released.
    state.budget = telemetry_dev.CaptureBudget(
        max_items=items(sibling) + items(replaced) + spare_items
    )
    record({"type": "content_block_start", "index": 0, "content_block": sibling}, state)
    record(
        {"type": "content_block_start", "index": 1, "content_block": {"type": "compaction"}},
        state,
    )
    record(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "compaction_delta", "content": "ok"},
        },
        state,
    )

    assert state.budget.truncated is False
    assert state.budget.items_used == sum(held for _, held in state.block_reservations.values())
    assert state.blocks.get(1) == (replaced if spare_items == 0 else None)


COMPACTION_ITERATION: dict[str, Any] = {
    "type": "compaction",
    "input_tokens": 40,
    "output_tokens": 7,
    "cache_creation_input_tokens": 4,
    "cache_read_input_tokens": 6,
    "cache_creation": {"ephemeral_5m_input_tokens": 1, "ephemeral_1h_input_tokens": 3},
}
FIRST_TURN: dict[str, Any] = {
    "type": "message",
    "model": "claude-opus-5",
    "input_tokens": 4,
    "output_tokens": 2,
    "cache_creation_input_tokens": 1,
    "cache_read_input_tokens": 1,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1},
}
SECOND_TURN: dict[str, Any] = {
    **FIRST_TURN,
    "type": "fallback_message",
    "model": "claude-opus-4-8",
    "input_tokens": 6,
    "output_tokens": 3,
    "cache_creation_input_tokens": 2,
    "cache_read_input_tokens": 2,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 2},
}
# Advisor tokens are billed at the advisor's rates and left out of the top-level usage.
ADVISOR_TURN: dict[str, Any] = {
    **FIRST_TURN,
    "type": "advisor_message",
    "model": "claude-haiku-4-5",
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 5},
}
# The top-level usage sums both executor turns, while message_start reports only the first.
TURNS_USAGE: dict[str, Any] = {
    "input_tokens": 10,
    "output_tokens": 5,
    "cache_creation_input_tokens": 3,
    "cache_read_input_tokens": 3,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 3},
}
FIRST_TURN_USAGE: dict[str, Any] = {
    "input_tokens": 4,
    "output_tokens": 0,
    "cache_creation_input_tokens": 1,
    "cache_read_input_tokens": 1,
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1},
}


def usage_span(
    memory: SimpleNamespace,
    streaming: bool,
    start_usage: dict[str, Any],
    final_usage: dict[str, Any],
) -> dict[str, object]:
    # A stream reports start_usage on message_start and final_usage, minus the one-hour split it
    # never carries there, on message_delta; a unary response reports final_usage.
    delta_usage = {key: value for key, value in final_usage.items() if key != "cache_creation"}
    events = [
        {
            "type": "message_start",
            "message": message_payload(id="msg_usage", content=[], usage=start_usage),
        },
        {"type": "message_delta", "delta": {}, "usage": delta_usage},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": delta_usage},
        {"type": "message_stop"},
    ]
    response = message_payload(usage=final_usage)

    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(events) if streaming else json_response(response)

    client = wrapped_sync_client(handler)
    if streaming:
        for _event in client.beta.messages.create(
            model="claude-opus-5", max_tokens=64, messages=BETA_MESSAGES, stream=True
        ):
            pass
    else:
        client.beta.messages.create(model="claude-opus-5", max_tokens=64, messages=BETA_MESSAGES)
    return attrs(only_span(memory))


@pytest.mark.parametrize("streaming", [False, True], ids=["unary", "stream"])
def test_compaction_iterations_are_added_to_usage_with_one_hour_cache_writes(
    memory: SimpleNamespace, streaming: bool
) -> None:
    a = usage_span(
        memory,
        streaming,
        FIRST_TURN_USAGE,
        {
            **TURNS_USAGE,
            "iterations": [
                FIRST_TURN,
                ADVISOR_TURN,
                COMPACTION_ITERATION,
                SECOND_TURN,
                {
                    **ADVISOR_TURN,
                    "input_tokens": 23,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 9,
                },
            ],
        },
    )

    assert a["anthropic.advisor.model"] == "claude-haiku-4-5"
    assert a["anthropic.advisor.usage.input_tokens"] == 123
    assert a["anthropic.advisor.usage.output_tokens"] == 57
    assert a["anthropic.advisor.usage.cache_read_input_tokens"] == 10
    assert a["anthropic.advisor.usage.cache_creation_input_tokens"] == 2
    assert a["gen_ai.usage.input_tokens"] == 10 + 3 + 3 + 40 + 6 + 4
    assert a["gen_ai.usage.output_tokens"] == 5 + 7
    assert a["gen_ai.usage.cache_read.input_tokens"] == 3 + 6
    assert a["gen_ai.usage.cache_creation.input_tokens"] == 3 + 4
    assert a["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"] == 1 + 2 + 3
    assert "telemetry.dev.capture.truncated" not in a


@pytest.mark.parametrize("streaming", [False, True], ids=["unary", "stream"])
def test_one_hour_cache_writes_are_recorded_without_iterations(
    memory: SimpleNamespace, streaming: bool
) -> None:
    a = usage_span(memory, streaming, FIRST_TURN_USAGE, {**FIRST_TURN_USAGE, "output_tokens": 2})

    assert a["gen_ai.usage.cache_creation.input_tokens"] == 1
    assert a["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"] == 1
    assert not any(key.startswith("anthropic.advisor.") for key in a)


@pytest.mark.parametrize("streaming,model", [(False, None), (True, None), (False, ""), (True, " ")])
def test_advisor_usage_preserves_zero_and_omits_absent_fields(
    memory: SimpleNamespace, streaming: bool, model: str | None
) -> None:
    a = usage_span(
        memory,
        streaming,
        FIRST_TURN_USAGE,
        {
            **TURNS_USAGE,
            "iterations": [
                {"type": "advisor_message", "model": model, "input_tokens": 0, "output_tokens": 0}
            ],
        },
    )
    assert a["anthropic.advisor.usage.input_tokens"] == 0
    assert a["anthropic.advisor.usage.output_tokens"] == 0
    assert "anthropic.advisor.model" not in a
    assert "anthropic.advisor.usage.cache_read_input_tokens" not in a
    assert "anthropic.advisor.usage.cache_creation_input_tokens" not in a
    assert a["gen_ai.usage.input_tokens"] == 16
    assert a["gen_ai.usage.output_tokens"] == 5
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize(
    "deltas,advisor_input",
    [
        ([], 100),
        ([{"type": "message_delta", "delta": {}, "usage": {"output_tokens": 3}}], 100),
        (
            [
                {
                    "type": "message_delta",
                    "delta": {},
                    "usage": {
                        **TURNS_USAGE,
                        "iterations": [FIRST_TURN, ADVISOR_TURN, ADVISOR_TURN],
                    },
                }
            ],
            200,
        ),
    ],
    ids=["start_only", "delta_without_iterations", "delta_with_cumulative_iterations"],
)
def test_interrupted_stream_keeps_advisor_usage_from_message_start(
    memory: SimpleNamespace, deltas: list[dict[str, Any]], advisor_input: int
) -> None:
    start_usage = {**FIRST_TURN_USAGE, "iterations": [FIRST_TURN, ADVISOR_TURN]}
    events = [
        {
            "type": "message_start",
            "message": message_payload(id="msg_usage", content=[], usage=start_usage),
        },
        *deltas,
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(events)

    client = wrapped_sync_client(handler)
    stream = client.beta.messages.create(
        model="claude-opus-5", max_tokens=64, messages=BETA_MESSAGES, stream=True
    )
    iterator = iter(stream)
    for _ in range(1 + len(deltas)):
        next(iterator)
    stream.close()

    a = attrs(only_span(memory))
    assert a["anthropic.advisor.model"] == "claude-haiku-4-5"
    assert a["anthropic.advisor.usage.input_tokens"] == advisor_input
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("streaming", [False, True], ids=["unary", "stream"])
async def test_async_advisor_usage(memory: SimpleNamespace, streaming: bool) -> None:
    usage = {**TURNS_USAGE, "iterations": [ADVISOR_TURN, ADVISOR_TURN]}
    events = [
        {"type": "message_start", "message": message_payload(content=[], usage=FIRST_TURN_USAGE)},
        {"type": "message_delta", "delta": {}, "usage": usage},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": usage},
        {"type": "message_stop"},
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return (
            named_sse_response(events) if streaming else json_response(message_payload(usage=usage))
        )

    client = wrapped_async_client(handler)
    result = await client.beta.messages.create(
        model="claude-opus-5", max_tokens=64, messages=BETA_MESSAGES, stream=streaming
    )
    if streaming:
        async for _event in result:
            pass
    await client.close()
    a = attrs(only_span(memory))
    assert a["anthropic.advisor.model"] == "claude-haiku-4-5"
    assert a["anthropic.advisor.usage.input_tokens"] == 200
    assert a["anthropic.advisor.usage.output_tokens"] == 100
    assert a["anthropic.advisor.usage.cache_read_input_tokens"] == 2
    assert a["anthropic.advisor.usage.cache_creation_input_tokens"] == 2
    assert a["gen_ai.usage.input_tokens"] == 16
    assert a["gen_ai.usage.output_tokens"] == 5


@pytest.mark.parametrize("streaming", [False, True], ids=["unary", "stream"])
def test_advisor_totals_with_mixed_models_are_not_assigned_one_models_rates(
    memory: SimpleNamespace, streaming: bool
) -> None:
    a = usage_span(
        memory,
        streaming,
        FIRST_TURN_USAGE,
        {
            **TURNS_USAGE,
            "iterations": [
                ADVISOR_TURN,
                {**ADVISOR_TURN, "model": "claude-opus-4-8"},
                ADVISOR_TURN,
            ],
        },
    )
    assert a["anthropic.advisor.usage.input_tokens"] == 300
    assert "anthropic.advisor.model" not in a
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("streaming", [False, True], ids=["unary", "stream"])
def test_compaction_usage_past_the_iteration_bound_is_flagged(
    memory: SimpleNamespace, streaming: bool
) -> None:
    a = usage_span(
        memory,
        streaming,
        FIRST_TURN_USAGE,
        {**FIRST_TURN_USAGE, "iterations": [COMPACTION_ITERATION] * 1_001},
    )

    assert a["gen_ai.usage.output_tokens"] == 7 * 1_000
    assert a["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"] == 1 + 3 * 1_000
    assert a["telemetry.dev.capture.truncated"] is True


def test_iterations_usage_reads_only_the_bounded_tail() -> None:
    class HostileIterations(list[dict[str, Any]]):
        def __iter__(self) -> Iterator[dict[str, Any]]:
            raise AssertionError("inspected an iteration outside the bounded tail")

        def __reversed__(self) -> Iterator[dict[str, Any]]:
            raise AssertionError("inspected an iteration outside the bounded tail")

        def __getitem__(self, index: Any) -> Any:
            # A slice copies the whole sequence, so a full scan of the copy would go unseen.
            if isinstance(index, slice):
                raise AssertionError("sliced the iterations")
            if isinstance(index, int) and index < 1_001:
                raise AssertionError("inspected an iteration outside the bounded tail")
            return super().__getitem__(index)

    implementation = cast(Any, telemetry_dev_anthropic)
    values = [FIRST_TURN] * 2_001
    values[1_001] = COMPACTION_ITERATION

    iterations = implementation._iterations_usage({"iterations": HostileIterations(values)})

    assert iterations.compaction["output_tokens"] == 7
    assert iterations.compaction_one_hour_cache_creation_tokens == 3
    # Executor iterations past the bound go unseen, so their sum isn't kept.
    assert iterations.executor_one_hour_cache_creation_tokens is None
    assert iterations.truncated is True


def test_stream_keeps_message_start_one_hour_split_when_an_iteration_leaves_it_out(
    memory: SimpleNamespace,
) -> None:
    a = usage_span(
        memory,
        True,
        {
            **FIRST_TURN_USAGE,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 10},
        },
        {**TURNS_USAGE, "iterations": [{**FIRST_TURN, "cache_creation": None}, SECOND_TURN]},
    )

    assert a["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"] == 10
