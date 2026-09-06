from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import MessageParam, TextBlock
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from telemetry_dev_anthropic import instrument_anthropic, uninstrument_anthropic, wrap_anthropic

SyncHandler = Callable[[httpx.Request], httpx.Response]
AsyncHandler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]

MESSAGES: list[MessageParam] = [{"role": "user", "content": "Say hi"}]


def only_span(env: SimpleNamespace) -> ReadableSpan:
    spans = env.span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


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
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        stop_sequences=["END"],
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
    assert a["gen_ai.usage.input_tokens"] == 11
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

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload(content=[tool_use], stop_reason="tool_use"))

    client = wrapped_sync_client(handler)
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=MESSAGES,
        tools=[tool],
    )

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [tool_use]}
    ]
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["tool_use"]
    assert json.loads(str(a["gen_ai.input.messages"])) == {"messages": MESSAGES, "tools": [tool]}


def test_create_streaming_preserves_events_and_records_aggregate(memory: SimpleNamespace) -> None:
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


def test_stream_mapping_error_reports_and_still_yields_provider_event(make: Any) -> None:
    errors: list[BaseException] = []
    memory = make(on_error=errors.append)

    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(stream_events())

    class BrokenDelta:
        def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
            raise ValueError("model dump failed")

    provider_event = SimpleNamespace(
        type="content_block_delta",
        index=0,
        delta=BrokenDelta(),
    )
    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    stream._inner = iter([provider_event])

    assert next(stream) is provider_event
    with pytest.raises(StopIteration):
        next(stream)
    client.close()

    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert str(errors[0]) == "model dump failed"
    assert only_span(memory).status.status_code == StatusCode.UNSET


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


async def test_async_create_streaming_matches_sync(memory: SimpleNamespace) -> None:
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
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]}
    ]
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 2


def test_sync_stream_bounds_retained_events_without_dropping_chunks(
    memory: SimpleNamespace,
) -> None:
    source_events = oversized_stream_events()

    def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(source_events)

    client = wrapped_sync_client(handler)
    stream = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    delivered = list(stream)
    state = stream._state

    assert len(delivered) == len(source_events)
    assert state.budget.truncated is True
    assert state.budget.bytes_used <= state.budget.max_bytes
    assert len(state.blocks[0]["text"]) < 64 * 1024


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
                "oversized": "x" * (64 * 1024),
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
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 3
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["tool_use"]


async def test_async_stream_bounds_retained_events_without_dropping_chunks(
    memory: SimpleNamespace,
) -> None:
    source_events = oversized_stream_events()

    async def handler(request: httpx.Request) -> httpx.Response:
        return named_sse_response(source_events)

    client = wrapped_async_client(handler)
    stream = await client.messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES, stream=True
    )
    delivered = [event async for event in stream]
    state = stream._state
    await client.close()

    assert len(delivered) == len(source_events)
    assert state.budget.truncated is True
    assert state.budget.bytes_used <= state.budget.max_bytes
    assert len(state.blocks[0]["text"]) < 64 * 1024


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
    for event in stream:
        if event.type == "content_block_delta":
            break
    stream.close()

    span = only_span(memory)
    assert span.status.status_code == StatusCode.UNSET
    a = attrs(span)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
    ]
    assert "gen_ai.response.finish_reasons" not in a


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
    assert attrs(span)["error.type"] == type(exc_info.value).__name__


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
    def handler(request: httpx.Request) -> httpx.Response:
        return error_response(500, "server broke")

    client = wrapped_sync_client(handler)
    with pytest.raises(anthropic.APIStatusError):
        with client.messages.stream(model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES):
            pass

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


def test_provider_mapping_for_bedrock_vertex_and_plain_client(memory: SimpleNamespace) -> None:
    anthropic_any: Any = anthropic
    bedrock = object.__new__(anthropic_any.AnthropicBedrock)
    vertex = object.__new__(anthropic_any.AnthropicVertex)

    def bedrock_create(**_kwargs: Any) -> dict[str, Any]:
        return message_payload(id="msg_bedrock")

    def vertex_create(**_kwargs: Any) -> dict[str, Any]:
        return message_payload(id="msg_vertex")

    def stream_stub(**_kwargs: Any) -> None:
        return None

    bedrock.messages = SimpleNamespace(create=bedrock_create, stream=stream_stub)
    vertex.messages = SimpleNamespace(create=vertex_create, stream=stream_stub)

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(message_payload())

    wrap_anthropic(bedrock).messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    )
    wrap_anthropic(vertex).messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    )
    wrapped_sync_client(handler).messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=MESSAGES
    )
    bedrock_span, vertex_span, plain_span = memory.span_exporter.get_finished_spans()
    assert attrs(bedrock_span)["gen_ai.provider.name"] == "aws.bedrock"
    assert attrs(vertex_span)["gen_ai.provider.name"] == "gcp.vertex_ai"
    assert attrs(plain_span)["gen_ai.provider.name"] == "anthropic"
