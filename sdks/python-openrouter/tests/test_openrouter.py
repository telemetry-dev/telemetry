from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
import telemetry_dev
from openrouter import OpenRouter
from openrouter.chat import Chat
from openrouter.components import (
    ChatMessagesTypedDict,
    ChatSystemMessage,
    ChatUserMessage,
    EasyInputMessage,
)
from openrouter.embeddings import Embeddings
from openrouter.operations.createembeddings import ContentText, Input
from openrouter.responses import Responses
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

import telemetry_dev_openrouter
from telemetry_dev_openrouter import (
    instrument_openrouter,
    uninstrument_openrouter,
    wrap_open_router,
)

SyncHandler = Callable[[httpx.Request], httpx.Response]
AsyncHandler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]

CHAT_MESSAGES: list[ChatMessagesTypedDict] = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hi"},
]


def only_span(env: SimpleNamespace) -> ReadableSpan:
    spans = env.span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def json_response(data: Mapping[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=data, headers={"Content-Type": "application/json"})


def sse_response(events: Sequence[Mapping[str, Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})


def sync_client(handler: SyncHandler) -> OpenRouter:
    return OpenRouter(
        api_key="test",
        server_url="https://api.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def async_client(handler: AsyncHandler) -> OpenRouter:
    return OpenRouter(
        api_key="test",
        server_url="https://api.test",
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def chat_payload(*, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "gen_123",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4o-mini-2024-07-18",
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Telemetry works."},
                "finish_reason": "stop",
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 6},
            "completion_tokens_details": {"reasoning_tokens": 2},
            "cost": 0.0012,
        },
    }


def chat_chunk(
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": "gen_stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "openai/gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    if error is not None:
        chunk["error"] = error
    return chunk


def chat_stream_events(*, usage: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    events = [
        chat_chunk({"role": "assistant", "content": "Hello"}),
        chat_chunk({"content": " world"}),
        chat_chunk({}, finish_reason="stop"),
    ]
    terminal_usage = (
        usage
        if usage is not None
        else {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost": 0.0004}
    )
    events.append(
        {
            "id": "gen_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-4o-mini",
            "choices": [],
            "usage": terminal_usage,
        }
    )
    return events


def responses_payload(
    *,
    response_id: str = "resp_123",
    text: str = "No jokes.",
    status: str = "completed",
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "completed_at": 2,
        "status": status,
        "model": "openai/gpt-4o-2024-08-06",
        "error": error,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "frequency_penalty": None,
        "presence_penalty": None,
        "temperature": None,
        "top_p": None,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 13,
            "output_tokens": 4,
            "total_tokens": 17,
            "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 5},
            "output_tokens_details": {"reasoning_tokens": 1},
            "cost": 0.002,
        },
    }


def embeddings_payload() -> dict[str, Any]:
    return {
        "id": "embed_123",
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "model": "openai/text-embedding-3-small",
        "usage": {"prompt_tokens": 6, "total_tokens": 6, "cost": 0.0001},
    }


def test_chat_send_maps_messages_usage_cost_and_sampling(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    client = wrap_open_router(sync_client(handler))
    result = client.chat.send(
        model="openai/gpt-4o-mini",
        messages=CHAT_MESSAGES,
        temperature=0.2,
        top_p=0.9,
        top_k=40,
        max_completion_tokens=64,
        stop=["END"],
        seed=7,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )

    assert result.id == "gen_123"
    span = only_span(memory)
    assert span.name == "chat openai/gpt-4o-mini"
    a = attrs(span)
    assert span.status.status_code == StatusCode.UNSET
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "openrouter"
    assert a["gen_ai.request.model"] == "openai/gpt-4o-mini"
    assert a["gen_ai.response.model"] == "openai/gpt-4o-mini-2024-07-18"
    assert a["gen_ai.response.id"] == "gen_123"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["stop"]
    assert a["gen_ai.request.temperature"] == 0.2
    assert a["gen_ai.request.top_p"] == 0.9
    assert a["gen_ai.request.top_k"] == 40
    assert a["gen_ai.request.max_tokens"] == 64
    assert list(cast(Any, a["gen_ai.request.stop_sequences"])) == ["END"]
    assert a["gen_ai.request.seed"] == 7
    assert a["gen_ai.request.frequency_penalty"] == 0.1
    assert a["gen_ai.request.presence_penalty"] == 0.2
    assert a["gen_ai.usage.input_tokens"] == 11
    assert a["gen_ai.usage.output_tokens"] == 7
    assert a["gen_ai.usage.total_tokens"] == 18
    assert a["gen_ai.usage.cache_read.input_tokens"] == 3
    assert a["gen_ai.usage.cache_creation.input_tokens"] == 6
    assert a["gen_ai.usage.reasoning.output_tokens"] == 2
    assert a["gen_ai.usage.cost"] == 0.0012
    assert json.loads(str(a["gen_ai.input.messages"])) == CHAT_MESSAGES
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Telemetry works."}
    ]


async def test_chat_send_async_maps_like_sync(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    client = wrap_open_router(async_client(handler))
    result = await client.chat.send_async(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    assert result.id == "gen_123"
    span = only_span(memory)
    assert span.name == "chat openai/gpt-4o-mini"
    a = attrs(span)
    assert a["gen_ai.provider.name"] == "openrouter"
    assert a["gen_ai.usage.cost"] == 0.0012
    assert a["gen_ai.usage.cache_read.input_tokens"] == 3
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Telemetry works."}
    ]


def test_model_backed_requests_capture_serialized_input(memory: SimpleNamespace) -> None:
    def chat_handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    def responses_handler(request: httpx.Request) -> httpx.Response:
        return json_response(responses_payload())

    def embeddings_handler(request: httpx.Request) -> httpx.Response:
        return json_response(embeddings_payload())

    chat_client = wrap_open_router(sync_client(chat_handler))
    chat_client.chat.send(
        model="openai/gpt-4o-mini",
        messages=[
            ChatSystemMessage(role="system", content="You are helpful."),
            ChatUserMessage(role="user", content="Say hi"),
        ],
    )
    responses_client = wrap_open_router(sync_client(responses_handler))
    responses_client.responses.send(
        model="openai/gpt-4o-mini",
        input=[EasyInputMessage(role="user", content="Tell me a joke")],
        instructions="be terse",
    )
    embeddings_client = wrap_open_router(sync_client(embeddings_handler))
    embeddings_client.embeddings.generate(
        model="openai/text-embedding-3-small",
        input=[Input(content=[ContentText(type="text", text="embed me")])],
    )

    chat_span, responses_span, embeddings_span = memory.span_exporter.get_finished_spans()
    assert json.loads(str(attrs(chat_span)["gen_ai.input.messages"])) == [
        {"content": "You are helpful.", "role": "system"},
        {"content": "Say hi", "role": "user"},
    ]
    responses_attrs = attrs(responses_span)
    assert json.loads(str(responses_attrs["gen_ai.input.messages"])) == [
        {"role": "user", "content": "Tell me a joke"}
    ]
    assert responses_attrs["gen_ai.system_instructions"] == "be terse"
    assert json.loads(str(attrs(embeddings_span)["gen_ai.input.messages"])) == [
        {"content": [{"text": "embed me", "type": "text"}]}
    ]


def test_chat_cost_falls_back_to_upstream_inference_cost(memory: SimpleNamespace) -> None:
    usage = {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
        "cost_details": {
            "upstream_inference_prompt_cost": 0.0003,
            "upstream_inference_completions_cost": 0.0006,
            "upstream_inference_cost": 0.0009,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload(usage=usage))

    client = wrap_open_router(sync_client(handler))
    client.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    assert attrs(only_span(memory))["gen_ai.usage.cost"] == 0.0009


def test_chat_explicit_zero_max_completion_tokens_takes_precedence(
    memory: SimpleNamespace,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    client = wrap_open_router(sync_client(handler))
    client.chat.send(
        model="openai/gpt-4o-mini",
        messages=CHAT_MESSAGES,
        max_completion_tokens=0,
        max_tokens=99,
    )

    assert attrs(only_span(memory))["gen_ai.request.max_tokens"] == 0


def test_chat_without_cost_omits_cost_attribute(memory: SimpleNamespace) -> None:
    usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload(usage=usage))

    client = wrap_open_router(sync_client(handler))
    client.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    assert "gen_ai.usage.cost" not in attrs(only_span(memory))


def test_chat_streaming_accumulates_output_usage_and_ttft(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.chat.send)(
        model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = next(stream)
    chunks = [first, *stream]

    assert len(chunks) == 4
    span = only_span(memory)
    a = attrs(span)
    assert span.name == "chat openai/gpt-4o-mini"
    assert span.status.status_code == StatusCode.UNSET
    assert a["gen_ai.response.id"] == "gen_stream"
    assert a["gen_ai.response.model"] == "openai/gpt-4o-mini"
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello world"}
    ]
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 2
    assert a["gen_ai.usage.total_tokens"] == 7
    assert a["gen_ai.usage.cost"] == 0.0004
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["stop"]


async def test_chat_streaming_async_accumulates_output(memory: SimpleNamespace) -> None:
    usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events(usage=usage))

    client = wrap_open_router(async_client(handler))
    stream = await cast(Any, client.chat.send_async)(
        model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = await stream.__anext__()
    chunks = [first, *[chunk async for chunk in stream]]

    assert len(chunks) == 4
    a = attrs(only_span(memory))
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello world"}
    ]
    assert a["gen_ai.usage.total_tokens"] == 7
    assert "gen_ai.usage.cost" not in a


def test_chat_streaming_captures_reasoning_details_and_refusal(memory: SimpleNamespace) -> None:
    events = [
        chat_chunk(
            {
                "role": "assistant",
                "reasoning": "Because ",
                "reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Because "}],
                "refusal": "I cannot ",
            }
        ),
        chat_chunk(
            {
                "reasoning": "safety.",
                "reasoning_details": [
                    {"type": "reasoning.summary", "index": 1, "summary": "Safety"}
                ],
                "refusal": "help.",
            },
            finish_reason="stop",
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.chat.send)(
        model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    list(stream)

    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "reasoning": "Because safety.",
            "reasoning_details": [
                {"type": "reasoning.text", "index": 0, "text": "Because "},
                {"type": "reasoning.summary", "index": 1, "summary": "Safety"},
            ],
            "refusal": "I cannot help.",
        }
    ]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chat_output_timing_decodes_reasoning_details_once_per_chunk(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, async_mode: bool
) -> None:
    calls: list[telemetry_dev.SpanHandle] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        assert timestamp_ms is not None
        calls.append(handle)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    events = [
        chat_chunk({"role": "assistant", "content": ""}),
        chat_chunk({"reasoning_details": [{"type": "reasoning.text", "text": "Inspect"}]}),
        chat_chunk(
            {
                "content": "once",
                "reasoning_details": [{"type": "reasoning.summary", "summary": "Summary"}],
            }
        ),
        chat_chunk(
            {"reasoning_details": [{"type": "reasoning.summary", "summary": "Summary only"}]}
        ),
        chat_chunk({"reasoning_details": [{"type": "reasoning.summary", "summary": ""}]}),
        chat_chunk(
            {
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "opaque"},
                    {"type": "reasoning.text", "signature": "signed", "text": ""},
                ]
            }
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    if async_mode:
        client = wrap_open_router(async_client(async_handler))
        stream = await cast(Any, client.chat.send_async)(
            model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        assert len([chunk async for chunk in stream]) == len(events)
    else:
        client = wrap_open_router(sync_client(handler))
        stream = cast(Any, client.chat.send)(
            model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        assert len(list(stream)) == len(events)

    assert len(calls) == 3
    assert len(memory.span_exporter.get_finished_spans()) == 1
    assert memory.metric_reader.get_metrics_data() is not None


@pytest.mark.parametrize("async_mode", [False, True])
async def test_responses_output_timing_accepts_code_and_mcp_deltas_until_stream_end(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, async_mode: bool
) -> None:
    calls: list[telemetry_dev.SpanHandle] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        assert timestamp_ms is not None
        calls.append(handle)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    events = [
        {"type": "response.code_interpreter_call_code.delta", "delta": "print(1)"},
        {"type": "response.code_interpreter_call_code.done", "code": "print(1)"},
        {"type": "response.mcp_call_arguments.delta", "delta": '{"city":"Paris"}'},
        {"type": "response.mcp_call_arguments.delta", "delta": ""},
        {"type": "response.mcp_call_arguments.done", "arguments": '{"city":"Paris"}'},
        {"type": "response.audio.transcript.delta", "delta": "hello"},
        {"type": "response.audio.transcript.delta", "delta": ""},
        {"type": "response.audio.transcript.done", "transcript": "hello"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    if async_mode:
        client = wrap_open_router(async_client(async_handler))
        stream = await cast(Any, client.responses.send_async)(
            model="openai/gpt-4o-mini", input="Run", stream=True
        )
        iterator = stream.__aiter__()
        for _ in events:
            await iterator.__anext__()
        await stream.close()
    else:
        client = wrap_open_router(sync_client(handler))
        stream = cast(Any, client.responses.send)(
            model="openai/gpt-4o-mini", input="Run", stream=True
        )
        for _ in events:
            next(stream)
        stream.close()

    assert len(calls) == 3
    assert len(memory.span_exporter.get_finished_spans()) == 1
    assert memory.metric_reader.get_metrics_data() is not None


def test_chat_stream_capture_is_bounded_without_losing_terminal_metadata(
    memory: SimpleNamespace,
) -> None:
    content = "x" * 1024
    deltas = [
        chat_chunk({**({"role": "assistant"} if index == 0 else {}), "content": content})
        for index in range(70)
    ]
    usage = {"prompt_tokens": 3, "completion_tokens": 70, "total_tokens": 73, "cost": 0.04}
    events = [*deltas, chat_chunk({}, finish_reason="stop", usage=usage)]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.chat.send)(
        model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    chunks = list(stream)

    assert len(chunks) == len(events)
    a = attrs(only_span(memory))
    output = json.loads(str(a["gen_ai.output.messages"]))
    assert 0 < len(output[0]["content"]) < len(content) * len(deltas)
    assert a["telemetry.dev.capture.truncated"] is True
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["stop"]
    assert a["gen_ai.usage.total_tokens"] == 73
    assert a["gen_ai.usage.cost"] == 0.04


def test_chat_stream_choice_states_are_bounded(memory: SimpleNamespace) -> None:
    choices = [
        {
            "index": index,
            "delta": {"role": "assistant", "content": "Hi"} if index == 0 else {},
            "finish_reason": None,
        }
        for index in range(1200)
    ]
    event = {
        "id": "gen_stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "openai/gpt-4o-mini",
        "choices": choices,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response([event])

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.chat.send)(
        model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    chunks = list(stream)

    assert len(chunks) == 1
    a = attrs(only_span(memory))
    output = json.loads(str(a["gen_ai.output.messages"]))
    assert len(output) == 1024
    assert output[0] == {"role": "assistant", "content": "Hi"}
    assert a["telemetry.dev.capture.truncated"] is True


def test_chat_stream_chunk_error_sets_error_status(memory: SimpleNamespace) -> None:
    events = [
        chat_chunk(
            {"role": "assistant", "content": "Hel"},
            error={"code": 502, "message": "upstream blew up"},
        )
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.chat.send)(
        model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    chunks = list(stream)

    assert len(chunks) == 1
    assert chunks[0].error is not None
    span = only_span(memory)
    a = attrs(span)
    assert span.status.status_code == StatusCode.ERROR
    assert a["error.type"] == "RuntimeError"
    events_recorded = list(span.events)
    assert len(events_recorded) == 1
    assert events_recorded[0].name == "exception"
    event_attrs = dict(events_recorded[0].attributes or {})
    assert event_attrs["exception.type"] == "RuntimeError"
    assert event_attrs["exception.message"] == "stream error 502: upstream blew up"
    assert json.loads(str(a["gen_ai.output.messages"])) == [{"role": "assistant", "content": "Hel"}]


def test_chat_stream_early_close_ends_span_once(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.chat.send)(
        model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = next(stream)
    stream.close()
    stream.close()

    assert first.choices[0].delta.content == "Hello"
    span = only_span(memory)
    assert span.status.status_code == StatusCode.UNSET
    assert json.loads(str(attrs(span)["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
    assert "gen_ai.usage.total_tokens" not in attrs(span)


async def test_chat_stream_async_early_close_ends_span_once(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_open_router(async_client(handler))
    stream = await cast(Any, client.chat.send_async)(
        model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = await stream.__anext__()
    await stream.close()

    assert first.choices[0].delta.content == "Hello"
    assert only_span(memory).status.status_code == StatusCode.UNSET


async def test_async_stream_close_completes_under_cancellation(memory: SimpleNamespace) -> None:
    responses = [sse_response(chat_stream_events()), sse_response([])]

    async def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    client = wrap_open_router(async_client(handler))
    streams = [
        await cast(Any, client.chat.send_async)(
            model="openai/gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        ),
        await cast(Any, client.responses.send_async)(
            model="openai/gpt-4o-mini", input="close", stream=True
        ),
    ]

    async def close_under_cancellation(stream: Any) -> None:
        inner = stream._inner
        original_close = inner.close
        release_started = asyncio.Event()
        release_allowed = asyncio.Event()
        released = False

        async def slow_close() -> None:
            nonlocal released
            release_started.set()
            await release_allowed.wait()
            result = original_close()
            if hasattr(result, "__await__"):
                await result
            released = True

        inner.close = slow_close
        close_task = asyncio.create_task(stream.close())
        await asyncio.wait_for(release_started.wait(), timeout=5)
        close_task.cancel()
        await asyncio.sleep(0)
        assert not close_task.done()
        release_allowed.set()
        try:
            await asyncio.wait_for(asyncio.shield(close_task), timeout=5)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("close should preserve cancellation")
        assert released

    for stream in streams:
        await close_under_cancellation(stream)

    assert len(memory.span_exporter.get_finished_spans()) == 2


def test_responses_send_maps_fields(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(responses_payload())

    client = wrap_open_router(sync_client(handler))
    result = client.responses.send(
        model="openai/gpt-4o-mini",
        input="Tell me a joke",
        instructions="You must never tell jokes",
        max_output_tokens=50,
        temperature=0.3,
        top_p=0.8,
        top_k=20,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )

    assert result.id == "resp_123"
    span = only_span(memory)
    assert span.name == "chat openai/gpt-4o-mini"
    a = attrs(span)
    assert a["gen_ai.provider.name"] == "openrouter"
    assert a["gen_ai.system_instructions"] == "You must never tell jokes"
    assert a["gen_ai.request.max_tokens"] == 50
    assert a["gen_ai.request.temperature"] == 0.3
    assert a["gen_ai.request.top_p"] == 0.8
    assert a["gen_ai.request.top_k"] == 20
    assert a["gen_ai.request.frequency_penalty"] == 0.1
    assert a["gen_ai.request.presence_penalty"] == 0.2
    assert a["gen_ai.response.id"] == "resp_123"
    assert a["gen_ai.response.model"] == "openai/gpt-4o-2024-08-06"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["stop"]
    assert a["gen_ai.usage.input_tokens"] == 13
    assert a["gen_ai.usage.output_tokens"] == 4
    assert a["gen_ai.usage.total_tokens"] == 17
    assert a["gen_ai.usage.cache_read.input_tokens"] == 2
    assert a["gen_ai.usage.cache_creation.input_tokens"] == 5
    assert a["gen_ai.usage.reasoning.output_tokens"] == 1
    assert a["gen_ai.usage.cost"] == 0.002
    assert json.loads(str(a["gen_ai.output.messages"])) == responses_payload()["output"]


async def test_responses_send_async_maps_like_sync(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return json_response(responses_payload(response_id="resp_async"))

    client = wrap_open_router(async_client(handler))
    result = await client.responses.send_async(
        model="openai/gpt-4o-mini",
        input="Tell me a joke",
    )

    assert result.id == "resp_async"
    a = attrs(only_span(memory))
    assert a["gen_ai.provider.name"] == "openrouter"
    assert a["gen_ai.response.id"] == "resp_async"
    assert a["gen_ai.usage.total_tokens"] == 17
    assert (
        json.loads(str(a["gen_ai.output.messages"]))
        == responses_payload(response_id="resp_async")["output"]
    )


def test_responses_failed_status_sets_error(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            responses_payload(
                status="failed", error={"code": "server_error", "message": "model blew up"}
            )
        )

    client = wrap_open_router(sync_client(handler))
    result = client.responses.send(model="openai/gpt-4o-mini", input="fail")

    assert result.status == "failed"
    span = only_span(memory)
    a = attrs(span)
    assert span.status.status_code == StatusCode.ERROR
    assert a["error.type"] == "RuntimeError"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["failed"]


def test_responses_stream_completed_event_ends_span(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        event = {
            "type": "response.completed",
            "sequence_number": 0,
            "response": responses_payload(response_id="resp_stream", text="Streamed."),
        }
        return sse_response([event])

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.responses.send)(
        model="openai/gpt-4o-mini", input="stream", stream=True
    )
    events = list(stream)

    assert [event.type for event in events] == ["response.completed"]
    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "resp_stream"
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)
    assert a["gen_ai.usage.total_tokens"] == 17
    assert a["gen_ai.usage.cost"] == 0.002
    assert (
        json.loads(str(a["gen_ai.output.messages"]))
        == responses_payload(response_id="resp_stream", text="Streamed.")["output"]
    )


def test_responses_stream_failed_event_sets_error(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        event = {
            "type": "response.failed",
            "sequence_number": 0,
            "response": responses_payload(
                response_id="resp_failed",
                status="failed",
                error={"code": "server_error", "message": "model blew up"},
            ),
        }
        return sse_response([event])

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.responses.send)(model="openai/gpt-4o-mini", input="fail", stream=True)
    events = list(stream)

    assert [event.type for event in events] == ["response.failed"]
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "RuntimeError"


def test_responses_stream_failed_event_retains_numeric_code(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        event = {
            "type": "response.failed",
            "sequence_number": 0,
            "response": responses_payload(
                response_id="resp_failed",
                status="failed",
                error={"code": 502, "message": "model blew up"},
            ),
        }
        return sse_response([event])

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.responses.send)(model="openai/gpt-4o-mini", input="fail", stream=True)
    list(stream)

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    events_recorded = list(span.events)
    assert len(events_recorded) == 1
    assert events_recorded[0].name == "exception"
    event_attrs = dict(events_recorded[0].attributes or {})
    assert event_attrs["exception.type"] == "RuntimeError"
    assert event_attrs["exception.message"] == "response.failed: 502: model blew up"


def test_responses_stream_error_event_retains_numeric_code(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        event = {
            "type": "error",
            "sequence_number": 0,
            "code": 502,
            "message": "provider disconnected",
        }
        return sse_response([event])

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.responses.send)(model="openai/gpt-4o-mini", input="fail", stream=True)
    list(stream)

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    events_recorded = list(span.events)
    assert len(events_recorded) == 1
    event_attrs = dict(events_recorded[0].attributes or {})
    assert event_attrs["exception.type"] == "RuntimeError"
    assert event_attrs["exception.message"] == "response.error: 502: provider disconnected"


async def test_responses_stream_async_completed_event_ends_span(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        event = {
            "type": "response.completed",
            "sequence_number": 0,
            "response": responses_payload(response_id="resp_async_stream", text="Streamed."),
        }
        return sse_response([event])

    client = wrap_open_router(async_client(handler))
    stream = await cast(Any, client.responses.send_async)(
        model="openai/gpt-4o-mini", input="stream", stream=True
    )
    events = [event async for event in stream]

    assert [event.type for event in events] == ["response.completed"]
    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "resp_async_stream"
    assert a["gen_ai.usage.total_tokens"] == 17
    assert (
        json.loads(str(a["gen_ai.output.messages"]))
        == responses_payload(response_id="resp_async_stream", text="Streamed.")["output"]
    )


def test_responses_stream_early_close_captures_final_text_events(memory: SimpleNamespace) -> None:
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_partial",
            "output_index": 0,
            "content_index": 0,
            "delta": "Partial",
            "logprobs": [],
        },
        {
            "type": "response.refusal.delta",
            "sequence_number": 2,
            "item_id": "msg_partial",
            "output_index": 0,
            "content_index": 1,
            "delta": "No",
        },
        {
            "type": "response.reasoning_text.delta",
            "sequence_number": 3,
            "item_id": "reason_partial",
            "output_index": 1,
            "content_index": 0,
            "delta": "Thinking",
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 4,
            "item_id": "reason_partial",
            "output_index": 1,
            "summary_index": 0,
            "delta": "Summary",
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 5,
            "item_id": "msg_partial",
            "output_index": 0,
            "content_index": 0,
            "text": "Final answer",
            "logprobs": [],
        },
        {
            "type": "response.refusal.done",
            "sequence_number": 6,
            "item_id": "msg_partial",
            "output_index": 0,
            "content_index": 1,
            "refusal": "Final refusal",
        },
        {
            "type": "response.reasoning_text.done",
            "sequence_number": 7,
            "item_id": "reason_partial",
            "output_index": 1,
            "content_index": 0,
            "text": "Final reasoning",
        },
        {
            "type": "response.reasoning_summary_text.done",
            "sequence_number": 8,
            "item_id": "reason_partial",
            "output_index": 1,
            "summary_index": 0,
            "text": "Final summary",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.responses.send)(
        model="openai/gpt-4o-mini", input="cancel", stream=True
    )
    for _ in events:
        next(stream)
    stream.close()
    stream.close()

    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {
            "id": "msg_partial",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Final answer", "annotations": []},
                {"type": "refusal", "refusal": "Final refusal"},
            ],
        },
        {
            "id": "reason_partial",
            "type": "reasoning",
            "status": "in_progress",
            "summary": [{"type": "summary_text", "text": "Final summary"}],
            "content": [{"type": "reasoning_text", "text": "Final reasoning"}],
        },
    ]


def test_responses_stream_early_close_retains_partial_text_events(
    memory: SimpleNamespace,
) -> None:
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_partial_only",
            "output_index": 0,
            "content_index": 0,
            "delta": "Partial",
            "logprobs": [],
        },
        {
            "type": "response.refusal.delta",
            "sequence_number": 2,
            "item_id": "msg_partial_only",
            "output_index": 0,
            "content_index": 1,
            "delta": "No",
        },
        {
            "type": "response.reasoning_text.delta",
            "sequence_number": 3,
            "item_id": "reason_partial_only",
            "output_index": 1,
            "content_index": 0,
            "delta": "Thinking",
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 4,
            "item_id": "reason_partial_only",
            "output_index": 1,
            "summary_index": 0,
            "delta": "Summary",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="cancel", stream=True
    )
    for _ in events:
        next(stream)
    stream.close()

    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {
            "id": "msg_partial_only",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Partial", "annotations": []},
                {"type": "refusal", "refusal": "No"},
            ],
        },
        {
            "id": "reason_partial_only",
            "type": "reasoning",
            "status": "in_progress",
            "summary": [{"type": "summary_text", "text": "Summary"}],
            "content": [{"type": "reasoning_text", "text": "Thinking"}],
        },
    ]


@pytest.mark.parametrize("event_type", ["response.created", "response.in_progress"])
def test_response_event_output_survives_responses_stream_early_close(
    memory: SimpleNamespace,
    event_type: str,
) -> None:
    response = responses_payload(response_id=f"resp_{event_type}", status="in_progress")
    response["completed_at"] = None
    response["output"] = [
        {
            "id": "msg_response_event",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Retained output", "annotations": []}],
        }
    ]
    events = [
        {"type": event_type, "sequence_number": 1, "response": response},
        {
            "type": "response.debug",
            "sequence_number": 2,
            "debug": {
                "timings": {"epoch_ms": 10, "event": "adapter_request", "start_ms": 2},
                "echo_upstream_body": {"prompt": "Sensitive prompt"},
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="cancel", stream=True
    )
    for _ in events:
        next(stream)
    stream.close()

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        *cast(list[Any], response["output"]),
        {
            "type": "telemetry.dev.response_stream_event",
            "event_type": "response.debug",
            "payload": {
                "type": "response.debug",
                "sequence_number": 2,
                "debug": {"timings": {"epoch_ms": 10, "event": "adapter_request", "start_ms": 2}},
            },
        },
    ]
    assert "telemetry.dev.capture.truncated" not in a


def test_terminal_responses_output_retains_consumed_provider_events(
    memory: SimpleNamespace,
) -> None:
    terminal = responses_payload(response_id="resp_provider_terminal")
    terminal["output"] = [
        {
            "id": "msg_terminal_provider",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Terminal output", "annotations": []}],
        }
    ]
    provider_event = {
        "type": "response.image_generation_call.completed",
        "sequence_number": 1,
        "item_id": "image_terminal",
        "output_index": 1,
    }
    events = [
        provider_event,
        {"type": "response.completed", "sequence_number": 2, "response": terminal},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="complete", stream=True
    )
    list(stream)

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        *cast(list[Any], terminal["output"]),
        {
            "type": "telemetry.dev.response_stream_event",
            "event_type": provider_event["type"],
            "payload": provider_event,
        },
    ]
    assert "telemetry.dev.capture.truncated" not in a


def test_terminal_responses_output_preserves_provider_event_truncation(
    memory: SimpleNamespace,
) -> None:
    terminal = responses_payload(response_id="resp_provider_truncated")
    terminal["output"] = [
        {
            "id": "msg_terminal_after_provider_truncation",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Terminal output", "annotations": []}],
        }
    ]
    events: list[dict[str, Any]] = [
        {
            "type": "response.image_generation_call.partial_image",
            "sequence_number": 1,
            "item_id": "image_oversized",
            "output_index": 1,
            "partial_image_index": 0,
            "partial_image_b64": "x" * (70 * 1024),
        },
        {"type": "response.completed", "sequence_number": 2, "response": terminal},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="complete", stream=True
    )
    list(stream)

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == terminal["output"]
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_early_close_retains_items_content_annotations_and_arguments(
    memory: SimpleNamespace,
) -> None:
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "function_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "item_id": "function_1",
            "output_index": 0,
            "delta": '{"city":"Par',
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 3,
            "item_id": "function_1",
            "output_index": 0,
            "name": "weather",
            "arguments": '{"city":"Paris"}',
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 4,
            "output_index": 1,
            "item": {
                "id": "message_1",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 5,
            "item_id": "message_1",
            "output_index": 1,
            "content_index": 0,
            "part": {"type": "output_text", "text": "Forecast", "annotations": []},
        },
        {
            "type": "response.output_text.annotation.added",
            "sequence_number": 6,
            "item_id": "message_1",
            "output_index": 1,
            "content_index": 0,
            "annotation_index": 0,
            "annotation": {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 8,
                "title": "Forecast",
                "url": "https://example.test/forecast",
            },
        },
        {
            "type": "response.content_part.done",
            "sequence_number": 7,
            "item_id": "message_1",
            "output_index": 1,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "Forecast ready",
                "annotations": [
                    {
                        "type": "url_citation",
                        "start_index": 0,
                        "end_index": 8,
                        "title": "Forecast",
                        "url": "https://example.test/forecast",
                    }
                ],
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="weather", stream=True
    )
    for _ in events:
        next(stream)
    stream.close()

    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {
            "id": "function_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "weather",
            "arguments": '{"city":"Paris"}',
            "status": "completed",
        },
        {
            "id": "message_1",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [
                {
                    "type": "output_text",
                    "text": "Forecast ready",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "start_index": 0,
                            "end_index": 8,
                            "title": "Forecast",
                            "url": "https://example.test/forecast",
                        }
                    ],
                }
            ],
        },
    ]


def test_responses_stream_capture_rejects_sparse_annotation_indexes(
    memory: SimpleNamespace,
) -> None:
    events: list[dict[str, Any]] = [
        {
            "type": "response.content_part.added",
            "sequence_number": 1,
            "item_id": "message_sparse_annotation",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "Retained", "annotations": []},
        },
        {
            "type": "response.output_text.annotation.added",
            "sequence_number": 2,
            "item_id": "message_sparse_annotation",
            "output_index": 0,
            "content_index": 0,
            "annotation_index": 1_000_000_000,
            "annotation": {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 8,
                "title": "Unsafe",
                "url": "https://example.test/unsafe",
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="annotate", stream=True
    )
    for _ in events:
        next(stream)
    stream.close()

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "id": "message_sparse_annotation",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Retained", "annotations": []}],
        }
    ]
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_boolean_indexes_do_not_collide_with_index_one() -> None:
    state_type = vars(telemetry_dev_openrouter)["_ResponsesStreamState"]
    state = state_type()
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 1,
            "item": {
                "id": "message_one",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Output one", "annotations": []}],
            },
        },
        {
            "type": "response.content_part.added",
            "sequence_number": 2,
            "item_id": "message_zero",
            "output_index": 0,
            "content_index": 1,
            "part": {"type": "output_text", "text": "Content one", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "item_id": "message_zero",
            "output_index": True,
            "content_index": True,
            "delta": "Content zero",
            "logprobs": [],
        },
        {
            "type": "response.reasoning_summary_part.added",
            "sequence_number": 4,
            "item_id": "reasoning_two",
            "output_index": 2,
            "summary_index": 1,
            "part": {"type": "summary_text", "text": "Summary one"},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 5,
            "item_id": "reasoning_two",
            "output_index": 2,
            "summary_index": True,
            "delta": "Summary zero",
        },
    ]

    for event in events:
        state.record(event)

    assert state.partial["output"] == [
        {
            "id": "message_zero",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Content zero", "annotations": []},
                {"type": "output_text", "text": "Content one", "annotations": []},
            ],
        },
        {
            "id": "message_one",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Output one", "annotations": []}],
        },
        {
            "id": "reasoning_two",
            "type": "reasoning",
            "status": "in_progress",
            "summary": [
                {"type": "summary_text", "text": "Summary zero"},
                {"type": "summary_text", "text": "Summary one"},
            ],
            "content": [],
        },
    ]


def test_responses_stream_early_close_retains_custom_tools_and_provider_payloads(
    memory: SimpleNamespace,
) -> None:
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "custom_1",
                "type": "custom_tool_call",
                "call_id": "call_custom",
                "name": "patch",
                "input": "",
            },
        },
        {
            "type": "response.custom_tool_call_input.delta",
            "sequence_number": 2,
            "item_id": "custom_1",
            "output_index": 0,
            "delta": "*** Begin",
        },
        {
            "type": "response.custom_tool_call_input.done",
            "sequence_number": 3,
            "item_id": "custom_1",
            "output_index": 0,
            "input": "*** Begin Patch",
        },
        {
            "type": "response.image_generation_call.partial_image",
            "sequence_number": 4,
            "item_id": "image_1",
            "output_index": 1,
            "partial_image_index": 0,
            "partial_image_b64": "aW1hZ2U=",
        },
        {
            "type": "response.apply_patch_call_operation_diff.delta",
            "sequence_number": 5,
            "item_id": "patch_1",
            "output_index": 2,
            "delta": "*** Begin",
        },
        {
            "type": "response.apply_patch_call_operation_diff.done",
            "sequence_number": 6,
            "item_id": "patch_1",
            "output_index": 2,
            "diff": "*** Begin Patch",
        },
        {
            "type": "response.fusion_call.panel.added",
            "sequence_number": 7,
            "item_id": "fusion_1",
            "output_index": 3,
            "model": "openai/gpt-4o",
        },
        {
            "type": "response.fusion_call.panel.delta",
            "sequence_number": 8,
            "item_id": "fusion_1",
            "output_index": 3,
            "model": "openai/gpt-4o",
            "delta": "Panel",
        },
        {
            "type": "response.fusion_call.panel.reasoning.delta",
            "sequence_number": 9,
            "item_id": "fusion_1",
            "output_index": 3,
            "model": "openai/gpt-4o",
            "delta": "Reasoning",
        },
        {
            "type": "response.fusion_call.panel.completed",
            "sequence_number": 10,
            "item_id": "fusion_1",
            "output_index": 3,
            "model": "openai/gpt-4o",
            "content": "Panel complete",
        },
        {
            "type": "response.fusion_call.panel.failed",
            "sequence_number": 11,
            "item_id": "fusion_2",
            "output_index": 3,
            "model": "anthropic/claude-sonnet-4",
            "error": "provider failed",
            "status_code": 502,
        },
        {
            "type": "response.reasoning_summary_part.added",
            "sequence_number": 12,
            "item_id": "reason_1",
            "output_index": 4,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": "Initial summary"},
        },
        {
            "type": "response.reasoning_summary_part.done",
            "sequence_number": 13,
            "item_id": "reason_1",
            "output_index": 4,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": "Final summary"},
        },
        {
            "type": "response.image_generation_call.in_progress",
            "sequence_number": 14,
            "item_id": "image_1",
            "output_index": 1,
        },
        {
            "type": "response.image_generation_call.generating",
            "sequence_number": 15,
            "item_id": "image_1",
            "output_index": 1,
        },
        {
            "type": "response.image_generation_call.completed",
            "sequence_number": 16,
            "item_id": "image_1",
            "output_index": 1,
        },
        {
            "type": "response.fusion_call.in_progress",
            "sequence_number": 17,
            "item_id": "fusion_1",
            "output_index": 3,
        },
        {
            "type": "response.fusion_call.analysis.in_progress",
            "sequence_number": 18,
            "item_id": "fusion_1",
            "output_index": 3,
            "judge_model": "openai/gpt-4o",
        },
        {
            "type": "response.fusion_call.analysis.completed",
            "sequence_number": 19,
            "item_id": "fusion_1",
            "output_index": 3,
            "analysis": {
                "blind_spots": [],
                "consensus": ["agreed"],
                "contradictions": [],
                "partial_coverage": [],
                "unique_insights": [],
            },
        },
        {
            "type": "response.fusion_call.completed",
            "sequence_number": 20,
            "item_id": "fusion_1",
            "output_index": 3,
        },
        {
            "type": "response.web_search_call.in_progress",
            "sequence_number": 21,
            "item_id": "search_1",
            "output_index": 5,
        },
        {
            "type": "response.web_search_call.searching",
            "sequence_number": 22,
            "item_id": "search_1",
            "output_index": 5,
        },
        {
            "type": "response.web_search_call.completed",
            "sequence_number": 23,
            "item_id": "search_1",
            "output_index": 5,
        },
        {
            "type": "response.debug",
            "sequence_number": 24,
            "debug": {
                "timings": {"epoch_ms": 10, "event": "adapter_request", "start_ms": 2},
                "echo_upstream_body": {"prompt": "Sensitive prompt"},
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="tools", stream=True
    )
    for _ in events:
        next(stream)
    stream.close()

    output = json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"]))
    assert {
        "id": output[0]["id"],
        "type": output[0]["type"],
        "input": output[0]["input"],
    } == {
        "id": "custom_1",
        "type": "custom_tool_call",
        "input": "*** Begin Patch",
    }
    assert output[1]["id"] == "reason_1"
    assert output[1]["type"] == "reasoning"
    assert output[1]["summary"] == [{"type": "summary_text", "text": "Final summary"}]
    payload_events = output[2:]
    assert [event["event_type"] for event in payload_events] == [
        "response.image_generation_call.partial_image",
        "response.apply_patch_call_operation_diff.delta",
        "response.apply_patch_call_operation_diff.done",
        "response.fusion_call.panel.added",
        "response.fusion_call.panel.delta",
        "response.fusion_call.panel.reasoning.delta",
        "response.fusion_call.panel.completed",
        "response.fusion_call.panel.failed",
        "response.image_generation_call.in_progress",
        "response.image_generation_call.generating",
        "response.image_generation_call.completed",
        "response.fusion_call.in_progress",
        "response.fusion_call.analysis.in_progress",
        "response.fusion_call.analysis.completed",
        "response.fusion_call.completed",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
        "response.debug",
    ]
    assert payload_events[0]["payload"]["partial_image_b64"] == "aW1hZ2U="
    assert payload_events[0]["payload"]["partial_image_index"] == 0
    assert payload_events[-1]["payload"] == {
        "type": "response.debug",
        "sequence_number": 24,
        "debug": {"timings": {"epoch_ms": 10, "event": "adapter_request", "start_ms": 2}},
    }


def test_final_responses_text_replaces_delta_truncated_output_with_fresh_budget(
    memory: SimpleNamespace,
) -> None:
    delta = "y" * 1024
    deltas: list[dict[str, Any]] = [
        {
            "type": "response.output_text.delta",
            "sequence_number": index,
            "item_id": "msg_done",
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
            "logprobs": [],
        }
        for index in range(70)
    ]
    events: list[dict[str, Any]] = [
        *deltas,
        {
            "type": "response.output_text.done",
            "sequence_number": len(deltas),
            "item_id": "msg_done",
            "output_index": 0,
            "content_index": 0,
            "text": "Final answer",
            "logprobs": [],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="long answer", stream=True
    )
    assert len(list(stream)) == len(events)

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "id": "msg_done",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Final answer", "annotations": []}],
        }
    ]
    assert "telemetry.dev.capture.truncated" not in a


def test_final_responses_text_preserves_provider_event_truncation(
    memory: SimpleNamespace,
) -> None:
    events: list[dict[str, Any]] = [
        {
            "type": "response.image_generation_call.partial_image",
            "sequence_number": 1,
            "item_id": "image_oversized_before_done",
            "output_index": 1,
            "partial_image_index": 0,
            "partial_image_b64": "x" * (70 * 1024),
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 2,
            "item_id": "msg_done_after_provider_truncation",
            "output_index": 0,
            "content_index": 0,
            "text": "Final answer",
            "logprobs": [],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="long answer", stream=True
    )
    assert len(list(stream)) == len(events)

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "id": "msg_done_after_provider_truncation",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Final answer", "annotations": []}],
        }
    ]
    assert a["telemetry.dev.capture.truncated"] is True


def test_oversized_final_responses_text_preserves_bounded_partial(
    memory: SimpleNamespace,
) -> None:
    delta = "y" * 1024
    deltas: list[dict[str, Any]] = [
        {
            "type": "response.output_text.delta",
            "sequence_number": index,
            "item_id": "msg_done_oversized",
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
            "logprobs": [],
        }
        for index in range(20)
    ]
    events: list[dict[str, Any]] = [
        *deltas,
        {
            "type": "response.output_text.done",
            "sequence_number": len(deltas),
            "item_id": "msg_done_oversized",
            "output_index": 0,
            "content_index": 0,
            "text": "z" * (70 * 1024),
            "logprobs": [],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="long answer", stream=True
    )
    list(stream)

    a = attrs(only_span(memory))
    output = json.loads(str(a["gen_ai.output.messages"]))
    assert output[0]["content"][0]["text"] == delta * len(deltas)
    assert "z" not in output[0]["content"][0]["text"]
    assert a["telemetry.dev.capture.truncated"] is True


def test_small_terminal_responses_output_replaces_delta_truncated_output(
    memory: SimpleNamespace,
) -> None:
    delta = "y" * 1024
    deltas: list[dict[str, Any]] = [
        {
            "type": "response.output_text.delta",
            "sequence_number": index,
            "item_id": "msg_bounded",
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
            "logprobs": [],
        }
        for index in range(70)
    ]
    events: list[dict[str, Any]] = [
        *deltas,
        {
            "type": "response.completed",
            "sequence_number": len(deltas),
            "response": responses_payload(response_id="resp_bounded"),
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    client = wrap_open_router(sync_client(handler))
    stream = cast(Any, client.responses.send)(
        model="openai/gpt-4o-mini", input="long answer", stream=True
    )
    received = list(stream)

    assert len(received) == len(events)
    a = attrs(only_span(memory))
    assert (
        json.loads(str(a["gen_ai.output.messages"]))
        == responses_payload(response_id="resp_bounded")["output"]
    )
    assert "telemetry.dev.capture.truncated" not in a
    assert a["gen_ai.response.id"] == "resp_bounded"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["stop"]
    assert a["gen_ai.usage.total_tokens"] == 17
    assert a["gen_ai.usage.cost"] == 0.002


def test_oversized_terminal_responses_output_preserves_bounded_partial(
    memory: SimpleNamespace,
) -> None:
    delta = "y" * 1024
    deltas: list[dict[str, Any]] = [
        {
            "type": "response.output_text.delta",
            "sequence_number": index,
            "item_id": "msg_oversized",
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
            "logprobs": [],
        }
        for index in range(20)
    ]
    terminal = responses_payload(response_id="resp_oversized")
    terminal["output"] = [
        {
            "id": "msg_terminal",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "z" * (70 * 1024), "annotations": []}],
        }
    ]
    events: list[dict[str, Any]] = [
        *deltas,
        {
            "type": "response.completed",
            "sequence_number": len(deltas),
            "response": terminal,
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    stream = cast(Any, wrap_open_router(sync_client(handler)).responses.send)(
        model="openai/gpt-4o-mini", input="long answer", stream=True
    )
    list(stream)

    a = attrs(only_span(memory))
    output = json.loads(str(a["gen_ai.output.messages"]))
    assert output[0]["content"][0]["text"] == delta * len(deltas)
    assert "z" not in output[0]["content"][0]["text"]
    assert a["telemetry.dev.capture.truncated"] is True
    assert a["gen_ai.response.id"] == "resp_oversized"


def test_embeddings_generate_maps_usage_without_output(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(embeddings_payload())

    client = wrap_open_router(sync_client(handler))
    result = client.embeddings.generate(model="openai/text-embedding-3-small", input="embed me")

    assert not isinstance(result, str)
    assert result.data[0].embedding == [0.1, 0.2]
    span = only_span(memory)
    assert span.name == "embeddings openai/text-embedding-3-small"
    a = attrs(span)
    assert a["gen_ai.operation.name"] == "embeddings"
    assert a["gen_ai.provider.name"] == "openrouter"
    assert a["gen_ai.request.model"] == "openai/text-embedding-3-small"
    assert a["gen_ai.response.model"] == "openai/text-embedding-3-small"
    assert a["gen_ai.response.id"] == "embed_123"
    assert a["gen_ai.input.messages"] == "embed me"
    assert a["gen_ai.usage.input_tokens"] == 6
    assert a["gen_ai.usage.total_tokens"] == 6
    assert a["gen_ai.usage.cost"] == 0.0001
    assert "gen_ai.output.messages" not in a


async def test_embeddings_generate_async_maps_usage(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return json_response(embeddings_payload())

    client = wrap_open_router(async_client(handler))
    result = await client.embeddings.generate_async(
        model="openai/text-embedding-3-small", input="embed me"
    )

    assert not isinstance(result, str)
    assert result.data[0].embedding == [0.1, 0.2]
    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "embeddings"
    assert a["gen_ai.provider.name"] == "openrouter"
    assert a["gen_ai.response.id"] == "embed_123"
    assert a["gen_ai.usage.total_tokens"] == 6
    assert "gen_ai.output.messages" not in a


def transport_failure_client() -> OpenRouter:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport exploded")

    return wrap_open_router(
        OpenRouter(
            api_key="test",
            server_url="https://api.test",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
    )


@pytest.mark.parametrize(
    ("operation", "span_name"),
    [
        ("chat", "chat openai/gpt-4o-mini"),
        ("responses", "chat openai/gpt-4o-mini"),
        ("embeddings", "embeddings openai/text-embedding-3-small"),
    ],
)
def test_request_failure_records_error_span(
    memory: SimpleNamespace, operation: str, span_name: str
) -> None:
    client = transport_failure_client()
    calls: dict[str, Callable[[], Any]] = {
        "chat": lambda: client.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES),
        "responses": lambda: client.responses.send(model="openai/gpt-4o-mini", input="fail"),
        "embeddings": lambda: client.embeddings.generate(
            model="openai/text-embedding-3-small", input="embed me"
        ),
    }

    with pytest.raises(RuntimeError, match="transport exploded"):
        calls[operation]()

    span = only_span(memory)
    assert span.name == span_name
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "RuntimeError"
    event_attrs = dict(next(iter(span.events)).attributes or {})
    assert event_attrs["exception.message"] == "transport exploded"


@pytest.mark.parametrize(
    ("operation", "span_name"),
    [
        ("chat", "chat openai/gpt-4o-mini"),
        ("responses", "chat openai/gpt-4o-mini"),
        ("embeddings", "embeddings openai/text-embedding-3-small"),
    ],
)
async def test_async_request_failure_records_error_span(
    memory: SimpleNamespace, operation: str, span_name: str
) -> None:
    client = transport_failure_client()
    calls: dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {
        "chat": lambda: client.chat.send_async(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES),
        "responses": lambda: client.responses.send_async(model="openai/gpt-4o-mini", input="fail"),
        "embeddings": lambda: client.embeddings.generate_async(
            model="openai/text-embedding-3-small", input="embed me"
        ),
    }

    with pytest.raises(RuntimeError, match="transport exploded"):
        await calls[operation]()

    span = only_span(memory)
    assert span.name == span_name
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "RuntimeError"


def test_wrap_open_router_and_instrument_openrouter_parity(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    wrapped = wrap_open_router(sync_client(handler))
    wrapped.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    instrument_openrouter()
    global_client = sync_client(handler)
    global_client.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    wrapped_span, global_span = memory.span_exporter.get_finished_spans()
    assert wrapped_span.name == global_span.name == "chat openai/gpt-4o-mini"
    assert attrs(wrapped_span) == attrs(global_span)


def test_wrap_open_router_is_idempotent(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    client = sync_client(handler)
    wrap_open_router(client)
    wrap_open_router(client)
    client.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_instrument_openrouter_is_idempotent(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    instrument_openrouter()
    instrument_openrouter()
    client = sync_client(handler)
    client.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_uninstrument_openrouter_restores_originals(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    originals = [
        (Chat, "send"),
        (Chat, "send_async"),
        (Responses, "send"),
        (Responses, "send_async"),
        (Embeddings, "generate"),
        (Embeddings, "generate_async"),
    ]
    before = {target: getattr(cls, name) for cls, name in originals for target in [(cls, name)]}

    instrument_openrouter()
    uninstrument_openrouter()

    for cls, name in originals:
        assert getattr(cls, name) is before[(cls, name)]
    client = sync_client(handler)
    result = client.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    assert result.id == "gen_123"
    assert len(memory.span_exporter.get_finished_spans()) == 0


def test_fail_open_without_telemetry_init() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_payload())

    client = wrap_open_router(sync_client(handler))
    result = client.chat.send(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    assert result.id == "gen_123"
    assert telemetry_dev.get_client() is None
