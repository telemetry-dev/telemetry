from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
import telemetry_dev
from openai import AsyncOpenAI, OpenAI
from openai.resources.chat.completions.completions import AsyncCompletions, Completions
from openai.resources.embeddings import AsyncEmbeddings, Embeddings
from openai.resources.responses.responses import AsyncResponses, Responses
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from telemetry_dev_openai import instrument_openai, uninstrument_openai, wrap_openai

SyncHandler = Callable[[httpx.Request], httpx.Response]
AsyncHandler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]

CHAT_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hi"},
]


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


def sse_response(events: Sequence[Mapping[str, Any]]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})


def named_sse_response(events: Sequence[tuple[str, Mapping[str, Any]]]) -> httpx.Response:
    body = "".join(f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})


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


def failing_sync_sse_response(event: Mapping[str, Any], message: str) -> httpx.Response:
    first_chunk = f"data: {json.dumps(event)}\n\n".encode()
    return httpx.Response(
        200,
        stream=FailingSyncByteStream(first_chunk, message),
        headers={"Content-Type": "text/event-stream"},
    )


def failing_async_sse_response(event: Mapping[str, Any], message: str) -> httpx.Response:
    first_chunk = f"data: {json.dumps(event)}\n\n".encode()
    return httpx.Response(
        200,
        stream=FailingAsyncByteStream(first_chunk, message),
        headers={"Content-Type": "text/event-stream"},
    )


def sync_client(handler: SyncHandler) -> OpenAI:
    return OpenAI(
        api_key="test",
        base_url="https://api.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def async_client(handler: AsyncHandler) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="test",
        base_url="https://api.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def chat_completion(
    *,
    choices: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl_123",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-2024-08-06",
        "choices": choices
        or [
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
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


def chat_stream_events() -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}}],
        },
        {
            "id": "chatcmpl_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"index": 0, "delta": {"content": " world"}}],
        },
        {
            "id": "chatcmpl_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl_stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    ]


def oversized_chat_stream_events() -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl_bounded",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": "x" * 100}}],
        }
        for _ in range(1100)
    ] + [
        {
            "id": "chatcmpl_bounded",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "chatcmpl_bounded",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    ]


def response_payload(*, response_id: str = "resp_123", text: str = "No jokes.") -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-2024-08-06",
        "output": [
            {
                "id": "msg_123",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 13,
            "output_tokens": 4,
            "total_tokens": 17,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    }


def embeddings_payload() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 6, "total_tokens": 6},
    }


class ParsedAnswer(BaseModel):
    answer: str


def test_chat_completion_maps_native_messages_usage_finish_provider_and_sampling(
    memory: SimpleNamespace,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return json_response(chat_completion())

    client = wrap_openai(sync_client(handler))
    completion = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        temperature=0.2,
        top_p=0.9,
        max_completion_tokens=64,
        stop=["END"],
        seed=7,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )

    assert completion.id == "chatcmpl_123"
    assert requests[0]["messages"] == CHAT_MESSAGES
    span = only_span(memory)
    assert span.name == "chat gpt-4o-mini"
    a = attrs(span)
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o-mini"
    assert a["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert a["gen_ai.response.id"] == "chatcmpl_123"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["stop"]
    assert a["gen_ai.request.temperature"] == 0.2
    assert a["gen_ai.request.top_p"] == 0.9
    assert a["gen_ai.request.max_tokens"] == 64
    assert list(cast(Any, a["gen_ai.request.stop_sequences"])) == ["END"]
    assert a["gen_ai.request.seed"] == 7
    assert a["gen_ai.request.frequency_penalty"] == 0.1
    assert a["gen_ai.request.presence_penalty"] == 0.2
    assert a["gen_ai.usage.input_tokens"] == 11
    assert a["gen_ai.usage.output_tokens"] == 7
    assert a["gen_ai.usage.total_tokens"] == 18
    assert a["gen_ai.usage.cache_read.input_tokens"] == 3
    assert a["gen_ai.usage.reasoning.output_tokens"] == 2
    assert json.loads(str(a["gen_ai.input.messages"])) == CHAT_MESSAGES
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Telemetry works."}
    ]


@pytest.mark.parametrize(
    ("base_url", "expected_provider"),
    [
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://api.openrouter.ai/api/v1", "openrouter"),
        ("https://openrouter.ai./api/v1", "openrouter"),
        ("https://api.openrouter.ai./api/v1", "openrouter"),
        ("https://openrouter.ai.example.com/api/v1", "openai"),
    ],
)
def test_openrouter_base_url_reports_expected_provider(
    memory: SimpleNamespace,
    base_url: str,
    expected_provider: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_completion())

    client = wrap_openai(
        OpenAI(
            api_key="test",
            base_url=base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )
    cast(Any, client.chat.completions.create)(model="openai/gpt-4o-mini", messages=CHAT_MESSAGES)

    span = only_span(memory)
    assert attrs(span)["gen_ai.provider.name"] == expected_provider


def test_wrap_openai_resolves_provider_from_current_base_url_for_each_operation(
    memory: SimpleNamespace,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return json_response(chat_completion())
        events: list[tuple[str, Mapping[str, Any]]] = [
            (
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        "id": "resp_dynamic_provider",
                        "status": "in_progress",
                        "output": [],
                    },
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": response_payload(response_id="resp_dynamic_provider"),
                },
            ),
        ]
        return named_sse_response(events)

    client = wrap_openai(
        OpenAI(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    client.base_url = "https://api.openai.com/v1"
    cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)
    client.base_url = "https://openrouter.ai/api/v1"
    with cast(Any, client.responses.stream)(response_id="resp_dynamic_provider") as stream:
        list(stream)

    chat_span, retrieve_span = memory.span_exporter.get_finished_spans()
    assert attrs(chat_span)["gen_ai.provider.name"] == "openai"
    assert attrs(retrieve_span)["gen_ai.provider.name"] == "openrouter"


def test_chat_completion_preserves_tool_calls_and_multiple_choices(
    memory: SimpleNamespace,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        if body.get("n") == 2:
            return json_response(
                chat_completion(
                    choices=[
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "first"},
                            "finish_reason": "stop",
                        },
                        {
                            "index": 1,
                            "message": {"role": "assistant", "content": "second"},
                            "finish_reason": "stop",
                        },
                    ]
                )
            )
        return json_response(
            chat_completion(
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location":"Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            )
        )

    client = wrap_openai(sync_client(handler))
    cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)
    cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES, n=2)

    tool_span, choices_span = memory.span_exporter.get_finished_spans()
    tool_output = json.loads(str(attrs(tool_span)["gen_ai.output.messages"]))
    assert tool_output == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Paris"}',
                    },
                }
            ],
        }
    ]
    assert list(cast(Any, attrs(tool_span)["gen_ai.response.finish_reasons"])) == ["tool_calls"]
    choices_output = json.loads(str(attrs(choices_span)["gen_ai.output.messages"]))
    assert choices_output == [
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]


def test_chat_streaming_injects_usage_when_opted_in_and_filters_synthetic_chunk(
    memory: SimpleNamespace,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return sse_response(chat_stream_events())

    client = wrap_openai(sync_client(handler), inject_stream_usage=True)
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    visible_chunks = list(stream)

    assert requests[0]["stream_options"] == {"include_usage": True}
    assert len(visible_chunks) == 3
    assert all(len(chunk.choices) == 1 for chunk in visible_chunks)
    span = only_span(memory)
    a = attrs(span)
    assert a["gen_ai.response.id"] == "chatcmpl_stream"
    assert a["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello world"}
    ]
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 2
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["stop"]


def test_chat_streaming_default_leaves_request_unchanged(memory: SimpleNamespace) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return sse_response([event for event in chat_stream_events() if event["choices"]])

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    visible_chunks = list(stream)

    assert "stream_options" not in requests[0]
    assert len(visible_chunks) == 3
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello world"}
    ]
    assert "gen_ai.usage.total_tokens" not in a


def test_chat_streaming_bounds_retained_state_without_dropping_metadata(
    memory: SimpleNamespace,
) -> None:
    source_events = oversized_chat_stream_events()
    source_content = "".join(
        cast(str, event["choices"][0]["delta"].get("content", ""))
        for event in source_events
        if event["choices"]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(source_events)

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    delivered = list(stream)
    budget = stream._budget
    state = stream._states[0]

    assert len(delivered) == len(source_events)
    assert budget.truncated is True
    assert budget.bytes_used <= budget.max_bytes
    assert source_content.startswith(state.content)
    assert len(state.content) < len(source_content)
    assert state.role == "assistant"
    a = attrs(only_span(memory))
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["gen_ai.usage.total_tokens"] == 7


def test_chat_streaming_preserves_explicit_usage_chunk(memory: SimpleNamespace) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return sse_response(chat_stream_events())

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        stream=True,
        stream_options={"include_usage": True},
    )
    visible_chunks = list(stream)

    assert requests[0]["stream_options"] == {"include_usage": True}
    assert len(visible_chunks[-1].choices) == 0
    assert visible_chunks[-1].usage.total_tokens == 7
    assert attrs(only_span(memory))["gen_ai.usage.total_tokens"] == 7


def test_chat_stream_close_ends_partial_span(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = next(stream)
    stream.close()

    assert first.choices[0].delta.content == "Hello"
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
    assert "gen_ai.usage.total_tokens" not in a


def test_responses_create_and_stream_map_instructions_and_completed_event(
    memory: SimpleNamespace,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        requests.append(body)
        if body.get("stream") is True:
            event = {
                "type": "response.completed",
                "sequence_number": 0,
                "response": response_payload(response_id="resp_stream", text="Streamed."),
            }
            return named_sse_response([("response.completed", event)])
        return json_response(response_payload())

    client = wrap_openai(sync_client(handler))
    cast(Any, client.responses.create)(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": "Tell me a joke"}],
        instructions="You must never tell jokes",
        max_output_tokens=50,
        temperature=0.3,
        top_p=0.8,
    )
    stream = cast(Any, client.responses.create)(model="gpt-4o-mini", input="stream", stream=True)
    assert [event.type for event in stream] == ["response.completed"]

    first_span, stream_span = memory.span_exporter.get_finished_spans()
    first_attrs = attrs(first_span)
    assert first_attrs["gen_ai.system_instructions"] == "You must never tell jokes"
    assert first_attrs["gen_ai.request.max_tokens"] == 50
    assert first_attrs["gen_ai.request.temperature"] == 0.3
    assert first_attrs["gen_ai.request.top_p"] == 0.8
    assert json.loads(str(first_attrs["gen_ai.input.messages"])) == [
        {"role": "user", "content": "Tell me a joke"}
    ]
    assert json.loads(str(first_attrs["gen_ai.output.messages"])) == response_payload()["output"]
    assert first_attrs["gen_ai.usage.cache_read.input_tokens"] == 2
    assert first_attrs["gen_ai.usage.reasoning.output_tokens"] == 1
    assert list(cast(Any, first_attrs["gen_ai.response.finish_reasons"])) == ["stop"]
    stream_attrs = attrs(stream_span)
    assert stream_attrs["gen_ai.response.id"] == "resp_stream"
    assert (
        json.loads(str(stream_attrs["gen_ai.output.messages"]))
        == response_payload(response_id="resp_stream", text="Streamed.")["output"]
    )
    assert isinstance(stream_attrs["gen_ai.response.time_to_first_chunk"], float)
    assert requests[1]["stream"] is True


def test_responses_stream_existing_response_is_instrumented(memory: SimpleNamespace) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        events: list[tuple[str, Mapping[str, Any]]] = [
            (
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {"id": "resp_existing", "status": "in_progress", "output": []},
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": response_payload(response_id="resp_existing", text="Streamed."),
                },
            ),
        ]
        return named_sse_response(events)

    client = wrap_openai(sync_client(handler))
    with cast(Any, client.responses.stream)(response_id="resp_existing") as stream:
        events = list(stream)

    assert [event.type for event in events] == ["response.created", "response.completed"]
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/v1/responses/resp_existing"
    span = only_span(memory)
    a = attrs(span)
    assert a["gen_ai.response.id"] == "resp_existing"
    assert (
        json.loads(str(a["gen_ai.output.messages"]))
        == response_payload(
            response_id="resp_existing",
            text="Streamed.",
        )["output"]
    )


def test_responses_stream_bounds_output_without_dropping_terminal_metadata(
    memory: SimpleNamespace,
) -> None:
    retained = response_payload(response_id="resp_bounded", text="prefix")
    retained["status"] = "in_progress"
    # 70k chars exceed the SDK's 64KiB default capture budget, so the completed
    # snapshot's output must be dropped while its metadata still lands.
    completed = response_payload(response_id="resp_bounded", text="x" * 70_000)

    def handler(request: httpx.Request) -> httpx.Response:
        events: list[tuple[str, Mapping[str, Any]]] = [
            (
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": retained,
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": completed,
                },
            ),
        ]
        return named_sse_response(events)

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.responses.create)(model="gpt-4o-mini", input="bounded", stream=True)
    assert [event.type for event in stream] == ["response.created", "response.completed"]

    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "resp_bounded"
    assert a["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert a["gen_ai.usage.total_tokens"] == 17
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "id": "msg_123",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "prefix", "annotations": []}],
        }
    ]


def test_embeddings_map_usage_without_output(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request_json(request)["input"] == "embed me"
        return json_response(embeddings_payload())

    client = wrap_openai(sync_client(handler))
    result = cast(Any, client.embeddings.create)(model="text-embedding-3-small", input="embed me")

    assert result.data[0].embedding == [0.1, 0.2]
    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "embeddings"
    assert a["gen_ai.request.model"] == "text-embedding-3-small"
    assert a["gen_ai.response.model"] == "text-embedding-3-small"
    assert a["gen_ai.input.messages"] == "embed me"
    assert a["gen_ai.usage.input_tokens"] == 6
    assert a["gen_ai.usage.total_tokens"] == 6
    assert "gen_ai.output.messages" not in a


def test_chat_and_responses_parse_are_instrumented(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return json_response(
                chat_completion(
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"answer":"chat parsed"}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                )
            )
        return json_response(
            response_payload(response_id="resp_parse", text='{"answer":"response parsed"}')
        )

    client = wrap_openai(sync_client(handler))
    chat = cast(Any, client.chat.completions.parse)(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        response_format=ParsedAnswer,
    )
    response = cast(Any, client.responses.parse)(
        model="gpt-4o-mini",
        input="parse this",
        text_format=ParsedAnswer,
    )

    assert chat.choices[0].message.parsed == ParsedAnswer(answer="chat parsed")
    assert response.output_parsed == ParsedAnswer(answer="response parsed")
    chat_span, response_span = memory.span_exporter.get_finished_spans()
    assert attrs(chat_span)["gen_ai.response.id"] == "chatcmpl_123"
    assert attrs(response_span)["gen_ai.response.id"] == "resp_parse"


async def test_async_chat_non_stream_and_stream(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        if body.get("stream") is True:
            return sse_response(chat_stream_events())
        return json_response(chat_completion())

    client = wrap_openai(async_client(handler), inject_stream_usage=True)
    completion = await cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES
    )
    stream = await cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    visible_chunks = [chunk async for chunk in stream]
    await client.close()

    assert completion.id == "chatcmpl_123"
    assert len(visible_chunks) == 3
    non_stream_span, stream_span = memory.span_exporter.get_finished_spans()
    assert attrs(non_stream_span)["gen_ai.response.id"] == "chatcmpl_123"
    assert json.loads(str(attrs(stream_span)["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello world"}
    ]
    assert attrs(stream_span)["gen_ai.usage.total_tokens"] == 7


async def test_async_chat_streaming_bounds_retained_state_without_dropping_metadata(
    memory: SimpleNamespace,
) -> None:
    source_events = oversized_chat_stream_events()
    source_content = "".join(
        cast(str, event["choices"][0]["delta"].get("content", ""))
        for event in source_events
        if event["choices"]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(source_events)

    client = wrap_openai(async_client(handler))
    stream = await cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    delivered = [chunk async for chunk in stream]
    budget = stream._budget
    state = stream._states[0]
    await client.close()

    assert len(delivered) == len(source_events)
    assert budget.truncated is True
    assert budget.bytes_used <= budget.max_bytes
    assert source_content.startswith(state.content)
    assert len(state.content) < len(source_content)
    assert state.role == "assistant"
    a = attrs(only_span(memory))
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["gen_ai.usage.total_tokens"] == 7


async def test_async_responses_create_and_embeddings_map_usage_like_sync(
    memory: SimpleNamespace,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/responses"):
            return json_response(response_payload())
        assert request.url.path.endswith("/embeddings")
        assert request_json(request)["input"] == "embed me"
        return json_response(embeddings_payload())

    client = wrap_openai(async_client(handler))
    response = await cast(Any, client.responses.create)(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": "Tell me a joke"}],
        instructions="You must never tell jokes",
    )
    embedding = await cast(Any, client.embeddings.create)(
        model="text-embedding-3-small", input="embed me"
    )
    await client.close()

    assert response.id == "resp_123"
    assert embedding.data[0].embedding == [0.1, 0.2]
    response_span, embedding_span = memory.span_exporter.get_finished_spans()
    response_attrs = attrs(response_span)
    assert response_span.name == "chat gpt-4o-mini"
    assert response_attrs["gen_ai.operation.name"] == "chat"
    assert response_attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert response_attrs["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert response_attrs["gen_ai.response.id"] == "resp_123"
    assert response_attrs["gen_ai.usage.input_tokens"] == 13
    assert response_attrs["gen_ai.usage.output_tokens"] == 4
    assert response_attrs["gen_ai.usage.total_tokens"] == 17
    embedding_attrs = attrs(embedding_span)
    assert embedding_span.name == "embeddings text-embedding-3-small"
    assert embedding_attrs["gen_ai.operation.name"] == "embeddings"
    assert embedding_attrs["gen_ai.request.model"] == "text-embedding-3-small"
    assert embedding_attrs["gen_ai.response.model"] == "text-embedding-3-small"
    assert embedding_attrs["gen_ai.usage.input_tokens"] == 6
    assert embedding_attrs["gen_ai.usage.total_tokens"] == 6


async def test_async_chat_stream_iteration_error_records_one_error_span(
    memory: SimpleNamespace,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return failing_async_sse_response(chat_stream_events()[0], "async stream broke")

    client = wrap_openai(async_client(handler))
    stream = await cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = await stream.__anext__()
    assert first.choices[0].delta.content == "Hello"

    with pytest.raises(Exception) as exc_info:
        await stream.__anext__()
    await client.close()

    assert "async stream broke" in str(exc_info.value)
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == type(exc_info.value).__name__
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
    events = list(span.events)
    assert len(events) == 1
    event_attrs = dict(events[0].attributes or {})
    assert events[0].name == "exception"
    assert event_attrs["exception.type"] == type(exc_info.value).__name__
    assert event_attrs["exception.message"] == str(exc_info.value)


def test_chat_stream_early_break_ends_span_once(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    for chunk in stream:
        assert chunk.choices[0].delta.content == "Hello"
        break

    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 1
    a = attrs(spans[0])
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]


def test_chat_stream_iteration_error_records_one_error_span(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return failing_sync_sse_response(chat_stream_events()[0], "sync stream broke")

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = next(stream)
    assert first.choices[0].delta.content == "Hello"

    with pytest.raises(Exception) as exc_info:
        next(stream)

    assert "sync stream broke" in str(exc_info.value)
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == type(exc_info.value).__name__
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
    events = list(span.events)
    assert len(events) == 1
    event_attrs = dict(events[0].attributes or {})
    assert events[0].name == "exception"
    assert event_attrs["exception.type"] == type(exc_info.value).__name__
    assert event_attrs["exception.message"] == str(exc_info.value)


def test_responses_stream_failed_sets_error_status(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        event = {
            "type": "response.failed",
            "sequence_number": 0,
            "response": {
                **response_payload(response_id="resp_failed", text=""),
                "status": "failed",
                "error": {"code": "server_error", "message": "model blew up"},
            },
        }
        return named_sse_response([("response.failed", event)])

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.responses.create)(model="gpt-4o-mini", input="fail", stream=True)
    events = list(stream)
    assert [event.type for event in events] == ["response.failed"]

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR


def test_responses_stream_bare_error_event_sets_error_status(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                {
                    "type": "error",
                    "code": "server_error",
                    "message": "model blew up",
                    "sequence_number": 0,
                }
            ]
        )

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.responses.create)(model="gpt-4o-mini", input="fail", stream=True)

    assert [event.type for event in stream] == ["error"]
    assert only_span(memory).status.status_code == StatusCode.ERROR


def test_responses_create_failed_body_records_error_span(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            {
                **response_payload(response_id="resp_failed", text=""),
                "status": "failed",
                "error": {"code": "server_error", "message": "model blew up"},
            }
        )

    client = wrap_openai(sync_client(handler))
    cast(Any, client.responses.create)(model="gpt-4o-mini", input="fail")

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == "RuntimeError"
    assert a["gen_ai.response.id"] == "resp_failed"
    events = list(span.events)
    assert len(events) == 1
    assert events[0].name == "exception"
    event_attrs = dict(events[0].attributes or {})
    assert event_attrs["exception.message"] == "response.failed: server_error: model blew up"


def test_instrument_openai_threads_inject_stream_usage(memory: SimpleNamespace) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return sse_response(chat_stream_events())

    instrument_openai(inject_stream_usage=True)
    try:
        client = sync_client(handler)
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        visible_chunks = list(stream)
    finally:
        uninstrument_openai()

    assert requests[0]["stream_options"] == {"include_usage": True}
    assert len(visible_chunks) == 3
    assert attrs(only_span(memory))["gen_ai.usage.total_tokens"] == 7


def test_double_instrument_is_idempotent_and_uninstrument_restores_once(
    memory: SimpleNamespace,
) -> None:
    originals = (
        Completions.create,
        Completions.parse,
        AsyncCompletions.create,
        AsyncCompletions.parse,
        Responses.create,
        Responses.retrieve,
        Responses.parse,
        AsyncResponses.create,
        AsyncResponses.parse,
        AsyncResponses.retrieve,
        Embeddings.create,
        AsyncEmbeddings.create,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_completion())

    instrument_openai()
    try:
        instrumented_create = Completions.create
        assert instrumented_create is not originals[0]
        instrument_openai()
        assert Completions.create is instrumented_create
        instrumented = sync_client(handler)
        cast(Any, instrumented.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)
        assert len(memory.span_exporter.get_finished_spans()) == 1
    finally:
        uninstrument_openai()

    assert (
        Completions.create,
        Completions.parse,
        AsyncCompletions.create,
        AsyncCompletions.parse,
        Responses.create,
        Responses.retrieve,
        Responses.parse,
        AsyncResponses.create,
        AsyncResponses.parse,
        AsyncResponses.retrieve,
        Embeddings.create,
        AsyncEmbeddings.create,
    ) == originals
    uninstrumented = sync_client(handler)
    cast(Any, uninstrumented.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)
    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_wrap_openai_is_idempotent(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_completion())

    client = sync_client(handler)
    wrapped_once = wrap_openai(client)
    wrapped_create = wrapped_once.chat.completions.create
    wrapped_twice = wrap_openai(wrapped_once)

    assert wrapped_twice is client
    assert client.chat.completions.create is wrapped_create
    cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)
    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_chat_completions_in_one_session_share_one_trace(
    make: Callable[..., SimpleNamespace],
) -> None:
    # Session traces are keyed by API key; a keyless client never joins one.
    memory = make(api_key="td_test_key")

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_completion())

    client = wrap_openai(sync_client(handler))
    with telemetry_dev.propagate_attributes(session_id="sess-1"):
        cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)
        cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)
    cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)

    first, second, solo = memory.span_exporter.get_finished_spans()
    assert first.context is not None and second.context is not None and solo.context is not None
    assert first.context.trace_id == second.context.trace_id
    assert first.parent is not None and second.parent is not None
    assert first.parent.span_id == second.parent.span_id
    assert solo.context.trace_id != first.context.trace_id
    assert solo.parent is None


def test_openai_wrappers_fail_open_without_telemetry_init() -> None:
    telemetry_dev.shutdown()
    uninstrument_openai()

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_completion())

    wrapped = wrap_openai(sync_client(handler))
    wrapped_response = cast(Any, wrapped.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES
    )

    instrument_openai()
    try:
        instrumented = sync_client(handler)
        instrumented_response = cast(Any, instrumented.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES
        )
    finally:
        uninstrument_openai()
        telemetry_dev.shutdown()

    uninstrumented = sync_client(handler)
    uninstrumented_response = cast(Any, uninstrumented.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES
    )

    assert wrapped_response.choices[0].message.content == "Telemetry works."
    assert instrumented_response.choices[0].message.content == "Telemetry works."
    assert uninstrumented_response.choices[0].message.content == "Telemetry works."


def test_wrap_openai_shadows_global_instrumentation_and_survives_uninstrument(
    memory: SimpleNamespace,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(chat_completion())

    instrument_openai()
    try:
        client = wrap_openai(sync_client(handler))
        first = cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES
        )
        assert first.choices[0].message.content == "Telemetry works."
        assert len(memory.span_exporter.get_finished_spans()) == 1
    finally:
        uninstrument_openai()

    second = cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)

    spans = memory.span_exporter.get_finished_spans()
    assert second.choices[0].message.content == "Telemetry works."
    assert len(spans) == 2
    assert all(span.name == "chat gpt-4o-mini" for span in spans)


def test_chat_completion_finish_reasons_preserve_single_and_multi_choice_order(
    memory: SimpleNamespace,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        if body.get("n") == 2:
            return json_response(
                chat_completion(
                    choices=[
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "first"},
                            "finish_reason": "stop",
                        },
                        {
                            "index": 1,
                            "message": {"role": "assistant", "content": "second"},
                            "finish_reason": "length",
                        },
                    ]
                )
            )
        return json_response(chat_completion())

    client = wrap_openai(sync_client(handler))
    cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES)
    cast(Any, client.chat.completions.create)(model="gpt-4o-mini", messages=CHAT_MESSAGES, n=2)

    single_span, multi_span = memory.span_exporter.get_finished_spans()
    assert attrs(single_span)["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs(multi_span)["gen_ai.response.finish_reasons"] == ("stop", "length")


def test_chat_streaming_finish_reasons_include_all_choices(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                {
                    "id": "chatcmpl_stream_choices",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": "first"}},
                        {"index": 1, "delta": {"role": "assistant", "content": "second"}},
                    ],
                },
                {
                    "id": "chatcmpl_stream_choices",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"},
                        {"index": 1, "delta": {}, "finish_reason": "length"},
                    ],
                },
            ]
        )

    client = wrap_openai(sync_client(handler))
    list(
        cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
    )

    assert attrs(only_span(memory))["gen_ai.response.finish_reasons"] == ("stop", "length")


def test_chat_streaming_reconstructs_split_tool_call_arguments(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                {
                    "id": "chatcmpl_stream_tools",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_weather",
                                        "type": "function",
                                        "function": {"name": "get_weather"},
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "id": "chatcmpl_stream_tools",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '{"location"'},
                                    }
                                ]
                            },
                        }
                    ],
                },
                {
                    "id": "chatcmpl_stream_tools",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": ':"Paris"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            ]
        )

    client = wrap_openai(sync_client(handler))
    list(
        cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
    )

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location":"Paris"}',
                    },
                }
            ],
        }
    ]
    assert a["gen_ai.response.finish_reasons"] == ("tool_calls",)


async def test_async_responses_stream_completed_and_failed_events(memory: SimpleNamespace) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = request_json(request)
        if body["input"] == "fail":
            event = {
                "type": "response.failed",
                "sequence_number": 0,
                "response": {
                    **response_payload(response_id="resp_failed", text=""),
                    "status": "failed",
                    "error": {"code": "server_error", "message": "model blew up"},
                },
            }
            return named_sse_response([("response.failed", event)])
        event = {
            "type": "response.completed",
            "sequence_number": 0,
            "response": response_payload(response_id="resp_stream", text="Streamed."),
        }
        return named_sse_response([("response.completed", event)])

    client = wrap_openai(async_client(handler))
    completed_stream = await cast(Any, client.responses.create)(
        model="gpt-4o-mini", input="ok", stream=True
    )
    completed_events = [event async for event in completed_stream]
    failed_stream = await cast(Any, client.responses.create)(
        model="gpt-4o-mini", input="fail", stream=True
    )
    failed_events = [event async for event in failed_stream]
    await client.close()

    assert [event.type for event in completed_events] == ["response.completed"]
    assert [event.type for event in failed_events] == ["response.failed"]
    completed_span, failed_span = memory.span_exporter.get_finished_spans()
    completed_attrs = attrs(completed_span)
    assert completed_span.status.status_code == StatusCode.UNSET
    assert completed_attrs["gen_ai.response.id"] == "resp_stream"
    assert completed_attrs["gen_ai.usage.input_tokens"] == 13
    assert completed_attrs["gen_ai.usage.output_tokens"] == 4
    assert completed_attrs["gen_ai.usage.total_tokens"] == 17
    assert failed_span.status.status_code == StatusCode.ERROR
    assert attrs(failed_span)["error.type"] == "RuntimeError"


async def test_async_responses_stream_bounds_output_without_dropping_terminal_metadata(
    memory: SimpleNamespace,
) -> None:
    retained = response_payload(response_id="resp_bounded_async", text="prefix")
    retained["status"] = "in_progress"
    # 70k chars exceed the SDK's 64KiB default capture budget, so the completed
    # snapshot's output must be dropped while its metadata still lands.
    completed = response_payload(response_id="resp_bounded_async", text="x" * 70_000)

    async def handler(request: httpx.Request) -> httpx.Response:
        events: list[tuple[str, Mapping[str, Any]]] = [
            (
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": retained,
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": completed,
                },
            ),
        ]
        return named_sse_response(events)

    client = wrap_openai(async_client(handler))
    stream = await cast(Any, client.responses.create)(
        model="gpt-4o-mini", input="bounded", stream=True
    )
    assert [event.type async for event in stream] == [
        "response.created",
        "response.completed",
    ]
    await client.close()

    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "resp_bounded_async"
    assert a["gen_ai.response.model"] == "gpt-4o-2024-08-06"
    assert a["gen_ai.usage.total_tokens"] == 17
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {
            "id": "msg_123",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "prefix", "annotations": []}],
        }
    ]


async def test_async_responses_stream_existing_response_is_instrumented(
    memory: SimpleNamespace,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        events: list[tuple[str, Mapping[str, Any]]] = [
            (
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {"id": "resp_existing", "status": "in_progress", "output": []},
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "sequence_number": 1,
                    "response": response_payload(response_id="resp_existing", text="Streamed."),
                },
            ),
        ]
        return named_sse_response(events)

    client = wrap_openai(async_client(handler))
    async with cast(Any, client.responses.stream)(response_id="resp_existing") as stream:
        events = [event async for event in stream]
    await client.close()

    assert [event.type for event in events] == ["response.created", "response.completed"]
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/v1/responses/resp_existing"
    span = only_span(memory)
    a = attrs(span)
    assert a["gen_ai.response.id"] == "resp_existing"
    assert (
        json.loads(str(a["gen_ai.output.messages"]))
        == response_payload(
            response_id="resp_existing",
            text="Streamed.",
        )["output"]
    )


async def test_async_responses_stream_bare_error_event_sets_error_status(
    memory: SimpleNamespace,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                {
                    "type": "error",
                    "code": "server_error",
                    "message": "model blew up",
                    "sequence_number": 0,
                }
            ]
        )

    client = wrap_openai(async_client(handler))
    stream = await cast(Any, client.responses.create)(
        model="gpt-4o-mini", input="fail", stream=True
    )
    events = [event async for event in stream]
    await client.close()

    assert [event.type for event in events] == ["error"]
    assert only_span(memory).status.status_code == StatusCode.ERROR


def test_chat_stream_context_manager_exit_ends_partial_span_once(
    memory: SimpleNamespace,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_openai(sync_client(handler))
    with cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    ) as stream:
        first = next(stream)

    spans = memory.span_exporter.get_finished_spans()
    assert first.choices[0].delta.content == "Hello"
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert json.loads(str(attrs(spans[0])["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]


async def test_async_chat_stream_context_manager_exit_ends_partial_span_once(
    memory: SimpleNamespace,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_openai(async_client(handler))
    async with await cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    ) as stream:
        first = await stream.__anext__()
    await client.close()

    spans = memory.span_exporter.get_finished_spans()
    assert first.choices[0].delta.content == "Hello"
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert json.loads(str(attrs(spans[0])["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]


def test_chat_stream_response_close_directly_ends_partial_span_once(
    memory: SimpleNamespace,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = next(stream)
    stream.response.close()

    spans = memory.span_exporter.get_finished_spans()
    assert first.choices[0].delta.content == "Hello"
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert json.loads(str(attrs(spans[0])["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]


def test_chat_stream_manager_context_exit_closes_response_and_ends_span_once(
    memory: SimpleNamespace,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(chat_stream_events())

    client = wrap_openai(sync_client(handler))
    with cast(Any, client.chat.completions.stream)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES
    ) as stream:
        first_event = next(stream)

    spans = memory.span_exporter.get_finished_spans()
    assert first_event.chunk.choices[0].delta.content == "Hello"
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert json.loads(str(attrs(spans[0])["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
