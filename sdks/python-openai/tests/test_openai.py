from __future__ import annotations

import importlib
import json
import time
import types
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import openai
import pytest
import telemetry_dev
from openai import AsyncOpenAI, OpenAI
from openai.resources.chat.completions.completions import AsyncCompletions, Completions
from openai.resources.embeddings import AsyncEmbeddings, Embeddings
from openai.resources.responses.responses import AsyncResponses, Responses
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

import telemetry_dev_openai
from telemetry_dev_openai import instrument_openai, uninstrument_openai, wrap_openai

if TYPE_CHECKING:
    import httpx2 as httpx
else:
    # openai 3.x is built on httpx2 and rejects httpx clients; 2.x uses httpx.
    _sdk_http = importlib.import_module("openai._base_client")
    httpx = getattr(_sdk_http, "httpx2", None) or _sdk_http.httpx

SyncHandler = Callable[[httpx.Request], httpx.Response]
AsyncHandler = Callable[[httpx.Request], Coroutine[Any, Any, httpx.Response]]


def assert_stream_transport_error(error: BaseException, message: str) -> None:
    # openai 3.x wraps a mid-stream transport failure in APIConnectionError; 2.x re-raises it.
    if openai.__version__.startswith("2."):
        transport = error
    else:
        assert type(error) is openai.APIConnectionError
        transport = error.__cause__
    assert type(transport) is httpx.ReadError
    assert str(transport) == message


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


@pytest.mark.parametrize("async_mode", [False, True])
async def test_response_mapping_failure_returns_provider_result_and_marks_capture(
    make: Any, async_mode: bool
) -> None:
    errors: list[BaseException] = []
    memory = make(on_error=errors.append)
    mapping_error = RuntimeError("mapping failed")
    provider_result = object()

    def request_mapper(_params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        return "chat gpt-4o", {"type": "generation", "model": "gpt-4o"}

    def response_mapper(_response: Any) -> dict[str, Any]:
        raise mapping_error

    def provider(_resource: object | None) -> str:
        return "openai"

    if async_mode:

        async def async_original(**_kwargs: Any) -> object:
            return provider_result

        wrapper = vars(telemetry_dev_openai)["_wrap_async"](
            async_original, "chat", request_mapper, response_mapper, provider, False
        )
        result = await wrapper(model="gpt-4o")
    else:

        def sync_original(**_kwargs: Any) -> object:
            return provider_result

        wrapper = vars(telemetry_dev_openai)["_wrap_sync"](
            sync_original, "chat", request_mapper, response_mapper, provider, False
        )
        result = wrapper(model="gpt-4o")

    assert result is provider_result
    assert errors == [mapping_error]
    assert attrs(only_span(memory))["telemetry.dev.capture.truncated"] is True


def small_chat_budget(max_bytes: int, reserve_output_list: bool = False) -> Any:
    budget = telemetry_dev.CaptureBudget(max_bytes, 1_000)
    if reserve_output_list:
        budget.accept([])
    return budget


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


class UnencodableText(str):
    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        raise AssertionError("capture must not encode the complete transcript")


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
            "prompt_tokens_details": {
                "cached_tokens": 6,
                "cached_tokens_details": {
                    "text_tokens": 1,
                    "image_tokens": 2,
                    "audio_tokens": 3,
                },
            },
            "completion_tokens_details": {
                "reasoning_tokens": 2,
                "text_tokens": 5,
                "audio_tokens": 2,
            },
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


def terminal_chat_stream_event() -> dict[str, Any]:
    event = chat_stream_events()[0]
    event["choices"][0]["finish_reason"] = "stop"
    return event


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
                    "delta": {"role": "assistant", "content": "not-retained"},
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
            "input_tokens_details": {
                "cached_tokens": 6,
                "cached_tokens_details": {
                    "text_tokens": 1,
                    "image_tokens": 2,
                    "audio_tokens": 3,
                },
            },
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


def test_image_generation_is_media_span_without_binary_capture(memory: SimpleNamespace) -> None:
    client = wrap_openai(
        sync_client(lambda _request: json_response({"created": 1, "data": [{"b64_json": "AAAA"}]}))
    )

    response = client.images.generate(model="gpt-image-1", prompt="A telemetry graph")

    assert response.data
    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "generate_content"
    assert a["gen_ai.output.type"] == "image"
    assert a["gen_ai.request.model"] == "gpt-image-1"
    assert "AAAA" not in json.dumps(a)


def test_responses_capture_recursively_omits_binary_media(memory: SimpleNamespace) -> None:
    responses_request = vars(telemetry_dev_openai)["_responses_request"]
    responses_response = vars(telemetry_dev_openai)["_responses_response"]
    _, request = responses_request(
        {
            "model": "gpt-4.1",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "INPUT_AUDIO", "format": "wav"},
                        },
                        {"type": "input_image", "image_url": "data:image/png;base64,INPUT_IMAGE"},
                        {"type": "input_image", "image_url": "https://example.com/image.png"},
                        {"type": "input_file", "file_data": "INPUT_FILE", "filename": "report.pdf"},
                    ],
                }
            ],
        }
    )
    response = responses_response(
        {
            "status": "completed",
            "output": [
                {"type": "output_audio", "data": "OUTPUT_AUDIO", "transcript": "hello"},
                {"type": "image_generation_call", "result": "IMAGE_RESULT", "status": "completed"},
            ],
        }
    )

    captured = json.dumps({"request": request, "response": response})
    assert "INPUT_AUDIO" not in captured
    assert "INPUT_IMAGE" not in captured
    assert "INPUT_FILE" not in captured
    assert "OUTPUT_AUDIO" not in captured
    assert "IMAGE_RESULT" not in captured
    assert "https://example.com/image.png" in captured
    assert "hello" in captured


def test_chat_capture_recursively_omits_binary_media(memory: SimpleNamespace) -> None:
    chat_request = vars(telemetry_dev_openai)["_chat_request"]
    chat_response = vars(telemetry_dev_openai)["_chat_response"]
    _, request = chat_request(
        {
            "model": "gpt-4o-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "INPUT_AUDIO", "format": "wav"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,INPUT_IMAGE"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        },
                    ],
                }
            ],
        }
    )
    response = chat_response(
        {
            "id": "chatcmpl_media",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "spoken reply",
                        "audio": {
                            "id": "audio_1",
                            "data": "OUTPUT_AUDIO",
                            "transcript": "spoken reply",
                        },
                    },
                }
            ],
        }
    )

    captured = json.dumps({"request": request, "response": response})
    assert "INPUT_AUDIO" not in captured
    assert "INPUT_IMAGE" not in captured
    assert "OUTPUT_AUDIO" not in captured
    assert "https://example.com/image.png" in captured
    assert "spoken reply" in captured


def test_disabled_responses_capture_does_not_convert_hostile_values(make: Any) -> None:
    class Hostile:
        def model_dump(self, **_kwargs: Any) -> Any:
            raise AssertionError("capture-disabled payload was traversed")

    make(capture_input=False, capture_output=False)
    responses_request = vars(telemetry_dev_openai)["_responses_request"]
    responses_response = vars(telemetry_dev_openai)["_responses_response"]

    _, request = responses_request({"model": "gpt-4.1", "input": Hostile()})
    response = responses_response({"status": "completed", "output": Hostile()})

    assert request["input"] is None
    assert response["output"] is None


def test_streamed_image_edit_maps_aggregate_output_tokens_to_image(memory: SimpleNamespace) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                {
                    "type": "image_edit.completed",
                    "usage": {"input_tokens": 4, "output_tokens": 9, "total_tokens": 13},
                }
            ]
        )

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.images.edit)(
        image=b"image",
        prompt="Add telemetry",
        model="gpt-image-1",
        stream=True,
    )

    assert [event.type for event in stream] == ["image_edit.completed"]
    span = only_span(memory)
    assert span.name == "image gpt-image-1"
    assert attrs(span)["gen_ai.usage.image.output_tokens"] == 9


@pytest.mark.parametrize("completed", [True, False])
def test_image_stream_is_flagged_only_without_a_terminal_event(
    memory: SimpleNamespace, completed: bool
) -> None:
    events: list[dict[str, Any]] = [
        {"type": "image_generation.partial_image", "b64_json": "QQ==", "partial_image_index": 0}
    ]
    if completed:
        events.append(
            {
                "type": "image_generation.completed",
                "b64_json": "QQ==",
                "usage": {"input_tokens": 4, "output_tokens": 9, "total_tokens": 13},
            }
        )
    client = wrap_openai(sync_client(lambda _request: sse_response(events)))

    stream = cast(Any, client.images.generate)(
        model="gpt-image-1", prompt="otter", stream=True, partial_images=1
    )

    assert len(list(stream)) == len(events)
    a = attrs(only_span(memory))
    if completed:
        assert "telemetry.dev.capture.truncated" not in a
    else:
        assert a["telemetry.dev.capture.truncated"] is True


def test_media_stream_preserves_sync_context_manager_cleanup(memory: SimpleNamespace) -> None:
    client = wrap_openai(
        sync_client(
            lambda _request: sse_response(
                [
                    {
                        "type": "image_edit.completed",
                        "usage": {"input_tokens": 4, "output_tokens": 9, "total_tokens": 13},
                    }
                ]
            )
        )
    )

    with cast(Any, client.images.edit)(
        image=b"image",
        prompt="Add telemetry",
        model="gpt-image-1",
        stream=True,
    ) as stream:
        assert next(stream).type == "image_edit.completed"

    assert len(memory.span_exporter.get_finished_spans()) == 1


async def test_media_stream_preserves_async_context_manager_cleanup(
    memory: SimpleNamespace,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return sse_response(
            [
                {
                    "type": "image_edit.completed",
                    "usage": {"input_tokens": 4, "output_tokens": 9, "total_tokens": 13},
                }
            ]
        )

    client = wrap_openai(async_client(handler))
    async with await cast(Any, client.images.edit)(
        image=b"image",
        prompt="Add telemetry",
        model="gpt-image-1",
        stream=True,
    ) as stream:
        assert (await stream.__anext__()).type == "image_edit.completed"
    await client.close()

    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_streaming_transcription_records_terminal_text_and_usage(memory: SimpleNamespace) -> None:
    events = [
        {"type": "transcript.text.delta", "delta": "partial"},
        {
            "type": "transcript.text.done",
            "text": "complete transcript",
            "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        },
    ]
    client = wrap_openai(sync_client(lambda _request: sse_response(events)))

    stream = cast(Any, client.audio.transcriptions.create)(
        model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
    )

    assert [event.type for event in stream] == [event["type"] for event in events]
    span = only_span(memory)
    assert span.name == "transcription gpt-4o-transcribe"
    a = attrs(span)
    assert a["gen_ai.output.messages"] == "complete transcript"
    assert a["gen_ai.usage.text.output_tokens"] == 3
    assert "telemetry.dev.capture.truncated" not in a


def test_unpaired_surrogate_does_not_replace_unary_result(memory: SimpleNamespace) -> None:
    text = "before\ud800after"
    client = wrap_openai(
        sync_client(
            lambda _request: httpx.Response(
                200,
                content=json.dumps({"text": text}),
                headers={"Content-Type": "application/json"},
            )
        )
    )

    response = cast(Any, client.audio.transcriptions.create)(
        model="gpt-4o-transcribe", file=("audio.wav", b"audio")
    )

    assert response.text == text
    assert attrs(only_span(memory))["gen_ai.output.messages"] == text


@pytest.mark.parametrize("async_mode", [False, True])
async def test_unpaired_surrogate_streams_preserve_events_and_cleanup(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    text = "before\ud800after"
    events = [
        {"type": "transcript.text.delta", "delta": text},
        {"type": "transcript.text.done", "text": text},
    ]

    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return sse_response(events)

        client = wrap_openai(async_client(handler))
        stream = await cast(Any, client.audio.transcriptions.create)(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )
        received = [event async for event in stream]
        assert stream.response.is_closed
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: sse_response(events)))
        stream = cast(Any, client.audio.transcriptions.create)(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )
        received = list(stream)
        assert stream.response.is_closed
        client.close()

    assert [event.type for event in received] == [event["type"] for event in events]
    assert received[-1].text == text
    assert attrs(only_span(memory))["gen_ai.output.messages"] == text


def test_streaming_transcription_terminal_text_replaces_truncated_deltas(
    memory: SimpleNamespace,
) -> None:
    events = [
        {"type": "transcript.text.delta", "delta": "x" * 70_000},
        {"type": "transcript.text.done", "text": "complete transcript"},
    ]
    client = wrap_openai(sync_client(lambda _request: sse_response(events)))

    stream = cast(Any, client.audio.transcriptions.create)(
        model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
    )
    assert [event.type for event in stream] == [event["type"] for event in events]

    a = attrs(only_span(memory))
    assert a["gen_ai.output.messages"] == "complete transcript"
    assert "telemetry.dev.capture.truncated" not in a


async def test_async_streaming_transcription_records_terminal_text_and_usage(
    memory: SimpleNamespace,
) -> None:
    events = [
        {
            "type": "transcript.text.segment",
            "id": "segment-1",
            "start": 0,
            "end": 1,
            "speaker": "A",
            "text": "partial",
        },
        {
            "type": "transcript.text.done",
            "text": "complete transcript",
            "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        },
    ]

    async def handler(_request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    client = wrap_openai(async_client(handler))
    stream = await cast(Any, client.audio.transcriptions.create)(
        model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
    )
    assert [event.type async for event in stream] == [event["type"] for event in events]
    await client.close()

    a = attrs(only_span(memory))
    assert a["gen_ai.output.messages"] == "complete transcript"
    assert a["gen_ai.usage.text.output_tokens"] == 3


@pytest.mark.parametrize("async_mode", [False, True])
async def test_interrupted_transcription_stream_records_bounded_partial_text(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    event = {"type": "transcript.text.delta", "delta": "x" * 70_000}
    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return failing_async_sse_response(event, "interrupted")

        client = wrap_openai(async_client(handler))
        stream = await cast(Any, client.audio.transcriptions.create)(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )
        with pytest.raises(Exception) as exc_info:
            _ = [item async for item in stream]
        assert_stream_transport_error(exc_info.value, "interrupted")
        await client.close()
    else:
        client = wrap_openai(
            sync_client(lambda _request: failing_sync_sse_response(event, "interrupted"))
        )
        stream = cast(Any, client.audio.transcriptions.create)(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )
        with pytest.raises(Exception) as exc_info:
            list(stream)
        assert_stream_transport_error(exc_info.value, "interrupted")

    span = only_span(memory)
    output = str(attrs(span)["gen_ai.output.messages"])
    assert output == "x" * 65_536
    assert attrs(span)["telemetry.dev.capture.truncated"] is True


def test_non_streaming_transcription_over_the_cap_reaches_the_mask(make: Any) -> None:
    seen: list[Any] = []

    def mask(value: Any, _context: Any) -> Any:
        seen.append(value)
        return value

    memory = make(max_attribute_length=1_000, mask=mask)
    transcript = "x" * 1_500
    client = wrap_openai(sync_client(lambda _request: json_response({"text": transcript})))

    cast(Any, client.audio.transcriptions.create)(
        model="gpt-4o-transcribe", file=("audio.wav", b"audio")
    )

    a = attrs(only_span(memory))
    assert seen == [transcript]
    assert len(str(a["gen_ai.output.messages"])) == 1_000
    assert str(a["gen_ai.output.messages"]).endswith("...[truncated]")
    assert "telemetry.dev.capture.truncated" not in a


@pytest.mark.parametrize(
    ("max_attribute_length", "redact", "expected"),
    [(1_000, False, "東" * 400), (1_000, True, "[redacted]"), (0, False, None)],
)
def test_chat_stream_configured_cap_applies_after_the_mask(
    make: Any, max_attribute_length: int, redact: bool, expected: str | None
) -> None:
    def mask(value: Any, _context: Any) -> Any:
        return [{"role": "assistant", "content": "[redacted]"}] if redact else value

    memory = make(max_attribute_length=max_attribute_length, mask=mask)
    content = "x" * 5_000 if redact else "東" * 400
    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl_cap",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
        }
    ]
    client = wrap_openai(sync_client(lambda _request: sse_response(events)))

    list(
        cast(Any, client.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
    )

    a = attrs(only_span(memory))
    assert "telemetry.dev.capture.truncated" not in a
    if expected is None:
        assert a.get("gen_ai.output.messages", "") == ""
    else:
        assert json.loads(str(a["gen_ai.output.messages"]))[0]["content"] == expected


def test_many_small_transcript_deltas_are_not_item_capped(memory: SimpleNamespace) -> None:
    events = [{"type": "transcript.text.delta", "delta": "x"} for _ in range(2_000)]
    client = wrap_openai(sync_client(lambda _request: sse_response(events)))

    list(
        cast(Any, client.audio.transcriptions.create)(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )
    )

    assert attrs(only_span(memory))["gen_ai.output.messages"] == "x" * 2_000


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_complete_transcript_above_the_chat_stream_budget_is_kept(
    make: Any, streaming: bool, masked: bool
) -> None:
    def identity_mask(value: Any, _context: Any) -> Any:
        return value

    memory = make(mask=identity_mask) if masked else make()
    transcript = "x" * 55_000

    def handler(_request: httpx.Request) -> httpx.Response:
        if streaming:
            return sse_response(
                [
                    {"type": "transcript.text.delta", "delta": transcript},
                    {"type": "transcript.text.done", "text": transcript},
                ]
            )
        return json_response({"text": transcript})

    client = wrap_openai(sync_client(handler))
    result = cast(Any, client.audio.transcriptions.create)(
        model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=streaming
    )
    if streaming:
        list(result)

    a = attrs(only_span(memory))
    assert a["gen_ai.output.messages"] == transcript
    assert "telemetry.dev.capture.truncated" not in a


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", ["responses", "transcription"])
@pytest.mark.parametrize("masked", [False, True])
async def test_complete_streams_are_not_flagged_incomplete(
    make: Any, async_mode: bool, kind: str, masked: bool
) -> None:
    def identity_mask(value: Any, _context: Any) -> Any:
        return value

    memory = make(mask=identity_mask) if masked else make()
    completed = response_payload(response_id="resp_done", text="done")
    if kind == "responses":
        in_progress: dict[str, Any] = {**completed, "status": "in_progress", "output": []}
        events: list[dict[str, Any]] = [
            {"type": "response.created", "response": in_progress},
            {"type": "response.completed", "response": completed},
        ]
    else:
        events = [
            {"type": "transcript.text.delta", "delta": "partial"},
            {"type": "transcript.text.done", "text": "complete transcript"},
        ]

    async def create_async(client: Any) -> Any:
        if kind == "responses":
            return await client.responses.create(model="gpt-4.1", input="x", stream=True)
        return await client.audio.transcriptions.create(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )

    def create_sync(client: Any) -> Any:
        if kind == "responses":
            return client.responses.create(model="gpt-4.1", input="x", stream=True)
        return client.audio.transcriptions.create(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )

    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return sse_response(events)

        client = wrap_openai(async_client(handler))
        delivered = [item async for item in await create_async(client)]
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: sse_response(events)))
        delivered = list(create_sync(client))

    assert len(delivered) == len(events)
    a = attrs(only_span(memory))
    assert "telemetry.dev.capture.truncated" not in a
    if kind == "responses":
        assert json.loads(str(a["gen_ai.output.messages"])) == completed["output"]
    else:
        assert a["gen_ai.output.messages"] == "complete transcript"


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("ending", ["error", "close", "eof"])
@pytest.mark.parametrize("masked", [False, True])
async def test_short_transcription_stream_without_terminal_event_is_incomplete(
    make: Any, async_mode: bool, ending: str, masked: bool
) -> None:
    def identity_mask(value: Any, _context: Any) -> Any:
        return value

    memory = make(mask=identity_mask) if masked else make()
    event = {"type": "transcript.text.delta", "delta": "partial"}

    def response() -> httpx.Response:
        if ending == "error":
            return (
                failing_async_sse_response(event, "interrupted")
                if async_mode
                else failing_sync_sse_response(event, "interrupted")
            )
        return sse_response([event])

    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return response()

        client = wrap_openai(async_client(handler))
        stream = await cast(Any, client.audio.transcriptions.create)(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )
        assert (await stream.__anext__()).type == "transcript.text.delta"
        if ending == "error":
            with pytest.raises(Exception) as exc_info:
                await stream.__anext__()
            assert_stream_transport_error(exc_info.value, "interrupted")
        elif ending == "close":
            await stream.close()
        else:
            assert [item async for item in stream] == []
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: response()))
        stream = cast(Any, client.audio.transcriptions.create)(
            model="gpt-4o-transcribe", file=("audio.wav", b"audio"), stream=True
        )
        assert next(stream).type == "transcript.text.delta"
        if ending == "error":
            with pytest.raises(Exception) as exc_info:
                next(stream)
            assert_stream_transport_error(exc_info.value, "interrupted")
        elif ending == "close":
            stream.close()
        else:
            assert list(stream) == []

    a = attrs(only_span(memory))
    assert a.get("gen_ai.output.messages") == (None if masked else "partial")
    assert a["telemetry.dev.capture.truncated"] is True


def test_terminal_transcription_capture_is_incrementally_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_dev_openai, "_TRANSCRIPT_CAPTURE_MAX_BYTES", 5)
    mapper = vars(telemetry_dev_openai)["_media_stream_event_fields"]

    fields = mapper(
        SimpleNamespace(type="transcript.text.done", text=UnencodableText("ééé")),
        vars(telemetry_dev_openai)["_text_media_response"],
        True,
    )

    assert fields["output"] == "éé"
    assert fields["attributes"]["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("capture_output", [False, True])
def test_text_media_response_uses_the_policy_that_created_its_span(
    monkeypatch: pytest.MonkeyPatch, capture_output: bool
) -> None:
    ended: list[dict[str, Any]] = []
    reported: list[BaseException] = []

    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return value

    def end(**fields: Any) -> None:
        ended.append(fields)

    handle = SimpleNamespace(
        _client=SimpleNamespace(capture_output=capture_output, mask=mask),
        _state=SimpleNamespace(capture_output=capture_output),
        report_error=reported.append,
    )
    monkeypatch.setattr(
        telemetry_dev,
        "get_client",
        lambda: SimpleNamespace(capture_output=not capture_output, mask=None),
    )
    implementation = cast(Any, telemetry_dev_openai)

    implementation._end_mapped_response(
        handle,
        end,
        implementation._text_media_response,
        SimpleNamespace(text="sensitive transcript"),
    )

    assert reported == []
    assert ended[-1].get("output") == ("sensitive transcript" if capture_output else None)
    assert "attributes" not in ended[-1]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_transcription_stream_uses_the_policy_that_created_its_span(
    monkeypatch: pytest.MonkeyPatch, async_mode: bool
) -> None:
    ended: list[dict[str, Any]] = []
    reported: list[BaseException] = []
    event = SimpleNamespace(type="transcript.text.done", text="sensitive transcript")

    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return value

    def end(**fields: Any) -> None:
        ended.append(fields)

    handle = SimpleNamespace(
        _client=SimpleNamespace(capture_output=True, mask=mask),
        _state=SimpleNamespace(capture_output=True),
        end=end,
        report_error=reported.append,
    )
    monkeypatch.setattr(
        telemetry_dev,
        "get_client",
        lambda: SimpleNamespace(capture_output=False, mask=None),
    )
    implementation = cast(Any, telemetry_dev_openai)

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            yield event

        stream = implementation._InstrumentedAsyncMediaStream(
            source(), handle, implementation._text_media_response
        )
        assert await stream.__anext__() is event
    else:
        stream = implementation._InstrumentedMediaStream(
            iter([event]), handle, implementation._text_media_response
        )
        assert next(stream) is event

    assert reported == []
    assert ended[-1]["output"] == "sensitive transcript"
    assert "attributes" not in ended[-1]


@pytest.mark.parametrize("async_mode", [False, True])
async def test_transcription_delta_capture_is_incrementally_bounded(
    make: Any, monkeypatch: pytest.MonkeyPatch, async_mode: bool
) -> None:
    make()
    monkeypatch.setattr(telemetry_dev_openai, "_TRANSCRIPT_CAPTURE_MAX_BYTES", 5)
    ended: list[dict[str, Any]] = []
    event = SimpleNamespace(type="transcript.text.delta", delta=UnencodableText("ééé"))

    class Handle:
        capture_output = True
        capture_masked = False

        def end(self, **fields: Any) -> None:
            ended.append(fields)

    if async_mode:

        class Inner:
            def __init__(self) -> None:
                self._remaining = True

            def __aiter__(self) -> Inner:
                return self

            async def __anext__(self) -> Any:
                if not self._remaining:
                    raise StopAsyncIteration
                self._remaining = False
                return event

        stream_type = vars(telemetry_dev_openai)["_InstrumentedAsyncMediaStream"]
        stream = stream_type(
            Inner(), cast(Any, Handle()), vars(telemetry_dev_openai)["_text_media_response"]
        )
        assert await stream.__anext__() is event
        await stream.close()
    else:
        stream_type = vars(telemetry_dev_openai)["_InstrumentedMediaStream"]
        stream = stream_type(
            iter([event]), cast(Any, Handle()), vars(telemetry_dev_openai)["_text_media_response"]
        )
        assert next(stream) is event
        stream.close()

    assert ended[-1]["output"] == "éé"
    assert ended[-1]["attributes"]["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
async def test_failed_media_stream_events_record_errors(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    event = SimpleNamespace(
        type="image_generation.failed",
        error=SimpleNamespace(code="provider_error", message="provider failed"),
    )
    handle = telemetry_dev.start_span("image failed", type="generation")

    if async_mode:

        class Inner:
            def __init__(self) -> None:
                self._remaining = True

            def __aiter__(self) -> Inner:
                return self

            async def __anext__(self) -> Any:
                if not self._remaining:
                    raise StopAsyncIteration
                self._remaining = False
                return event

        stream_type = vars(telemetry_dev_openai)["_InstrumentedAsyncMediaStream"]
        stream = stream_type(Inner(), handle, vars(telemetry_dev_openai)["_media_response"])
        assert await stream.__anext__() is event
    else:
        stream_type = vars(telemetry_dev_openai)["_InstrumentedMediaStream"]
        stream = stream_type(iter([event]), handle, vars(telemetry_dev_openai)["_media_response"])
        assert next(stream) is event

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "RuntimeError"
    assert dict(span.events[0].attributes or {})["exception.message"] == (
        "provider_error: provider failed"
    )


def test_disabled_transcription_stream_capture_does_not_read_text(make: Any) -> None:
    class Event:
        type = "transcript.text.done"
        usage = SimpleNamespace(output_tokens=2)

        @property
        def text(self) -> str:
            raise AssertionError("capture-disabled output was read")

    class Handle:
        capture_output = False
        capture_masked = False
        max_attribute_length = 5

        def end(self, **_fields: Any) -> None:
            pass

    make(capture_output=False)
    stream_type = vars(telemetry_dev_openai)["_InstrumentedMediaStream"]
    stream = stream_type(
        iter([Event()]), cast(Any, Handle()), vars(telemetry_dev_openai)["_text_media_response"]
    )

    assert next(stream).type == "transcript.text.done"


@pytest.mark.parametrize("delegated", [False, True])
def test_media_stream_context_exit_records_body_and_delegated_errors(delegated: bool) -> None:
    errors: list[BaseException] = []

    class Handle:
        def end(self, **fields: Any) -> None:
            errors.append(fields["error"])

    class Inner:
        def __exit__(self, _type: Any, _exc: Any, _tb: Any) -> bool:
            if delegated:
                raise RuntimeError("exit failed")
            return False

    stream_type = vars(telemetry_dev_openai)["_InstrumentedMediaStream"]
    stream = stream_type(
        Inner(), cast(Any, Handle()), vars(telemetry_dev_openai)["_media_response"]
    )
    body_error = ValueError("body failed")
    if delegated:
        with pytest.raises(RuntimeError, match="exit failed"):
            stream.__exit__(None, None, None)
        assert str(errors[0]) == "exit failed"
    else:
        assert stream.__exit__(ValueError, body_error, None) is False
        assert errors == [body_error]


@pytest.mark.parametrize("delegated", [False, True])
async def test_async_media_stream_context_exit_records_body_and_delegated_errors(
    delegated: bool,
) -> None:
    errors: list[BaseException] = []

    class Handle:
        def end(self, **fields: Any) -> None:
            errors.append(fields["error"])

    class Inner:
        async def __aexit__(self, _type: Any, _exc: Any, _tb: Any) -> bool:
            if delegated:
                raise RuntimeError("exit failed")
            return False

    stream_type = vars(telemetry_dev_openai)["_InstrumentedAsyncMediaStream"]
    stream = stream_type(
        Inner(), cast(Any, Handle()), vars(telemetry_dev_openai)["_media_response"]
    )
    body_error = ValueError("body failed")
    if delegated:
        with pytest.raises(RuntimeError, match="exit failed"):
            await stream.__aexit__(None, None, None)
        assert str(errors[0]) == "exit failed"
    else:
        assert await stream.__aexit__(ValueError, body_error, None) is False
        assert errors == [body_error]


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
    assert a["gen_ai.usage.cache_read.input_tokens"] == 6
    assert a["gen_ai.usage.reasoning.output_tokens"] == 2
    assert a["gen_ai.usage.text.cache_read.input_tokens"] == 1
    assert a["gen_ai.usage.image.cache_read.input_tokens"] == 2
    assert a["gen_ai.usage.audio.cache_read.input_tokens"] == 3
    assert a["gen_ai.usage.text.output_tokens"] == 5
    assert a["gen_ai.usage.audio.output_tokens"] == 2
    assert json.loads(str(a["gen_ai.input.messages"])) == CHAT_MESSAGES
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Telemetry works."}
    ]


def test_chat_completion_records_the_body_values_the_sdk_sends(memory: SimpleNamespace) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request_json(request))
        return json_response(chat_completion())

    client = wrap_openai(sync_client(handler))
    cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        temperature=openai.omit,
        extra_body={
            "model": openai.NOT_GIVEN,
            "temperature": 0.2,
            "top_p": openai.omit,
            "max_completion_tokens": 32,
        },
    )

    a = attrs(only_span(memory))
    assert requests[0]["model"] == a["gen_ai.request.model"] == "gpt-4o-mini"
    assert requests[0]["temperature"] == a["gen_ai.request.temperature"] == 0.2
    assert requests[0]["max_completion_tokens"] == a["gen_ai.request.max_tokens"] == 32
    assert "top_p" not in requests[0]
    assert "gen_ai.request.top_p" not in a


@pytest.mark.parametrize(
    ("base_url", "expected_provider"),
    [
        ("https://openrouter.ai/api/v1", "openrouter"),
        ("https://api.openrouter.ai/api/v1", "openrouter"),
        ("https://openrouter.ai./api/v1", "openrouter"),
        ("https://api.openrouter.ai./api/v1", "openrouter"),
        ("https://openrouter.ai.example.com/api/v1", "openai"),
        ("https://api.groq.com/openai/v1", "groq"),
        ("https://api.x.ai/v1", "x_ai"),
        ("https://api.deepseek.com/v1", "deepseek"),
        ("https://api.together.xyz/v1", "together_ai"),
        ("https://api.fireworks.ai/inference/v1", "fireworks_ai"),
        ("https://groq.com.example.org/v1", "openai"),
        ("https://notx.ai.example.org/v1", "openai"),
        ("https://deepseek.com.evil.test/v1", "openai"),
        ("https://together.xyz.invalid/v1", "openai"),
        ("https://fireworks.ai.example.com/v1", "openai"),
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


def test_chat_stream_reads_fields_the_sdk_keeps_as_pydantic_extras(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk_calls: list[float | None] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        chunk_calls.append(timestamp_ms)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    base = {
        "id": "chatcmpl_extra",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
    }
    events: list[dict[str, Any]] = [
        {
            **base,
            "choices": [{"index": 0, "delta": {"audio": {"id": "audio_1", "data": "QQ=="}}}],
        },
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_custom",
                                "type": "custom",
                                "custom": {"name": "lookup", "input": "order 42"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    client = wrap_openai(sync_client(lambda _request: sse_response(events)))

    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o", messages=CHAT_MESSAGES, stream=True
    )

    assert len(list(stream)) == len(events)
    assert len(chunk_calls) == 2
    a = attrs(only_span(memory))
    output = str(a["gen_ai.output.messages"])
    assert "lookup" in output
    assert "order 42" in output
    assert "telemetry.dev.capture.truncated" not in a


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


def test_field_reads_never_run_storage_properties() -> None:
    calls: list[str] = []

    class HostileDelta:
        @property
        def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
            calls.append("__dict__")
            return {"custom": {"input": "x" * 1_000}}

        @property
        def __pydantic_extra__(self) -> dict[str, Any]:
            calls.append("__pydantic_extra__")
            return {"custom": {"input": "x" * 1_000}}

    own_field = vars(telemetry_dev_openai)["_own_field"]
    budget = telemetry_dev.CaptureBudget(max_bytes=0)

    assert own_field(HostileDelta(), "custom", budget) is None
    for hostile in _spoofed_and_aliased_storage(calls):
        assert own_field(hostile, "custom", budget) is None
    assert calls == []


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chunk_timing_excludes_control_and_empty_deltas(
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
    deltas: list[dict[str, Any]] = [
        {"role": "assistant"},
        {"content": "A"},
        {"content": None},
        {"content": ""},
        {"tool_calls": []},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
        {"function_call": {"name": "lookup", "arguments": ""}},
        {"function_call": {"name": "lookup"}},
        {"function_call": {"arguments": '{"city":'}},
        {"content": "once", "function_call": {"arguments": '"Paris"}'}},
        {"content": "B"},
        {},
    ]
    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl_timing",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": delta}],
        }
        for delta in deltas
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    if async_mode:
        async_wrapped = wrap_openai(async_client(async_handler))
        async_stream = await cast(Any, async_wrapped.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
        received = [chunk async for chunk in async_stream]
        await async_wrapped.close()
    else:
        wrapped = wrap_openai(sync_client(handler))
        stream = cast(Any, wrapped.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
        received = list(stream)
        wrapped.close()
    assert len(received) == len(events)
    assert len(calls) == 6
    assert memory.metric_reader.get_metrics_data() is not None


def test_chat_streaming_reconstructs_legacy_function_call() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"function_call": {"name": "lookup", "arguments": '{"city":'}},
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"function_call": {"arguments": '"Paris"}'}},
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    for event in events:
        implementation._record_chat_chunk(
            event,
            states,
            {},
            {},
            set(),
            output_budget,
            finish_reason_budget,
            True,
        )

    assert implementation._chat_output(states) == [
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "lookup", "arguments": '{"city":"Paris"}'},
        }
    ]
    assert output_budget.truncated is False


@pytest.mark.parametrize(
    ("event_type", "chunks"),
    [("response.shell_call_command.delta", 2), ("response.shell_call_output_content.delta", 0)],
)
def test_responses_chunk_timing_counts_model_generated_shell_commands(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, event_type: str, chunks: int
) -> None:
    calls: list[telemetry_dev.SpanHandle] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        calls.append(handle)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    delta = {"item_id": "shell_1", "output_index": 0, "sequence_number": 0}
    events: list[dict[str, Any]] = [
        {"type": event_type, "delta": "ls", **delta},
        {"type": event_type, "delta": "", **delta},
        {"type": event_type, "delta": " -la", **delta},
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {"id": "resp_shell", "model": "gpt-5", "status": "completed", "output": []},
        },
    ]

    stream = cast(Any, wrap_openai(sync_client(lambda _request: sse_response(events))).responses)
    received = list(stream.create(model="gpt-5", input="List files", stream=True))

    assert len(received) == len(events)
    assert len(calls) == chunks


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chunk_receipt_time_precedes_mapping(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, async_mode: bool
) -> None:
    clock = 12.0
    recorded: list[float] = []
    original_record = telemetry_dev.SpanHandle.record_output_chunk
    original_record_chunk = vars(telemetry_dev_openai)["_record_chat_chunk"]

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float
    ) -> telemetry_dev.SpanHandle:
        recorded.append(timestamp_ms)
        return original_record(handle, timestamp_ms)

    def perf_counter() -> float:
        return clock

    def delayed_record_chunk(*args: Any, **kwargs: Any) -> Any:
        nonlocal clock
        result = original_record_chunk(*args, **kwargs)
        clock = 47.0
        return result

    monkeypatch.setattr(time, "perf_counter", perf_counter)
    monkeypatch.setattr(telemetry_dev_openai, "_record_chat_chunk", delayed_record_chunk)
    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    events = [
        {
            "id": "chatcmpl_receipt",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": text}}],
        }
        for text in ["hello", " world"]
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    if async_mode:
        async_wrapped = wrap_openai(async_client(async_handler))
        async_stream = await async_wrapped.chat.completions.create(
            model="gpt-4o", messages=[], stream=True
        )
        received = [chunk async for chunk in async_stream]
        await async_wrapped.close()
    else:
        wrapped = wrap_openai(sync_client(handler))
        stream = wrapped.chat.completions.create(model="gpt-4o", messages=[], stream=True)
        received = list(stream)
        wrapped.close()
    assert len(received) == 2
    assert recorded == [12_000.0, 47_000.0]
    assert only_span(memory).status.status_code == StatusCode.UNSET


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize(
    "event_type",
    [
        "response.custom_tool_call_input",
        "response.code_interpreter_call_code",
        "response.mcp_call_arguments",
        "response.audio.transcript",
    ],
)
async def test_responses_output_timing(
    memory: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
    interrupted: bool,
    event_type: str,
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
    output_events = [
        {"type": f"{event_type}.delta", "delta": "first"},
        {"type": f"{event_type}.delta", "delta": ""},
        {"type": f"{event_type}.delta", "delta": "second"},
        {
            "type": f"{event_type}.done",
            "input": "firstsecond",
            "code": "firstsecond",
            "arguments": "firstsecond",
        },
    ]
    events = [
        *output_events,
        {"type": "response.completed", "response": response_payload()},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    if async_mode:
        client = wrap_openai(async_client(async_handler))
        stream = await cast(Any, client.responses.create)(
            model="gpt-4o-mini", input="Run", stream=True
        )
        received = [await stream.__anext__() for _ in output_events]
        if not interrupted:
            received.extend([event async for event in stream])
        await stream.close()
        await client.close()
    else:
        client = wrap_openai(sync_client(handler))
        stream = cast(Any, client.responses.create)(model="gpt-4o-mini", input="Run", stream=True)
        received = [next(stream) for _ in output_events]
        if not interrupted:
            received.extend(stream)
        stream.close()
        client.close()

    assert len(received) == len(output_events if interrupted else events)
    assert received[0].delta == "first"
    assert received[2].delta == "second"
    assert len(calls) == 2
    assert calls[0] is calls[1]
    assert only_span(memory).status.status_code == StatusCode.UNSET


@pytest.mark.parametrize(
    "stream_kind", ["sync-chat", "async-chat", "sync-responses", "async-responses"]
)
async def test_streaming_preserves_tracing_without_output_chunk_support(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, stream_kind: str
) -> None:
    monkeypatch.delattr(telemetry_dev.SpanHandle, "record_output_chunk")

    if stream_kind == "sync-chat":

        def sync_chat_handler(request: httpx.Request) -> httpx.Response:
            return sse_response(chat_stream_events())

        client = wrap_openai(sync_client(sync_chat_handler))
        received = list(
            cast(Any, client.chat.completions.create)(
                model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
            )
        )
        client.close()
    elif stream_kind == "async-chat":

        async def async_chat_handler(request: httpx.Request) -> httpx.Response:
            return sse_response(chat_stream_events())

        client = wrap_openai(async_client(async_chat_handler))
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        received = [chunk async for chunk in stream]
        await client.close()
    else:
        delta = {
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "item_id": "item_old_core",
            "output_index": 0,
            "content_index": 0,
            "delta": "Still traced.",
        }
        response = {
            "type": "response.completed",
            "sequence_number": 1,
            "response": response_payload(response_id="resp_old_core", text="Still traced."),
        }
        events = [("response.output_text.delta", delta), ("response.completed", response)]
        if stream_kind == "sync-responses":

            def sync_responses_handler(request: httpx.Request) -> httpx.Response:
                return named_sse_response(events)

            client = wrap_openai(sync_client(sync_responses_handler))
            received = list(
                cast(Any, client.responses.create)(model="gpt-4o-mini", input="stream", stream=True)
            )
            client.close()
        else:

            async def async_responses_handler(request: httpx.Request) -> httpx.Response:
                return named_sse_response(events)

            client = wrap_openai(async_client(async_responses_handler))
            stream = await cast(Any, client.responses.create)(
                model="gpt-4o-mini", input="stream", stream=True
            )
            received = [event async for event in stream]
            await client.close()
        assert len(received) == 2
        assert received[0].delta == "Still traced."

    assert received
    span = only_span(memory)
    assert span.status.status_code == StatusCode.UNSET
    assert attrs(span)["gen_ai.response.id"] in {"chatcmpl_stream", "resp_old_core"}


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
    assert state.role is None
    assert not state.content.endswith("not-retained")
    a = attrs(only_span(memory))
    assert len(str(a["gen_ai.output.messages"]).encode()) <= 48 * 1024
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["gen_ai.usage.total_tokens"] == 7
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("max_attribute_length", [0, 256])
async def test_chat_stream_finish_reasons_do_not_depend_on_content_limit(
    make: Any, async_mode: bool, max_attribute_length: int
) -> None:
    memory = make(max_attribute_length=max_attribute_length)
    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl_limit",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
        }
    ]

    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return sse_response(events)

        client = wrap_openai(async_client(handler))
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
        delivered = [item async for item in stream]
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: sse_response(events)))
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
        delivered = list(stream)

    assert len(delivered) == 1
    a = attrs(only_span(memory))
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    if max_attribute_length == 0:
        assert a.get("gen_ai.output.messages", "") == ""
    else:
        assert json.loads(str(a["gen_ai.output.messages"])) == [
            {"role": "assistant", "content": "Hi"}
        ]
        assert "telemetry.dev.capture.truncated" not in a


@pytest.mark.parametrize("async_mode", [False, True])
async def test_multi_choice_finish_reasons_survive_a_zero_content_limit(
    make: Any, async_mode: bool
) -> None:
    memory = make(max_attribute_length=0, capture_input=False, capture_output=False)
    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl_two",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
                {"index": 1, "delta": {}, "finish_reason": "length"},
            ],
        }
    ]

    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return sse_response(events)

        client = wrap_openai(async_client(handler))
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
        delivered = [item async for item in stream]
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: sse_response(events)))
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
        delivered = list(stream)

    assert len(delivered) == 1
    a = attrs(only_span(memory))
    assert a["gen_ai.response.finish_reasons"] == ("stop", "length")
    assert "telemetry.dev.capture.truncated" not in a


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("masked", [False, True])
async def test_post_finish_content_filter_chunk_keeps_chat_stream_complete(
    make: Any, async_mode: bool, masked: bool
) -> None:
    def identity_mask(value: Any, _context: Any) -> Any:
        return value

    memory = make(mask=identity_mask) if masked else make()
    base = {"id": "chatcmpl_azure", "object": "chat.completion.chunk", "created": 1}
    events: list[dict[str, Any]] = [
        {
            **base,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}
            ],
        },
        {
            **base,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            **base,
            "model": "",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None,
                    "content_filter_offsets": {
                        "check_offset": 0,
                        "start_offset": 0,
                        "end_offset": 2,
                    },
                    "content_filter_results": {"hate": {"filtered": False, "severity": "safe"}},
                }
            ],
        },
    ]

    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return sse_response(events)

        client = wrap_openai(async_client(handler))
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
        delivered = [item async for item in stream]
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: sse_response(events)))
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o", messages=CHAT_MESSAGES, stream=True
        )
        delivered = list(stream)

    assert len(delivered) == len(events)
    a = attrs(only_span(memory))
    assert "telemetry.dev.capture.truncated" not in a
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert json.loads(str(a["gen_ai.output.messages"])) == [{"role": "assistant", "content": "Hi"}]


def test_chat_streaming_omits_bounded_prefix_when_mask_cannot_inspect_complete_output(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return "[REDACTED]" if "SECRET" in json.dumps(value) else value

    monkeypatch.setattr(telemetry_dev_openai, "_CHAT_STREAM_CAPTURE_MAX_BYTES", 128)
    memory = make(mask=mask)
    base = {
        "id": "chatcmpl_masked_stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    events: list[dict[str, Any]] = [
        {**base, "choices": [{"index": 0, "delta": {"content": "x"}}]} for _ in range(200)
    ]
    events.append(
        {
            **base,
            "choices": [{"index": 0, "delta": {"content": "SECRET"}, "finish_reason": "stop"}],
        }
    )

    stream = cast(
        Any,
        wrap_openai(sync_client(lambda _request: sse_response(events))).chat.completions.create,
    )(model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True)

    assert len(list(stream)) == len(events)
    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def test_chat_streaming_omits_masked_output_without_terminal_finish_reason(make: Any) -> None:
    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return value

    memory = make(mask=mask)
    event = {
        "id": "chatcmpl_incomplete",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"content": "partial"}}],
    }

    stream = cast(
        Any,
        wrap_openai(sync_client(lambda _request: sse_response([event]))).chat.completions.create,
    )(model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True)

    assert len(list(stream)) == 1
    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chat_streaming_bounds_choice_and_finish_reason_state(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    event: dict[str, Any] = {
        "id": "chatcmpl_many_choices",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": index, "delta": {}, "finish_reason": "stop"} for index in range(400)],
    }

    if async_mode:

        async def async_handler(_request: httpx.Request) -> httpx.Response:
            return sse_response([event])

        client = wrap_openai(async_client(async_handler))
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        delivered = [chunk async for chunk in stream]
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: sse_response([event])))
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        delivered = list(stream)
        client.close()

    assert len(delivered) == 1
    assert len(stream._states) == 333
    a = attrs(only_span(memory))
    output = json.loads(str(a["gen_ai.output.messages"]))
    finish_reasons = cast(Sequence[str], a["gen_ai.response.finish_reasons"])
    assert len(output) == 333
    assert len(finish_reasons) == 200
    assert len(str(a["gen_ai.output.messages"]).encode()) <= 48 * 1024
    assert a["telemetry.dev.capture.truncated"] is True


def test_chat_streaming_reclaims_finish_reason_items_on_replacement(make: Any) -> None:
    memory = make(capture_output=False)
    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl_finish_reasons",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": index, "delta": {}, "finish_reason": "stop"} for index in range(200)
            ],
        },
        {
            "id": "chatcmpl_finish_reasons",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
        },
    ]
    client = wrap_openai(sync_client(lambda _request: sse_response(events)))

    list(
        cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
    )

    a = attrs(only_span(memory))
    finish_reasons = cast(Sequence[str], a["gen_ai.response.finish_reasons"])
    assert len(finish_reasons) == 200
    assert finish_reasons[0] == "length"
    assert "gen_ai.output.messages" not in a
    assert "telemetry.dev.capture.truncated" not in a


def test_chat_streaming_retains_exact_multibyte_finish_reason_budget(make: Any) -> None:
    memory = make(capture_output=False)
    reason = ('"🙂\\\n' * 4_905) + "🙂"
    event: dict[str, Any] = {
        "id": "chatcmpl_finish_reason_budget",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    }
    client = wrap_openai(sync_client(lambda _request: sse_response([event])))

    list(
        cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
    )

    a = attrs(only_span(memory))
    finish_reasons = list(cast(Sequence[str], a["gen_ai.response.finish_reasons"]))
    assert len(reason) == 19_621
    assert finish_reasons == [reason]
    serialized = json.dumps(finish_reasons, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(serialized) == 49_058
    assert "telemetry.dev.capture.truncated" not in a


def test_chat_streaming_clears_recovered_finish_reason_truncation(memory: SimpleNamespace) -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    finish_reason_reservations: dict[int, tuple[int, int]] = {}
    rejected_finish_reasons: set[int] = set()
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()

    implementation._record_chat_chunk(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "x" * 70_000}]},
        states,
        finish_reason_states,
        finish_reason_reservations,
        rejected_finish_reasons,
        output_budget,
        finish_reason_budget,
        False,
    )
    implementation._record_chat_chunk(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        states,
        finish_reason_states,
        finish_reason_reservations,
        rejected_finish_reasons,
        output_budget,
        finish_reason_budget,
        False,
    )

    assert finish_reason_states == {0: "stop"}
    assert rejected_finish_reasons == set()
    partial = implementation._chat_partial(
        states,
        finish_reason_states,
        None,
        False,
        output_budget.truncated or finish_reason_budget.truncated or bool(rejected_finish_reasons),
    )
    assert finish_reason_budget.truncated is False
    assert partial["attributes"] is None


def test_chat_streaming_incrementally_accounts_for_many_tiny_deltas(
    memory: SimpleNamespace,
) -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    finish_reason_reservations: dict[int, tuple[int, int]] = {}
    rejected_finish_reasons: set[int] = set()
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()

    for _ in range(60_000):
        implementation._record_chat_chunk(
            {"choices": [{"index": 0, "delta": {"content": "x"}}]},
            states,
            finish_reason_states,
            finish_reason_reservations,
            rejected_finish_reasons,
            output_budget,
            finish_reason_budget,
            True,
        )

    retained_content = "x" * 49_036
    assert output_budget.truncated is True
    assert states[0].content == retained_content
    assert states[0].content_fragments.getvalue() == retained_content
    assert output_budget.bytes_used == 48 * 1024
    assert json.dumps(implementation._chat_output(states), separators=(",", ":")) == json.dumps(
        [{"role": "assistant", "content": retained_content}], separators=(",", ":")
    )


def test_chat_streaming_replaces_default_role_within_exact_budget(
    memory: SimpleNamespace,
) -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    finish_reason_reservations: dict[int, tuple[int, int]] = {}
    rejected_finish_reasons: set[int] = set()
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()

    implementation._record_chat_chunk(
        {"choices": [{"index": 0, "delta": {}}]},
        states,
        finish_reason_states,
        finish_reason_reservations,
        rejected_finish_reasons,
        output_budget,
        finish_reason_budget,
        True,
    )
    retained_content = "x" * 49_036
    implementation._record_chat_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "developer", "content": retained_content},
                }
            ]
        },
        states,
        finish_reason_states,
        finish_reason_reservations,
        rejected_finish_reasons,
        output_budget,
        finish_reason_budget,
        True,
    )

    assert output_budget.bytes_used == output_budget.max_bytes
    assert output_budget.truncated is False
    assert states[0].role == "developer"
    assert states[0].content == retained_content


def test_chat_streaming_recovers_after_rejecting_role_replacement() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    finish_reason_reservations: dict[int, tuple[int, int]] = {}
    rejected_finish_reasons: set[int] = set()
    output_budget = small_chat_budget(160, reserve_output_list=True)
    finish_reason_budget = small_chat_budget(160)

    for role in ("x" * 256, "developer"):
        implementation._record_chat_chunk(
            {"choices": [{"index": 0, "delta": {"role": role}}]},
            states,
            finish_reason_states,
            finish_reason_reservations,
            rejected_finish_reasons,
            output_budget,
            finish_reason_budget,
            True,
        )

    assert states[0].role == "developer"
    assert states[0].role_resolved is True
    assert output_budget.truncated is False


def test_recoverable_scalar_capture_recovers_after_item_limit_rejection() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    budget = small_chat_budget(160)
    budget.items_used = budget.max_items

    assert (
        implementation._capture_chat_string("developer", budget, "role", recoverable=True) is False
    )
    assert budget.truncated is False

    budget.items_used = 0
    assert (
        implementation._capture_chat_string("developer", budget, "role", recoverable=True) is True
    )
    assert budget.truncated is False


def test_chat_streaming_replaces_retained_scalar_after_additive_truncation() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    finish_reason_reservations: dict[int, tuple[int, int]] = {}
    rejected_finish_reasons: set[int] = set()
    output_budget = small_chat_budget(240, reserve_output_list=True)
    finish_reason_budget = small_chat_budget(240)

    for event in (
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "old"}]}}]},
        {"choices": [{"index": 0, "delta": {"content": "x" * 300}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "new"}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ):
        implementation._record_chat_chunk(
            event,
            states,
            finish_reason_states,
            finish_reason_reservations,
            rejected_finish_reasons,
            output_budget,
            finish_reason_budget,
            True,
        )

    assert implementation._chat_output(states) == [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "new"}]}
    ]
    assert output_budget.truncated is True


@pytest.mark.parametrize("field", ["id", "type", "name"])
def test_chat_streaming_reclaims_replaced_tool_call_scalars(field: str) -> None:
    implementation = cast(Any, telemetry_dev_openai)
    state = implementation._ChatChoice()
    previous = "x" * 128
    if field == "name":
        state.tool_calls[0] = {"function": {"name": previous}}
        delta = {"index": 0, "function": {"name": "replaced"}}
    else:
        state.tool_calls[0] = {field: previous}
        delta = {"index": 0, field: "replaced"}
    budget = implementation._chat_capture_budget()
    budget.bytes_used = budget.max_bytes
    budget.items_used = 10

    implementation._capture_tool_call_delta(
        state, implementation._read_tool_call_delta(delta, budget), budget
    )

    captured = (
        state.tool_calls[0]["function"]["name"] if field == "name" else state.tool_calls[0][field]
    )
    assert captured == "replaced"
    assert budget.bytes_used == budget.max_bytes - (len(previous) - len("replaced"))
    assert budget.truncated is False


def test_chat_streaming_recovers_after_rejecting_tool_call_scalar_growth() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    state = implementation._ChatChoice()
    budget = implementation._chat_capture_budget()
    initial = implementation._read_tool_call_delta({"index": 0, "id": "xxxx"}, budget)
    implementation._capture_tool_call_delta(state, initial, budget)
    budget.bytes_used = budget.max_bytes
    delta = implementation._read_tool_call_delta({"index": 0, "id": '""""'}, budget)

    implementation._capture_tool_call_delta(state, delta, budget)

    assert "id" not in state.tool_calls[0]
    assert budget.bytes_used == budget.max_bytes - 38
    assert budget.truncated is False
    assert state.unresolved_tool_scalars == {0: {"id"}}
    assert "id" not in implementation._chat_message(state)["tool_calls"][0]

    recovery = implementation._read_tool_call_delta({"index": 0, "id": "ok"}, budget)
    implementation._capture_tool_call_delta(state, recovery, budget)

    assert state.tool_calls[0]["id"] == "ok"
    assert state.unresolved_tool_scalars == {}
    assert budget.bytes_used == budget.max_bytes - 2
    assert budget.truncated is False


def test_chat_streaming_persists_function_shell_after_rejected_name() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    state = implementation._ChatChoice()
    budget = small_chat_budget(225)

    rejected = implementation._read_tool_call_delta(
        {"index": 0, "function": {"name": "x" * 256}}, budget
    )
    implementation._capture_tool_call_delta(state, rejected, budget)

    assert state.tool_calls[0]["function"] == {}
    assert state.unresolved_tool_scalars == {0: {"function.name"}}
    shell_bytes = budget.bytes_used

    recovery = implementation._read_tool_call_delta(
        {"index": 0, "function": {"name": "lookup", "arguments": "{}"}}, budget
    )
    implementation._capture_tool_call_delta(state, recovery, budget)

    assert state.tool_calls[0]["function"]["name"] == "lookup"
    assert state.tool_calls[0]["function"]["arguments"].getvalue() == "{}"
    assert state.unresolved_tool_scalars == {}
    assert budget.bytes_used > shell_bytes
    assert budget.truncated is False


def test_chat_streaming_replaces_tool_call_null_content_at_exact_budget() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    finish_reason_reservations: dict[int, tuple[int, int]] = {}
    rejected_finish_reasons: set[int] = set()
    output_budget = small_chat_budget(256, reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()

    implementation._record_chat_chunk(
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call"}]}}]},
        states,
        finish_reason_states,
        finish_reason_reservations,
        rejected_finish_reasons,
        output_budget,
        finish_reason_budget,
        True,
    )
    retained_content = "x" * 44
    implementation._record_chat_chunk(
        {"choices": [{"index": 0, "delta": {"content": retained_content}}]},
        states,
        finish_reason_states,
        finish_reason_reservations,
        rejected_finish_reasons,
        output_budget,
        finish_reason_budget,
        True,
    )

    assert output_budget.remaining_bytes == 0
    assert output_budget.truncated is False
    assert states[0].content == retained_content


def test_chat_streaming_does_not_retain_repeated_empty_tool_arguments() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    state = implementation._ChatChoice()
    budget = implementation._chat_capture_budget()
    delta = implementation._read_tool_call_delta(
        {"index": 0, "function": {"arguments": ""}}, budget
    )

    for _ in range(10_000):
        implementation._capture_tool_call_delta(state, delta, budget)

    assert state.tool_calls[0]["function"]["arguments"].getvalue() == ""
    assert budget.truncated is False


def test_chat_streaming_rejects_the_first_tool_argument_byte_beyond_the_budget() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    state = implementation._ChatChoice()
    budget = small_chat_budget(256)
    empty = implementation._read_tool_call_delta(
        {"index": 0, "function": {"arguments": ""}}, budget
    )
    implementation._capture_tool_call_delta(state, empty, budget)
    fitting = implementation._read_tool_call_delta(
        {
            "index": 0,
            "function": {"arguments": "x" * 78},
        },
        budget,
    )
    implementation._capture_tool_call_delta(state, fitting, budget)
    arguments = state.tool_calls[0]["function"]["arguments"]

    assert arguments.getvalue() == "x" * 78
    assert budget.remaining_bytes == 0
    assert budget.truncated is False

    overflow = implementation._read_tool_call_delta(
        {"index": 0, "function": {"arguments": "y"}}, budget
    )
    implementation._capture_tool_call_delta(state, overflow, budget)

    assert arguments.getvalue() == "x" * 78
    assert budget.truncated is True


def test_chat_streaming_marks_failed_null_content_replacement_incomplete() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    output_budget = small_chat_budget(512, reserve_output_list=True)
    finish_reason_budget = small_chat_budget(512)

    implementation._record_chat_chunk(
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call"}]}}]},
        states,
        {},
        {},
        set(),
        output_budget,
        finish_reason_budget,
        True,
    )
    assert implementation._chat_message(states[0])["content"] is None
    content = "x" * (output_budget.remaining_bytes + 1)
    implementation._record_chat_chunk(
        {"choices": [{"index": 0, "delta": {"content": content}}]},
        states,
        {},
        {},
        set(),
        output_budget,
        finish_reason_budget,
        True,
    )

    assert implementation._chat_message(states[0])["content"] is None
    assert output_budget.truncated is True


def test_chat_streaming_caps_per_chunk_choice_and_tool_call_inspection() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    choice_reads = 0
    tool_call_reads = 0

    class HostileSequence(Sequence[Any]):
        def __init__(self, values: list[Any]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index: int | slice) -> Any:
            if isinstance(index, slice):
                if (index.stop or len(self.values)) > 1_000:
                    raise AssertionError("inspected beyond the per-chunk limit")
                return self.values[index]
            if index >= 1_000:
                raise AssertionError("inspected beyond the per-chunk limit")
            return self.values[index]

    class CountingChoice(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            nonlocal choice_reads
            if key == "delta":
                choice_reads += 1
            return super().get(key, default)

    choices = HostileSequence(
        [
            CountingChoice(
                index=index,
                delta={"content": "beyond-limit"} if index == 1_000 else {},
            )
            for index in range(5_000)
        ]
    )
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()
    choice_update = implementation._record_chat_chunk(
        {"choices": choices},
        {},
        {},
        {},
        set(),
        output_budget,
        finish_reason_budget,
        True,
    )

    class CountingToolCall(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            nonlocal tool_call_reads
            if key == "id":
                tool_call_reads += 1
            return super().get(key, default)

    tool_calls = HostileSequence(
        [
            CountingToolCall(
                index=index,
                id=f"call-{index}",
                function={"arguments": "beyond-limit"} if index == 1_000 else None,
            )
            for index in range(5_000)
        ]
    )
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()
    tool_update = implementation._record_chat_chunk(
        {"choices": [{"index": 0, "delta": {"tool_calls": tool_calls}}]},
        {},
        {},
        {},
        set(),
        output_budget,
        finish_reason_budget,
        True,
    )

    assert choice_reads == 1_000
    assert tool_call_reads == 1_000
    assert choice_update["has_output"] is True
    assert tool_update["has_output"] is True
    assert output_budget.truncated is True


def test_chat_streaming_retains_exact_content_for_all_fragment_accumulators() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()

    for _ in range(200):
        implementation._record_chat_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "c",
                            "refusal": "r",
                            "function_call": {"arguments": "f"},
                            "tool_calls": [{"index": 0, "function": {"arguments": "t"}}],
                        },
                    }
                ]
            },
            states,
            {},
            {},
            set(),
            output_budget,
            finish_reason_budget,
            True,
        )

    state = states[0]
    assert state.content == "c" * 200
    assert state.refusal == "r" * 200
    assert implementation._chat_message(state)["function_call"]["arguments"] == "f" * 200
    assert implementation._chat_message(state)["tool_calls"][0]["function"]["arguments"] == (
        "t" * 200
    )
    assert output_budget.items_used <= output_budget.max_items


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chat_streaming_reconstructs_legacy_function_calls_and_custom_tools(
    make: Any, async_mode: bool
) -> None:
    memory = make()
    implementation = cast(Any, telemetry_dev_openai)
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "function_call": {"name": "legacy_lookup", "arguments": '{"city":'},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "custom_1",
                                "type": "custom",
                                "custom": {"name": "code_exec", "input": "print("},
                            }
                        ],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "function_call": {"arguments": '"Paris"}'},
                        "tool_calls": [{"index": 0, "custom": {"input": "42)"}}],
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    ]
    handle = telemetry_dev.start_span(
        "chat gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            for event in events:
                yield event

        stream = implementation._InstrumentedAsyncStream(
            source(), handle, False, time.perf_counter()
        )
        delivered = [event async for event in stream]
    else:
        stream = implementation._InstrumentedStream(
            iter(events), handle, False, time.perf_counter()
        )
        delivered = list(stream)

    assert delivered == events
    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "legacy_lookup", "arguments": '{"city":"Paris"}'},
            "tool_calls": [
                {
                    "id": "custom_1",
                    "type": "custom",
                    "custom": {"name": "code_exec", "input": "print(42)"},
                }
            ],
        }
    ]


def test_chat_streaming_bounds_rejected_finish_reason_indexes() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    finish_reason_reservations: dict[int, tuple[int, int]] = {}
    rejected_finish_reasons: set[int] = set()
    output_budget = implementation._chat_capture_budget()
    finish_reason_budget = implementation._chat_capture_budget()
    oversized_reason = "x" * 70_000

    for indexes in (range(1_000), range(1_000, 1_100)):
        implementation._record_chat_chunk(
            {
                "choices": [
                    {"index": index, "delta": {}, "finish_reason": oversized_reason}
                    for index in indexes
                ]
            },
            states,
            finish_reason_states,
            finish_reason_reservations,
            rejected_finish_reasons,
            output_budget,
            finish_reason_budget,
            False,
        )

    assert len(rejected_finish_reasons) == 1_000
    assert finish_reason_budget.truncated is True


def test_chat_streaming_finish_reason_budget_counts_json_escapes() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    reservations: dict[int, tuple[int, int]] = {}
    budget = implementation._chat_capture_budget()
    reason = '"\\\n'

    assert implementation._replace_finish_reason(0, reason, budget, reservations) is True
    assert reservations[0] == (98 + len(json.dumps(reason).encode()) - 2, 5)


def test_chat_streaming_rejected_replacement_invalidates_retained_finish_reason() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    reservations: dict[int, tuple[int, int]] = {}
    rejected: set[int] = set()
    output_budget = implementation._chat_capture_budget()
    finish_reason_budget = small_chat_budget(130)

    for index, reason in ((0, "x" * 20), (0, "x" * 200), (1, "stop")):
        implementation._record_chat_chunk(
            {"choices": [{"index": index, "delta": {}, "finish_reason": reason}]},
            states,
            finish_reason_states,
            reservations,
            rejected,
            output_budget,
            finish_reason_budget,
            False,
        )

    assert finish_reason_states == {1: "stop"}
    assert set(reservations) == {1}
    assert rejected == {0}


def test_chat_streaming_mapping_attribute_error_is_reported_and_marks_capture() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    errors: list[BaseException] = []

    class ThrowingMapping(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AttributeError(f"cannot read {key}")

    states: dict[int, Any] = {}
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()

    update = implementation._record_chat_chunk(
        ThrowingMapping(),
        states,
        {},
        {},
        set(),
        output_budget,
        finish_reason_budget,
        True,
        errors.append,
    )

    assert update["has_output"] is False
    assert states == {}
    assert output_budget.truncated is True
    assert errors
    assert all(isinstance(error, AttributeError) for error in errors)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["record", "classifier"])
async def test_chat_stream_instrumentation_failures_do_not_interrupt_iteration(
    make: Any,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
    failure: str,
) -> None:
    errors: list[BaseException] = []
    memory = make(on_error=errors.append)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{failure} failed")

    monkeypatch.setattr(
        telemetry_dev_openai,
        "_record_chat_chunk" if failure == "record" else "_synthetic_usage_chunk",
        fail,
    )
    events = [terminal_chat_stream_event(), terminal_chat_stream_event()]

    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return sse_response(events)

        client = wrap_openai(async_client(handler), inject_stream_usage=True)
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        delivered = [chunk async for chunk in stream]
        await client.close()
    else:
        client = wrap_openai(
            sync_client(lambda _request: sse_response(events)), inject_stream_usage=True
        )
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        delivered = list(stream)
        client.close()

    assert len(delivered) == 2
    assert all(chunk.id == "chatcmpl_stream" for chunk in delivered)
    assert all(chunk.model == "gpt-4o-mini-2024-07-18" for chunk in delivered)
    assert all(chunk.choices[0].delta.role == "assistant" for chunk in delivered)
    assert all(chunk.choices[0].delta.content == "Hello" for chunk in delivered)
    assert all(chunk.choices[0].finish_reason == "stop" for chunk in delivered)
    assert [str(error) for error in errors if str(error) == f"{failure} failed"] == [
        f"{failure} failed"
    ]
    assert attrs(only_span(memory))["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chat_stream_instrumentation_propagates_process_control_exceptions(
    memory: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
) -> None:
    def interrupt(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(telemetry_dev_openai, "_record_chat_chunk", interrupt)
    events = [terminal_chat_stream_event()]

    if async_mode:

        async def handler(_request: httpx.Request) -> httpx.Response:
            return sse_response(events)

        client = wrap_openai(async_client(handler))
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        with pytest.raises(KeyboardInterrupt):
            [chunk async for chunk in stream]
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: sse_response(events)))
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        with pytest.raises(KeyboardInterrupt):
            list(stream)
        client.close()

    assert attrs(only_span(memory))["telemetry.dev.capture.truncated"] is True


def test_chat_streaming_ignores_inherited_tool_call_fields_and_getters() -> None:
    implementation = cast(Any, telemetry_dev_openai)
    getter_reads = {"outer": 0, "nested": 0}

    class InheritedToolCall:
        index = 7
        id = "inherited"
        type = "function"

        @property
        def function(self) -> object:
            getter_reads["outer"] += 1
            return {"name": "inherited", "arguments": "{}"}

    class InheritedFunction:
        name = "inherited"

        @property
        def arguments(self) -> str:
            getter_reads["nested"] += 1
            return "{}"

    nested = SimpleNamespace(index=3, function=InheritedFunction())
    states: dict[int, Any] = {}
    finish_reason_states: dict[int, str] = {}
    finish_reason_reservations: dict[int, tuple[int, int]] = {}
    rejected_finish_reasons: set[int] = set()
    output_budget = implementation._chat_capture_budget(reserve_output_list=True)
    finish_reason_budget = implementation._chat_capture_budget()

    implementation._record_chat_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [InheritedToolCall(), nested]},
                }
            ]
        },
        states,
        finish_reason_states,
        finish_reason_reservations,
        rejected_finish_reasons,
        output_budget,
        finish_reason_budget,
        True,
    )

    assert implementation._chat_output(states) == [{"role": "assistant"}]
    assert getter_reads == {"outer": 0, "nested": 0}
    assert output_budget.truncated is False


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chat_streaming_skips_output_state_when_capture_is_disabled(
    make: Any,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
) -> None:
    memory = make(capture_output=False)
    implementation = cast(Any, telemetry_dev_openai)
    original_own_field = implementation._own_field
    output_reads: list[str] = []

    def counting_own_field(value: Any, name: str, *args: Any) -> Any:
        if isinstance(value, ChoiceDelta) and name == "role":
            output_reads.append(name)
        return original_own_field(value, name, *args)

    monkeypatch.setattr(implementation, "_own_field", counting_own_field)
    events = chat_stream_events()

    if async_mode:

        async def async_handler(_request: httpx.Request) -> httpx.Response:
            return sse_response(events)

        client = wrap_openai(async_client(async_handler))
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        delivered = [chunk async for chunk in stream]
        states = stream._states
        await client.close()
    else:
        client = wrap_openai(sync_client(lambda _request: sse_response(events)))
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        delivered = list(stream)
        states = stream._states
        client.close()

    assert len(delivered) == len(events)
    assert states == {}
    assert output_reads == []
    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["gen_ai.usage.total_tokens"] == 7
    assert "telemetry.dev.capture.truncated" not in a


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chat_stream_clean_eof_without_finish_reason_is_incomplete_when_output_disabled(
    make: Any, async_mode: bool
) -> None:
    memory = make(capture_output=False)
    implementation = cast(Any, telemetry_dev_openai)
    events = [{"choices": [{"index": 0, "delta": {"content": "partial"}}]}]
    handle = telemetry_dev.start_span(
        "chat gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            for event in events:
                yield event

        stream = implementation._InstrumentedAsyncStream(
            source(), handle, False, time.perf_counter()
        )
        delivered = [event async for event in stream]
    else:
        stream = implementation._InstrumentedStream(
            iter(events), handle, False, time.perf_counter()
        )
        delivered = list(stream)

    assert delivered == events
    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert "gen_ai.response.finish_reasons" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def test_chat_stream_uses_the_client_policy_that_created_its_span(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return value

    original = make(
        capture_output=True,
        mask=mask,
        max_attribute_length=256,
    )
    handle = telemetry_dev.start_span(
        "chat gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )
    monkeypatch.setattr(
        telemetry_dev,
        "get_client",
        lambda: SimpleNamespace(capture_output=False, mask=None, max_attribute_length=1),
    )
    old_handle = SimpleNamespace(
        _client=cast(Any, handle)._client,
        _state=cast(Any, handle)._state,
        end=handle.end,
        update=handle.update,
    )
    implementation = cast(Any, telemetry_dev_openai)
    stream = implementation._InstrumentedStream(
        iter(
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "x" * 100},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ]
        ),
        old_handle,
        False,
        time.perf_counter(),
    )
    list(stream)

    a = attrs(only_span(original))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "x" * 100}
    ]
    assert "telemetry.dev.capture.truncated" not in a


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
    assert "gen_ai.response.finish_reasons" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def test_chat_stream_close_marks_truncation_when_output_capture_is_disabled(make: Any) -> None:
    memory = make(capture_output=False)
    client = wrap_openai(sync_client(lambda _request: sse_response(chat_stream_events())))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )

    next(stream)
    stream.close()

    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert "gen_ai.response.finish_reasons" not in a
    assert a["telemetry.dev.capture.truncated"] is True


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
    assert first_attrs["gen_ai.usage.cache_read.input_tokens"] == 6
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
    assert a["telemetry.dev.capture.truncated"] is True


def empty_created_then_oversized_terminal() -> list[tuple[str, dict[str, Any]]]:
    created = response_payload(text="")
    created["status"] = "in_progress"
    created["output"] = []
    completed = response_payload(text="x" * 70_000)
    completed["output"] = [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"a"}',
            "status": "completed",
        },
        *completed["output"],
    ]
    return [
        ("response.created", {"type": "response.created", "response": created}),
        ("response.completed", {"type": "response.completed", "response": completed}),
    ]


def assert_empty_created_output_not_reported(memory: SimpleNamespace) -> None:
    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_does_not_report_empty_created_output_for_truncated_terminal(
    memory: SimpleNamespace,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return named_sse_response(empty_created_then_oversized_terminal())

    list(
        cast(Any, wrap_openai(sync_client(handler)).responses.create)(
            model="gpt-4o-mini", input="x", stream=True
        )
    )
    assert_empty_created_output_not_reported(memory)


def test_responses_stream_fitting_terminal_clears_prior_truncation(memory: SimpleNamespace) -> None:
    retained = response_payload(text="partial")
    retained["status"] = "in_progress"
    oversized = response_payload(text="x" * 70_000)
    oversized["status"] = "in_progress"
    completed = response_payload(text="final")

    def handler(_request: httpx.Request) -> httpx.Response:
        return named_sse_response(
            [
                ("response.created", {"type": "response.created", "response": retained}),
                (
                    "response.in_progress",
                    {"type": "response.in_progress", "response": oversized},
                ),
                ("response.completed", {"type": "response.completed", "response": completed}),
            ]
        )

    list(
        cast(Any, wrap_openai(sync_client(handler)).responses.create)(
            model="gpt-4o-mini", input="x", stream=True
        )
    )

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"]))[0]["content"][0]["text"] == "final"
    assert "telemetry.dev.capture.truncated" not in a


@pytest.mark.parametrize("async_mode", [False, True])
async def test_responses_stream_mapping_failure_preserves_provider_events(
    make: Any, async_mode: bool
) -> None:
    errors: list[BaseException] = []
    memory = make(on_error=errors.append)
    mapping_error = RuntimeError("response mapping failed")
    retained_response = response_payload(response_id="resp_mapping_failure", text="public")
    retained = {
        "type": "response.in_progress",
        "response": retained_response,
    }

    class BrokenResponse:
        id = "resp_mapping_failure"
        status = "completed"

        @property
        def output(self) -> Any:
            raise mapping_error

    failed = {"type": "response.completed", "response": BrokenResponse()}
    events = [retained, failed]
    handle = telemetry_dev.start_span(
        "responses gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )
    implementation = cast(Any, telemetry_dev_openai)

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            for event in events:
                yield event

        stream = implementation._InstrumentedAsyncResponsesStream(
            source(), handle, time.perf_counter()
        )
        delivered = [event async for event in stream]
    else:
        stream = implementation._InstrumentedResponsesStream(
            iter(events), handle, time.perf_counter()
        )
        delivered = list(stream)

    assert delivered[0] is retained
    assert delivered[1] is failed
    assert errors == [mapping_error]
    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == retained_response["output"]
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
async def test_responses_stream_mask_omits_stale_output_after_terminal_truncation(
    make: Any, async_mode: bool
) -> None:
    def mask(value: Any, _context: telemetry_dev.MaskContext) -> Any:
        return "[REDACTED]" if "SECRET" in json.dumps(value) else value

    memory = make(mask=mask)
    retained = response_payload(response_id="resp_masked_terminal", text="public")
    retained["status"] = "in_progress"
    terminal = response_payload(response_id="resp_masked_terminal", text=f"SECRET{'x' * 70_000}")
    events = [
        {"type": "response.in_progress", "response": retained},
        {"type": "response.completed", "response": terminal},
    ]
    handle = telemetry_dev.start_span(
        "responses gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )
    implementation = cast(Any, telemetry_dev_openai)

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            for event in events:
                yield event

        stream = implementation._InstrumentedAsyncResponsesStream(
            source(), handle, time.perf_counter()
        )
        delivered = [event async for event in stream]
    else:
        stream = implementation._InstrumentedResponsesStream(
            iter(events), handle, time.perf_counter()
        )
        delivered = list(stream)

    assert delivered == events
    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
async def test_responses_stream_keeps_provider_failure_when_mapping_failed_event_throws(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    mapping_error = RuntimeError("response mapping failed")

    class FailedResponse:
        id = "resp_failed_mapping"
        status = "failed"
        error = SimpleNamespace(code="server_error", message="boom")

        @property
        def output(self) -> Any:
            raise mapping_error

    event = SimpleNamespace(type="response.failed", response=FailedResponse())
    reported: list[BaseException] = []
    handle = telemetry_dev.start_span(
        "responses gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )
    cast(Any, handle).report_error = reported.append
    implementation = cast(Any, telemetry_dev_openai)

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            yield event

        stream = implementation._InstrumentedAsyncResponsesStream(
            source(), handle, time.perf_counter()
        )
        delivered = [item async for item in stream]
    else:
        stream = implementation._InstrumentedResponsesStream(
            iter([event]), handle, time.perf_counter()
        )
        delivered = list(stream)

    assert delivered == [event]
    assert reported == [mapping_error]
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.events[0].attributes or {})["exception.message"] == (
        "response.failed: server_error: boom"
    )
    assert attrs(span)["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("event_type", ["transcript.text.delta", "image_generation.failed"])
async def test_media_stream_mapping_failure_fails_open(
    memory: SimpleNamespace, async_mode: bool, event_type: str
) -> None:
    mapping_error = RuntimeError("media mapping failed")

    class BrokenEvent:
        type = event_type
        error = SimpleNamespace(code="server_error", message="boom")

        @property
        def usage(self) -> Any:
            raise mapping_error

    events = [BrokenEvent(), BrokenEvent()]
    reported: list[BaseException] = []
    handle = telemetry_dev.start_span(
        "transcription gpt-4o-transcribe", type="generation", model="gpt-4o-transcribe"
    )
    cast(Any, handle).report_error = reported.append
    implementation = cast(Any, telemetry_dev_openai)

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            for event in events:
                yield event

        stream = implementation._InstrumentedAsyncMediaStream(
            source(), handle, implementation._text_media_response
        )
        delivered = [item async for item in stream]
    else:
        stream = implementation._InstrumentedMediaStream(
            iter(events), handle, implementation._text_media_response
        )
        delivered = list(stream)

    assert delivered == events
    assert reported == [mapping_error]
    span = only_span(memory)
    assert attrs(span)["telemetry.dev.capture.truncated"] is True
    if event_type.endswith(".failed"):
        assert span.status.status_code == StatusCode.ERROR
        assert dict(span.events[0].attributes or {})["exception.message"] == "server_error: boom"
    else:
        assert span.status.status_code != StatusCode.ERROR


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("stream_limited", [False, True])
async def test_responses_stream_retention_follows_the_stream_ceiling_not_the_cap(
    make: Any, monkeypatch: pytest.MonkeyPatch, async_mode: bool, stream_limited: bool
) -> None:
    if stream_limited:
        monkeypatch.setattr(telemetry_dev_openai, "_CHAT_STREAM_CAPTURE_MAX_BYTES", 128)
        memory = make()
    else:
        memory = make(max_attribute_length=128)
    terminal = response_payload(response_id="resp_small_limit", text="x" * 1_000)
    event = {"type": "response.completed", "response": terminal}
    handle = telemetry_dev.start_span(
        "responses gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )
    implementation = cast(Any, telemetry_dev_openai)

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            yield event

        stream = implementation._InstrumentedAsyncResponsesStream(
            source(), handle, time.perf_counter()
        )
        delivered = [item async for item in stream]
    else:
        stream = implementation._InstrumentedResponsesStream(
            iter([event]), handle, time.perf_counter()
        )
        delivered = list(stream)

    assert delivered == [event]
    a = attrs(only_span(memory))
    if stream_limited:
        assert "gen_ai.output.messages" not in a
        assert a["telemetry.dev.capture.truncated"] is True
    else:
        assert len(str(a["gen_ai.output.messages"])) == 128
        assert str(a["gen_ai.output.messages"]).endswith("...[truncated]")
        assert "telemetry.dev.capture.truncated" not in a


@pytest.mark.parametrize("async_mode", [False, True])
async def test_responses_stream_honors_hard_capture_limit(make: Any, async_mode: bool) -> None:
    memory = make(max_attribute_length=100 * 1024)
    terminal = response_payload(response_id="resp_hard_limit", text="x" * (60 * 1024))
    event = {"type": "response.completed", "response": terminal}
    handle = telemetry_dev.start_span(
        "responses gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )
    implementation = cast(Any, telemetry_dev_openai)

    if async_mode:

        async def source() -> AsyncIterator[Any]:
            yield event

        stream = implementation._InstrumentedAsyncResponsesStream(
            source(), handle, time.perf_counter()
        )
        delivered = [item async for item in stream]
    else:
        stream = implementation._InstrumentedResponsesStream(
            iter([event]), handle, time.perf_counter()
        )
        delivered = list(stream)

    assert delivered == [event]
    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("mode", ["eof", "early_close", "error_event", "throw"])
async def test_responses_stream_omits_masked_partial_output_after_interruption(
    make: Any, async_mode: bool, mode: str
) -> None:
    def identity_mask(value: Any, _context: Any) -> Any:
        return value

    memory = make(mask=identity_mask)
    retained_response = response_payload(response_id="resp_interrupted", text="partial")
    retained_response["status"] = "in_progress"
    retained = {"type": "response.in_progress", "response": retained_response}
    error_event = {"type": "error", "code": "stream_error", "message": "interrupted"}
    terminal = {
        "type": "response.completed",
        "response": {**retained_response, "status": "completed"},
    }
    stream_error = RuntimeError("responses stream interrupted")
    handle = telemetry_dev.start_span(
        "responses gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )
    implementation = cast(Any, telemetry_dev_openai)

    if async_mode:

        async def async_source() -> AsyncIterator[Any]:
            yield retained
            if mode == "error_event":
                yield error_event
            elif mode == "throw":
                raise stream_error
            elif mode == "early_close":
                yield terminal

        stream = implementation._InstrumentedAsyncResponsesStream(
            async_source(), handle, time.perf_counter()
        )
        if mode == "early_close":
            assert await stream.__anext__() is retained
            await stream.close()
        elif mode == "throw":
            with pytest.raises(RuntimeError, match="responses stream interrupted"):
                [item async for item in stream]
        else:
            [item async for item in stream]
    else:

        def source() -> Iterator[Any]:
            yield retained
            if mode == "error_event":
                yield error_event
            elif mode == "throw":
                raise stream_error
            elif mode == "early_close":
                yield terminal

        stream = implementation._InstrumentedResponsesStream(source(), handle, time.perf_counter())
        if mode == "early_close":
            assert next(stream) is retained
            stream.close()
        elif mode == "throw":
            with pytest.raises(RuntimeError, match="responses stream interrupted"):
                list(stream)
        else:
            list(stream)

    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
async def test_responses_stream_propagates_process_control_mapping_failures(
    make: Any, async_mode: bool
) -> None:
    errors: list[BaseException] = []
    make(on_error=errors.append)

    class InterruptedResponse:
        status = "completed"

        @property
        def output(self) -> Any:
            raise KeyboardInterrupt

    event = {"type": "response.completed", "response": InterruptedResponse()}
    handle = telemetry_dev.start_span(
        "responses gpt-4o-mini", type="generation", model="gpt-4o-mini", provider="openai"
    )
    implementation = cast(Any, telemetry_dev_openai)

    with pytest.raises(KeyboardInterrupt):
        if async_mode:

            async def source() -> AsyncIterator[Any]:
                yield event

            stream = implementation._InstrumentedAsyncResponsesStream(
                source(), handle, time.perf_counter()
            )
            [item async for item in stream]
        else:
            stream = implementation._InstrumentedResponsesStream(
                iter([event]), handle, time.perf_counter()
            )
            list(stream)

    assert errors == []


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
    assert state.role is None
    assert not state.content.endswith("not-retained")
    a = attrs(only_span(memory))
    assert len(str(a["gen_ai.output.messages"]).encode()) <= 48 * 1024
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["gen_ai.usage.total_tokens"] == 7
    assert a["telemetry.dev.capture.truncated"] is True


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
    assert response_attrs["gen_ai.usage.cache_read.input_tokens"] == 6
    assert response_attrs["gen_ai.usage.text.cache_read.input_tokens"] == 1
    assert response_attrs["gen_ai.usage.image.cache_read.input_tokens"] == 2
    assert response_attrs["gen_ai.usage.audio.cache_read.input_tokens"] == 3
    embedding_attrs = attrs(embedding_span)
    assert embedding_span.name == "embeddings text-embedding-3-small"
    assert embedding_attrs["gen_ai.operation.name"] == "embeddings"
    assert embedding_attrs["gen_ai.request.model"] == "text-embedding-3-small"
    assert embedding_attrs["gen_ai.response.model"] == "text-embedding-3-small"
    assert embedding_attrs["gen_ai.usage.input_tokens"] == 6
    assert embedding_attrs["gen_ai.usage.total_tokens"] == 6


@pytest.mark.parametrize("method", ["retrieve", "cancel"])
def test_batch_positional_id_is_recorded_when_sync_request_errors(
    memory: SimpleNamespace, method: str
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "batch failed"}})

    client = wrap_openai(sync_client(handler))

    with pytest.raises(openai.APIStatusError):
        getattr(client.batches, method)("batch_positional")

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["openai.batch.id"] == "batch_positional"


@pytest.mark.parametrize("method", ["retrieve", "cancel"])
async def test_batch_positional_id_is_recorded_when_async_request_errors(
    memory: SimpleNamespace, method: str
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "batch failed"}})

    client = wrap_openai(async_client(handler))

    with pytest.raises(openai.APIStatusError):
        await getattr(client.batches, method)("batch_positional")
    await client.close()

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["openai.batch.id"] == "batch_positional"


async def test_async_batch_list_remains_directly_async_iterable(memory: SimpleNamespace) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return json_response({"object": "list", "data": [], "has_more": False})

    client = wrap_openai(async_client(handler))
    batches = [batch async for batch in client.batches.list()]
    await client.close()

    assert batches == []
    assert memory.span_exporter.get_finished_spans() == ()


async def test_async_chat_stream_iteration_error_records_one_error_span(
    memory: SimpleNamespace,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return failing_async_sse_response(terminal_chat_stream_event(), "async stream broke")

    client = wrap_openai(async_client(handler))
    stream = await cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = await stream.__anext__()
    assert first.choices[0].delta.content == "Hello"

    with pytest.raises(Exception) as exc_info:
        await stream.__anext__()
    await client.close()

    assert_stream_transport_error(exc_info.value, "async stream broke")
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == type(exc_info.value).__name__
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["telemetry.dev.capture.truncated"] is True
    events = list(span.events)
    assert len(events) == 1
    event_attrs = dict(events[0].attributes or {})
    assert events[0].name == "exception"
    assert event_attrs["exception.type"] == type(exc_info.value).__name__
    assert event_attrs["exception.message"] == str(exc_info.value)


@pytest.mark.parametrize("async_mode", [False, True])
async def test_chat_stream_terminal_chunk_then_early_close_is_incomplete(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    events = [terminal_chat_stream_event(), *chat_stream_events()[1:]]

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    async def async_handler(request: httpx.Request) -> httpx.Response:
        return sse_response(events)

    if async_mode:
        client = wrap_openai(async_client(async_handler))
        stream = await cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        assert (await stream.__anext__()).choices[0].delta.content == "Hello"
        await stream.close()
        await client.close()
    else:
        client = wrap_openai(sync_client(handler))
        stream = cast(Any, client.chat.completions.create)(
            model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
        )
        assert next(stream).choices[0].delta.content == "Hello"
        stream.close()
        client.close()

    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 1
    a = attrs(spans[0])
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["telemetry.dev.capture.truncated"] is True


def test_chat_stream_plain_for_break_ends_span_once(memory: SimpleNamespace) -> None:
    client = wrap_openai(sync_client(lambda _request: sse_response(chat_stream_events())))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )

    for chunk in stream:
        assert chunk.choices[0].delta.content == "Hello"
        break

    span = only_span(memory)
    assert json.loads(str(attrs(span)["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
    assert attrs(span)["telemetry.dev.capture.truncated"] is True
    client.close()


def test_chat_stream_iteration_error_records_one_error_span(memory: SimpleNamespace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return failing_sync_sse_response(terminal_chat_stream_event(), "sync stream broke")

    client = wrap_openai(sync_client(handler))
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )
    first = next(stream)
    assert first.choices[0].delta.content == "Hello"

    with pytest.raises(Exception) as exc_info:
        next(stream)

    assert_stream_transport_error(exc_info.value, "sync stream broke")
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == type(exc_info.value).__name__
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "Hello"}
    ]
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["telemetry.dev.capture.truncated"] is True
    events = list(span.events)
    assert len(events) == 1
    event_attrs = dict(events[0].attributes or {})
    assert events[0].name == "exception"
    assert event_attrs["exception.type"] == type(exc_info.value).__name__
    assert event_attrs["exception.message"] == str(exc_info.value)


def test_nonterminal_chat_stream_error_omits_finish_reason(memory: SimpleNamespace) -> None:
    first_event = chat_stream_events()[0]
    client = wrap_openai(
        sync_client(lambda _request: failing_sync_sse_response(first_event, "stream broke early"))
    )
    stream = cast(Any, client.chat.completions.create)(
        model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True
    )

    next(stream)
    with pytest.raises(Exception) as exc_info:
        next(stream)
    assert_stream_transport_error(exc_info.value, "stream broke early")

    a = attrs(only_span(memory))
    assert "gen_ai.response.finish_reasons" not in a
    assert a["telemetry.dev.capture.truncated"] is True


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


def test_uninstrument_preserves_a_later_class_owner() -> None:
    original = Completions.create

    def later_owner(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original(self, *args, **kwargs)

    instrument_openai()
    Completions.create = later_owner
    uninstrument_openai()

    assert Completions.create is later_owner
    Completions.create = original


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
    assert a["telemetry.dev.capture.truncated"] is True


async def test_async_responses_stream_does_not_report_empty_created_output_for_truncated_terminal(
    memory: SimpleNamespace,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return named_sse_response(empty_created_then_oversized_terminal())

    client = wrap_openai(async_client(handler))
    stream = await cast(Any, client.responses.create)(model="gpt-4o-mini", input="x", stream=True)
    [event async for event in stream]
    await client.close()
    assert_empty_created_output_not_reported(memory)


async def test_async_responses_stream_fitting_terminal_clears_prior_truncation(
    memory: SimpleNamespace,
) -> None:
    retained = response_payload(text="partial")
    retained["status"] = "in_progress"
    oversized = response_payload(text="x" * 70_000)
    oversized["status"] = "in_progress"
    completed = response_payload(text="final")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return named_sse_response(
            [
                ("response.created", {"type": "response.created", "response": retained}),
                (
                    "response.in_progress",
                    {"type": "response.in_progress", "response": oversized},
                ),
                ("response.completed", {"type": "response.completed", "response": completed}),
            ]
        )

    client = wrap_openai(async_client(handler))
    stream = await cast(Any, client.responses.create)(model="gpt-4o-mini", input="x", stream=True)
    assert len([event async for event in stream]) == 3
    await client.close()

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"]))[0]["content"][0]["text"] == "final"
    assert "telemetry.dev.capture.truncated" not in a


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
    assert attrs(spans[0])["telemetry.dev.capture.truncated"] is True


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
    assert attrs(spans[0])["telemetry.dev.capture.truncated"] is True


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
    assert attrs(spans[0])["telemetry.dev.capture.truncated"] is True


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
    assert attrs(spans[0])["telemetry.dev.capture.truncated"] is True


def test_text_media_response_captures_text_and_falls_back_text_tokens(
    memory: SimpleNamespace,
) -> None:
    mapper = vars(telemetry_dev_openai)["_text_media_response"]
    fields = mapper(
        SimpleNamespace(text="transcript", usage=SimpleNamespace(output_tokens=7, total_tokens=7))
    )
    assert fields["output"] == "transcript"
    assert fields["usage"]["text_output_tokens"] == 7


def test_text_media_response_passes_the_complete_transcript_to_the_core(make: Any) -> None:
    make(max_attribute_length=5)
    mapper = vars(telemetry_dev_openai)["_text_media_response"]

    fields = mapper(SimpleNamespace(text="ééé"))

    assert fields["output"] == "ééé"
    assert "attributes" not in fields


def test_bounded_responses_conversion_does_not_model_dump(memory: SimpleNamespace) -> None:
    class Bomb:
        def __init__(self) -> None:
            self.output = [{"type": "output_text", "text": "x" * 100_000}]
            self.status = "completed"

        def model_dump(self, **kwargs: Any) -> Any:
            raise AssertionError("model_dump must not run")

    fields = vars(telemetry_dev_openai)["_responses_response"](Bomb())
    assert fields["output"] is None
    assert fields["attributes"]["telemetry.dev.capture.truncated"] is True


def test_bounded_responses_capture_repeats_shared_acyclic_values(
    memory: SimpleNamespace,
) -> None:
    shared = {"text": "same"}

    captured, truncated = vars(telemetry_dev_openai)["_bounded_responses_capture"]([shared, shared])

    assert captured == [{"text": "same"}, {"text": "same"}]
    assert truncated is False


def test_bounded_responses_capture_normalizes_opaque_values(memory: SimpleNamespace) -> None:
    class Opaque:
        __slots__ = ()

    captured, truncated = vars(telemetry_dev_openai)["_bounded_responses_capture"](
        {"opaque": Opaque()}
    )

    assert captured == {"opaque": None}
    assert json.dumps(captured) == '{"opaque": null}'
    assert truncated is False


def test_bounded_responses_capture_propagates_process_control_exceptions(
    memory: SimpleNamespace,
) -> None:
    class InterruptingMapping(dict[str, Any]):
        def items(self) -> Any:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        vars(telemetry_dev_openai)["_bounded_responses_capture"](
            InterruptingMapping(value="unread")
        )


@pytest.mark.parametrize("kind", ["oversized", "deep", "cyclic", "hostile"])
def test_responses_input_capture_failure_never_blocks_provider(
    memory: SimpleNamespace, kind: str
) -> None:
    class HostileMapping(dict[str, Any]):
        def items(self) -> Any:
            raise RuntimeError("hostile mapping")

    if kind == "oversized":
        value: Any = {"text": "x" * 70_000}
    elif kind == "deep":
        value = {}
        cursor = value
        for _ in range(40):
            child: dict[str, Any] = {}
            cursor["child"] = child
            cursor = child
    elif kind == "cyclic":
        value = {}
        value["self"] = value
    else:
        value = HostileMapping(value="x")
    called = False

    def provider(**_kwargs: Any) -> SimpleNamespace:
        nonlocal called
        called = True
        return SimpleNamespace()

    def response_mapper(_response: Any) -> dict[str, Any]:
        return {}

    def provider_resolver(_resource: object | None) -> str:
        return "openai"

    wrapper = vars(telemetry_dev_openai)["_wrap_sync"](
        provider,
        "responses",
        vars(telemetry_dev_openai)["_responses_request"],
        response_mapper,
        provider_resolver,
        False,
    )

    wrapper(model="gpt-4.1", input=value)
    assert called

    if kind == "oversized":
        span = memory.span_exporter.get_finished_spans()[-1]
        assert "gen_ai.input.messages" not in attrs(span)
        assert attrs(span)["telemetry.dev.capture.truncated"] is True
