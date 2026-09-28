from __future__ import annotations

import importlib
import json
import time
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
from openai.types.chat import ChatCompletionChunk
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
    assert attrs(only_span(memory))["gen_ai.usage.image.output_tokens"] == 9


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
    a = attrs(only_span(memory))
    assert a["gen_ai.output.messages"] == "complete transcript"
    assert a["gen_ai.usage.text.output_tokens"] == 3


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


def test_terminal_transcription_capture_is_incrementally_bounded(make: Any) -> None:
    make(max_attribute_length=5)
    mapper = vars(telemetry_dev_openai)["_media_stream_event_fields"]

    fields = mapper(
        SimpleNamespace(type="transcript.text.done", text=UnencodableText("ééé")),
        vars(telemetry_dev_openai)["_text_media_response"],
        True,
    )

    assert fields["output"] == "éé"
    assert fields["attributes"]["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("async_mode", [False, True])
async def test_transcription_delta_capture_is_incrementally_bounded(
    make: Any, async_mode: bool
) -> None:
    make(max_attribute_length=5)
    ended: list[dict[str, Any]] = []
    event = SimpleNamespace(type="transcript.text.delta", delta=UnencodableText("ééé"))

    class Handle:
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

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float
    ) -> telemetry_dev.SpanHandle:
        recorded.append(timestamp_ms)
        return original_record(handle, timestamp_ms)

    def perf_counter() -> float:
        return clock

    original_getattribute = ChatCompletionChunk.__getattribute__

    def delayed_choices(chunk: ChatCompletionChunk, name: str) -> Any:
        nonlocal clock
        if name == "choices":
            clock = 47.0
        return original_getattribute(chunk, name)

    monkeypatch.setattr(time, "perf_counter", perf_counter)
    monkeypatch.setattr(ChatCompletionChunk, "__getattribute__", delayed_choices)
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

    assert_stream_transport_error(exc_info.value, "async stream broke")
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

    assert_stream_transport_error(exc_info.value, "sync stream broke")
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


def test_text_media_response_captures_text_and_falls_back_text_tokens(
    memory: SimpleNamespace,
) -> None:
    mapper = vars(telemetry_dev_openai)["_text_media_response"]
    fields = mapper(
        SimpleNamespace(text="transcript", usage=SimpleNamespace(output_tokens=7, total_tokens=7))
    )
    assert fields["output"] == "transcript"
    assert fields["usage"]["text_output_tokens"] == 7


def test_text_media_response_capture_is_incrementally_bounded(make: Any) -> None:
    make(max_attribute_length=5)
    mapper = vars(telemetry_dev_openai)["_text_media_response"]

    fields = mapper(SimpleNamespace(text=UnencodableText("ééé")))

    assert fields["output"] == "éé"
    assert fields["attributes"]["telemetry.dev.capture.truncated"] is True


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
