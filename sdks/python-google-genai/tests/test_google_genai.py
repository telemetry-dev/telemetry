# Tests monkeypatch google-genai's protected transport seam and inspect dynamic pydantic output.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportPrivateUsage=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import gc
import json
import weakref
from collections.abc import AsyncIterator, Iterator, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
import telemetry_dev
from google import genai
from google.genai import errors, types
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

import telemetry_dev_google_genai as google_genai_instrumentation
from telemetry_dev_google_genai import (
    instrument_google_genai,
    uninstrument_google_genai,
    wrap_google_genai,
)

USER_CONTENT = [{"role": "user", "parts": [{"text": "Tell me a joke about OpenTelemetry"}]}]
SYSTEM_INSTRUCTION = {"parts": [{"text": "You must never tell jokes"}]}


def redact_media(_value: Any, context: telemetry_dev.MaskContext) -> str:
    return f"redacted:{context.key}"


def only_span(env: SimpleNamespace) -> ReadableSpan:
    spans = env.span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


class FakeTransport:
    def __init__(self, queue: list[Any]) -> None:
        self.queue = list(queue)
        self.calls: list[dict[str, Any]] = []

    def _next(self) -> Any:
        if not self.queue:
            raise AssertionError("transport queue exhausted")
        item = self.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def request(
        self,
        http_method: str,
        path: str,
        request_dict: dict[str, object],
        http_options: object | None = None,
    ) -> types.HttpResponse:
        self.calls.append(request_dict)
        payload = self._next()
        return types.HttpResponse(headers={}, body=json.dumps(payload))

    def request_streamed(
        self,
        http_method: str,
        path: str,
        request_dict: dict[str, object],
        http_options: object | None = None,
    ) -> Iterator[types.HttpResponse]:
        self.calls.append(request_dict)
        payloads = self._next()
        for payload in payloads:
            if isinstance(payload, BaseException):
                raise payload
            yield types.HttpResponse(headers={}, body=json.dumps(payload))

    async def async_request(
        self,
        http_method: str,
        path: str,
        request_dict: dict[str, object],
        http_options: object | None = None,
    ) -> types.HttpResponse:
        return self.request(http_method, path, request_dict, http_options)

    async def async_request_streamed(
        self,
        http_method: str,
        path: str,
        request_dict: dict[str, object],
        http_options: object | None = None,
    ) -> AsyncIterator[types.HttpResponse]:
        self.calls.append(request_dict)
        payloads = self._next()

        async def async_generator() -> AsyncIterator[types.HttpResponse]:
            for payload in payloads:
                if isinstance(payload, BaseException):
                    raise payload
                yield types.HttpResponse(headers={}, body=json.dumps(payload))

        return async_generator()


def client_with_transport(queue: list[Any]) -> tuple[Any, FakeTransport]:
    client = genai.Client(api_key="test")
    transport = FakeTransport(queue)
    client._api_client.request = transport.request
    client._api_client.request_streamed = transport.request_streamed
    client._api_client.async_request = transport.async_request
    client._api_client.async_request_streamed = transport.async_request_streamed
    return client, transport


def _native_history(response: Any) -> list[Any]:
    history = response.automatic_function_calling_history or []
    out: list[Any] = []
    for item in history:
        parts = []
        for part in item.parts or []:
            native = part.model_dump(mode="json", by_alias=True, exclude_none=True)
            parts.append(native)
        out.append({"role": item.role, "parts": parts})
    return out


def happy_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "Telemetry works."}]},
                "finishReason": "STOP",
            }
        ],
        "modelVersion": "gemini-2.5-flash-001",
        "responseId": "resp_123",
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "totalTokenCount": 18,
            "cachedContentTokenCount": 3,
            "thoughtsTokenCount": 2,
            "toolUsePromptTokenCount": 1,
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 8},
                {"modality": "IMAGE", "tokenCount": 3},
            ],
            "candidatesTokensDetails": [{"modality": "AUDIO", "tokenCount": 7}],
            "cacheTokensDetails": [{"modality": "TEXT", "tokenCount": 3}],
        },
    }
    payload.update(overrides)
    return payload


def stream_chunks(**final_overrides: Any) -> list[dict[str, Any]]:
    final = {
        "candidates": [
            {
                "index": 0,
                "finishReason": "STOP",
                "content": {"role": "model", "parts": []},
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 2,
            "totalTokenCount": 7,
        },
    }
    final.update(final_overrides)
    return [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Hello"}]},
                }
            ],
            "responseId": "stream_123",
            "modelVersion": "gemini-2.5-flash-001",
        },
        {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": " world"}]}}]},
        final,
    ]


def embed_payload() -> dict[str, Any]:
    return {
        "embeddings": [{"values": [0.1, 0.2, 0.3], "statistics": {"tokenCount": 6}}],
        "metadata": {"billableCharacterCount": 12},
    }


def test_generate_images_is_media_span_without_binary_capture(memory: SimpleNamespace) -> None:
    client, _transport = client_with_transport(
        [{"generatedImages": [{"image": {"imageBytes": "AAAA", "mimeType": "image/png"}}]}]
    )
    wrapped = wrap_google_genai(client)

    wrapped.models.generate_images(model="imagen-4.0-generate-001", prompt="A graph")

    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "generate_content"
    assert a["gen_ai.output.type"] == "image"
    assert a["gen_ai.provider.name"] == "gcp.gemini"
    assert "AAAA" not in json.dumps(a)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"capture_input": False, "capture_output": False}, None),
        ({"mask": redact_media}, "redacted"),
    ],
    ids=["disabled", "masked"],
)
def test_media_content_obeys_capture_privacy_controls(
    make: Any, options: dict[str, Any], expected: str | None
) -> None:
    env = make(**options)
    client, _transport = client_with_transport([])

    def generate_images(**_kwargs: Any) -> dict[str, Any]:
        return {"generatedImages": [{"image": {"gcsUri": "gs://secret/output.png"}}]}

    client.models.generate_images = generate_images
    wrapped = wrap_google_genai(client)

    wrapped.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt="secret prompt",
        config={
            "negative_prompt": "secret negative prompt",
            "labels": {"private": "secret label"},
            "output_gcs_uri": "gs://secret/output-prefix",
        },
    )

    a = attrs(only_span(env))
    serialized = json.dumps(a)
    assert "secret prompt" not in serialized
    assert "secret label" not in serialized
    assert "gs://secret" not in serialized
    assert a["google_genai.response.image_count"] == 1
    if expected is None:
        assert "gen_ai.input.messages" not in a
        assert "gen_ai.output.messages" not in a
    else:
        assert expected in str(a["gen_ai.input.messages"])
        assert expected in str(a["gen_ai.output.messages"])


class _TraversalGuard(Sequence[Any]):
    def __len__(self) -> int:
        raise AssertionError("media input was traversed")

    def __getitem__(self, _index: int | slice) -> Any:
        raise AssertionError("media input was traversed")


class _LargeReferences(Sequence[Any]):
    def __init__(self) -> None:
        self.reads = 0

    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            raise AssertionError("reference images must be traversed incrementally")
        if index >= len(self):
            raise IndexError
        self.reads += 1
        if self.reads > 400:
            raise AssertionError("reference image traversal exceeded the capture budget")
        return {"reference_image": {"uri": f"gs://bucket/reference-{index}.png"}}


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_media_capture_disabled_does_not_traverse_inputs(make: Any, async_mode: bool) -> None:
    env = make(capture_input=False)
    client, _transport = client_with_transport([])
    called = False

    if async_mode:

        async def async_generate_images(**_kwargs: Any) -> dict[str, Any]:
            nonlocal called
            called = True
            return {"generatedImages": []}

        client.aio.models.generate_images = async_generate_images
    else:

        def sync_generate_images(**_kwargs: Any) -> dict[str, Any]:
            nonlocal called
            called = True
            return {"generatedImages": []}

        client.models.generate_images = sync_generate_images
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        asyncio.run(
            wrapped.aio.models.generate_images(
                model="imagen-4.0", reference_images=_TraversalGuard()
            )
        )
    else:
        wrapped.models.generate_images(model="imagen-4.0", reference_images=_TraversalGuard())

    assert called
    assert "gen_ai.input.messages" not in attrs(only_span(env))


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_media_capture_bounds_large_and_cyclic_inputs(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    client, _transport = client_with_transport([])
    references = _LargeReferences()
    labels: dict[str, Any] = {}
    labels["cycle"] = labels

    if async_mode:

        async def async_generate_images(**_kwargs: Any) -> dict[str, Any]:
            return {"generatedImages": []}

        client.aio.models.generate_images = async_generate_images
    else:

        def sync_generate_images(**_kwargs: Any) -> dict[str, Any]:
            return {"generatedImages": []}

        client.models.generate_images = sync_generate_images
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        asyncio.run(
            wrapped.aio.models.generate_images(
                model="imagen-4.0",
                reference_images=references,
                config={"labels": labels},
            )
        )
    else:
        wrapped.models.generate_images(
            model="imagen-4.0", reference_images=references, config={"labels": labels}
        )

    a = attrs(only_span(memory))
    assert a["telemetry.dev.capture.truncated"] is True
    assert references.reads <= 400


def test_media_capture_projects_cyclic_labels(memory: SimpleNamespace) -> None:
    labels: dict[str, Any] = {"ordinary": "value"}
    labels["cycle"] = labels

    contents, truncated = google_genai_instrumentation._media_contents(
        {"config": {"labels": labels}}
    )

    assert truncated is False
    assert contents == {"labels": {"ordinary": "value", "cycle": None}}


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("submission_done", [False, None], ids=["done-false", "done-omitted"])
def test_generate_videos_span_stays_open_until_terminal_poll(
    memory: SimpleNamespace, async_mode: bool, submission_done: bool | None
) -> None:
    submission: dict[str, Any] = {"name": "operations/video-1"}
    if submission_done is not None:
        submission["done"] = submission_done
    client, _transport = client_with_transport([submission])
    pending_operation = types.GenerateVideosOperation.model_validate(
        {"name": "operations/video-1", "done": False}
    )
    completed_operation = types.GenerateVideosOperation.model_validate(
        {
            "name": "operations/video-1",
            "done": True,
            "response": {"generatedVideos": [{"video": {"uri": "gs://bucket/video.mp4"}}]},
        }
    )
    polls = iter((pending_operation, completed_operation))

    def get_operation(operation: Any) -> Any:
        return next(polls)

    client.operations.get = get_operation

    async def async_get(operation: Any) -> Any:
        return next(polls)

    client.aio.operations.get = async_get
    wrapped = wrap_google_genai(client)
    source = types.GenerateVideosSource(
        prompt="A telemetry graph in motion",
        image=types.Image(image_bytes=b"image-secret", mime_type="image/png"),
        video=types.Video(video_bytes=b"video-secret", mime_type="video/mp4"),
    )

    if async_mode:

        async def run() -> tuple[Any, Any, Any]:
            submitted = await wrapped.aio.models.generate_videos(model="veo-3.0", source=source)
            assert memory.span_exporter.get_finished_spans() == ()
            pending = await wrapped.aio.operations.get(submitted)
            assert memory.span_exporter.get_finished_spans() == ()
            completed = await wrapped.aio.operations.get(pending)
            return submitted, pending, completed

        import asyncio

        submitted, pending, completed = asyncio.run(run())
    else:
        submitted = wrapped.models.generate_videos(model="veo-3.0", source=source)
        assert memory.span_exporter.get_finished_spans() == ()
        pending = wrapped.operations.get(submitted)
        assert memory.span_exporter.get_finished_spans() == ()
        completed = wrapped.operations.get(pending)

    assert submitted.name == pending.name == completed.name
    a = attrs(only_span(memory))
    serialized = json.dumps(a)
    assert a["gen_ai.output.type"] == "video"
    assert a["gen_ai.response.id"] == "operations/video-1"
    assert a["google_genai.response.operation_done"] is True
    assert a["google_genai.response.video_count"] == 1
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"type": "video", "uri": "gs://bucket/video.mp4"}
    ]
    assert "google_genai.response.video_uris" not in a
    assert "A telemetry graph in motion" in serialized
    assert "image-secret" not in serialized
    assert "video-secret" not in serialized


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("global_instrumentation", [False, True], ids=["wrapped", "global"])
def test_client_close_ends_pending_video_span(
    memory: SimpleNamespace, async_mode: bool, global_instrumentation: bool
) -> None:
    if global_instrumentation:
        instrument_google_genai()
    client, _transport = client_with_transport([{"name": "operations/close", "done": False}])
    wrapped = client if global_instrumentation else wrap_google_genai(client)

    if async_mode:

        async def run() -> None:
            await wrapped.aio.models.generate_videos(model="veo-3.0", prompt="storm")
            assert memory.span_exporter.get_finished_spans() == ()
            await wrapped.aio.aclose()

        import asyncio

        asyncio.run(run())
    else:
        wrapped.models.generate_videos(model="veo-3.0", prompt="storm")
        assert memory.span_exporter.get_finished_spans() == ()
        wrapped.close()

    span = only_span(memory)
    assert span.status.status_code is StatusCode.ERROR
    event = next(event for event in span.events if event.name == "exception")
    assert "client closed" in str((event.attributes or {})["exception.message"])


@pytest.mark.parametrize(
    ("async_mode", "global_instrumentation"),
    [(False, False), (True, True)],
    ids=["wrapped-sync", "global-async"],
)
def test_client_context_exit_ends_pending_video_span(
    memory: SimpleNamespace, async_mode: bool, global_instrumentation: bool
) -> None:
    if global_instrumentation:
        instrument_google_genai()
    client, _transport = client_with_transport([{"name": "operations/context", "done": False}])
    wrapped = client if global_instrumentation else wrap_google_genai(client)

    if async_mode:

        async def run() -> None:
            async with wrapped.aio:
                await wrapped.aio.models.generate_videos(model="veo-3.0", prompt="storm")
                assert memory.span_exporter.get_finished_spans() == ()

        import asyncio

        asyncio.run(run())
    else:
        with wrapped:
            wrapped.models.generate_videos(model="veo-3.0", prompt="storm")
            assert memory.span_exporter.get_finished_spans() == ()

    span = only_span(memory)
    assert span.status.status_code is StatusCode.ERROR
    event = next(event for event in span.events if event.name == "exception")
    assert "client closed" in str((event.attributes or {})["exception.message"])


def test_global_video_tracker_releases_closed_client_identity() -> None:
    class Scope:
        pass

    class Resource:
        def __init__(self, scope: Scope) -> None:
            self._api_client = scope

    tracker = google_genai_instrumentation._VideoOperationTracker(scoped=True)
    scope = Scope()
    resource = Resource(scope)
    tracker.close_scope(resource, "sync")
    scope_reference = weakref.ref(scope)

    assert len(tracker._closed_scopes) == 1

    del resource
    del scope
    gc.collect()

    assert scope_reference() is None
    assert len(tracker._closed_scopes) == 0


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_generate_videos_poll_exception_ends_span_and_preserves_error(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    failure = RuntimeError("poll transport failed")
    client, _transport = client_with_transport(
        [{"name": "operations/video-error", "done": False}, failure]
    )
    wrapped = wrap_google_genai(client)

    if async_mode:

        async def run() -> None:
            submitted = await wrapped.aio.models.generate_videos(model="veo-3.0", prompt="storm")
            with pytest.raises(RuntimeError, match="poll transport failed") as caught:
                await wrapped.aio.operations.get(submitted)
            assert caught.value is failure

        import asyncio

        asyncio.run(run())
    else:
        submitted = wrapped.models.generate_videos(model="veo-3.0", prompt="storm")
        with pytest.raises(RuntimeError, match="poll transport failed") as caught:
            wrapped.operations.get(submitted)
        assert caught.value is failure

    span = only_span(memory)
    assert span.status.status_code is StatusCode.ERROR


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_generate_videos_terminal_operation_error_ends_span(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    client, _transport = client_with_transport(
        [{"name": "operations/video-terminal-error", "done": False}]
    )
    failed = types.GenerateVideosOperation.model_validate(
        {
            "name": "operations/video-terminal-error",
            "done": True,
            "error": {"code": 13, "message": "generation failed"},
        }
    )

    def get_operation(operation: Any) -> Any:
        return failed

    client.operations.get = get_operation

    async def async_get(operation: Any) -> Any:
        return failed

    client.aio.operations.get = async_get
    wrapped = wrap_google_genai(client)

    if async_mode:

        async def run() -> Any:
            submitted = await wrapped.aio.models.generate_videos(model="veo-3.0", prompt="storm")
            return await wrapped.aio.operations.get(submitted)

        import asyncio

        result = asyncio.run(run())
    else:
        submitted = wrapped.models.generate_videos(model="veo-3.0", prompt="storm")
        result = wrapped.operations.get(submitted)

    assert result is failed
    span = only_span(memory)
    assert span.status.status_code is StatusCode.ERROR
    assert attrs(span)["google_genai.response.operation_done"] is True


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_generate_content_maps_request_response_usage_and_sampling(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    client, transport = client_with_transport([happy_payload()])
    wrapped = wrap_google_genai(client)
    config = {
        "systemInstruction": SYSTEM_INSTRUCTION,
        "temperature": 0.2,
        "topP": 0.9,
        "topK": 40,
        "maxOutputTokens": 64,
        "stopSequences": ["END"],
        "seed": 7,
        "frequencyPenalty": 0.1,
        "presencePenalty": 0.2,
    }
    expected_input = json.loads(json.dumps(USER_CONTENT))

    if async_mode:

        async def run() -> Any:
            return await wrapped.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=expected_input,
                config=config,
            )

        import asyncio

        response = asyncio.run(run())
    else:
        response = wrapped.models.generate_content(
            model="gemini-2.5-flash",
            contents=expected_input,
            config=config,
        )

    assert response.text == "Telemetry works."
    span = only_span(memory)
    assert span.name == "chat gemini-2.5-flash"
    a = attrs(span)
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "gcp.gemini"
    assert a["gen_ai.request.model"] == "gemini-2.5-flash"
    assert a["gen_ai.response.model"] == "gemini-2.5-flash-001"
    assert a["gen_ai.response.id"] == "resp_123"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["STOP"]
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
    assert a["gen_ai.usage.text.input_tokens"] == 8
    assert a["gen_ai.usage.image.input_tokens"] == 3
    assert a["gen_ai.usage.audio.output_tokens"] == 7
    assert a["gen_ai.usage.text.cache_read.input_tokens"] == 3
    assert a["google_genai.usage.tool_use_prompt_tokens"] == 1
    assert json.loads(str(a["gen_ai.input.messages"])) == expected_input
    assert expected_input == USER_CONTENT
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Telemetry works."}]}
    ]
    assert json.loads(str(a["gen_ai.system_instructions"])) == SYSTEM_INSTRUCTION
    assert len(transport.calls) == 1


def test_generate_content_normalizes_shorthand_contents(memory: SimpleNamespace) -> None:
    client, _transport = client_with_transport([happy_payload()])
    wrapped = wrap_google_genai(client)

    wrapped.models.generate_content(
        model="gemini-2.5-flash",
        contents=["hello", {"inline_data": {"mime_type": "text/plain", "data": "aGVsbG8="}}],
    )

    assert json.loads(str(attrs(only_span(memory))["gen_ai.input.messages"])) == [
        {
            "role": "user",
            "parts": [
                {"text": "hello"},
                {"inline_data": {"mime_type": "text/plain", "data": "aGVsbG8="}},
            ],
        }
    ]


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_structured_output_sets_json_type(memory: SimpleNamespace, async_mode: bool) -> None:
    client, _transport = client_with_transport([happy_payload()])
    wrapped = wrap_google_genai(client)
    config = {
        "responseMimeType": "application/json",
        "responseJsonSchema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    }

    if async_mode:

        async def run() -> Any:
            return await wrapped.aio.models.generate_content(
                model="gemini-2.5-flash", contents=USER_CONTENT, config=config
            )

        pytest.importorskip("asyncio")
        import asyncio

        asyncio.run(run())
    else:
        wrapped.models.generate_content(
            model="gemini-2.5-flash", contents=USER_CONTENT, config=config
        )

    assert attrs(only_span(memory))["gen_ai.output.type"] == "json"


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_function_tools_and_tool_config(memory: SimpleNamespace, async_mode: bool) -> None:
    payload = happy_payload(
        candidates=[
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "Paris"},
                            }
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    )
    client, _transport = client_with_transport([payload])
    wrapped = wrap_google_genai(client)
    config = {
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parametersJsonSchema": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    }
                ]
            }
        ],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
    }

    if async_mode:
        import asyncio

        asyncio.run(
            wrapped.aio.models.generate_content(
                model="gemini-2.5-flash", contents=USER_CONTENT, config=config
            )
        )
    else:
        wrapped.models.generate_content(
            model="gemini-2.5-flash", contents=USER_CONTENT, config=config
        )

    a = attrs(only_span(memory))
    definitions = json.loads(str(a["gen_ai.tool.definitions"]))
    assert definitions == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
    ]
    assert json.loads(str(a["google_genai.request.tool_config"])) == config["toolConfig"]
    output = json.loads(str(a["gen_ai.output.messages"]))
    assert output[0]["parts"][0]["functionCall"]["name"] == "get_weather"


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_multi_candidate_response(memory: SimpleNamespace, async_mode: bool) -> None:
    payload = happy_payload(
        candidates=[
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "first"}]},
                "finishReason": "STOP",
            },
            {
                "index": 1,
                "content": {"role": "model", "parts": [{"text": "second"}]},
                "finishReason": "MAX_TOKENS",
            },
        ]
    )
    client, _transport = client_with_transport([payload])
    wrapped = wrap_google_genai(client)
    config = {"candidateCount": 2}

    if async_mode:
        import asyncio

        asyncio.run(
            wrapped.aio.models.generate_content(
                model="gemini-2.5-flash", contents=USER_CONTENT, config=config
            )
        )
    else:
        wrapped.models.generate_content(
            model="gemini-2.5-flash", contents=USER_CONTENT, config=config
        )

    a = attrs(only_span(memory))
    assert a["gen_ai.request.choice.count"] == 2
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["STOP", "MAX_TOKENS"]
    output = json.loads(str(a["gen_ai.output.messages"]))
    assert len(output) == 2


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_streaming_aggregates_output_usage_and_first_chunk(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    chunks = stream_chunks()
    client, _transport = client_with_transport([chunks])
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        async def run() -> list[Any]:
            stream = await wrapped.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents=USER_CONTENT
            )
            return [chunk async for chunk in stream]

        collected = asyncio.run(run())
    else:
        stream = wrapped.models.generate_content_stream(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
        collected = list(stream)

    assert len(collected) == 3
    assert "".join(chunk.text or "" for chunk in collected) == "Hello world"
    a = attrs(only_span(memory))
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)
    assert a["gen_ai.response.id"] == "stream_123"
    assert a["gen_ai.response.model"] == "gemini-2.5-flash-001"
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello world"}]}
    ]
    assert a["gen_ai.usage.total_tokens"] == 7


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("interrupted", [False, True], ids=["completed", "interrupted"])
@pytest.mark.parametrize(
    "partial_args",
    [
        [{"jsonPath": "$.enabled", "boolValue": False}],
        [{"jsonPath": "$.retries", "numberValue": 0}],
        [
            {"jsonPath": "$.enabled", "boolValue": False},
            {"jsonPath": "$.retries", "numberValue": 0},
        ],
    ],
)
def test_vertex_streaming_counts_partial_function_argument_values(
    memory: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
    interrupted: bool,
    partial_args: list[dict[str, Any]],
) -> None:
    recorded: list[telemetry_dev.SpanHandle] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        assert timestamp_ms is not None
        recorded.append(handle)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    shell = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "set_flags",
                                "partialArgs": [
                                    {"jsonPath": "$.enabled", "willContinue": True},
                                    {"jsonPath": "$.label", "stringValue": ""},
                                ],
                            }
                        }
                    ],
                },
            }
        ]
    }
    partial_values = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "set_flags",
                                "partialArgs": partial_args,
                            }
                        }
                    ],
                },
            }
        ]
    }
    null_value = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "set_flags",
                                "partialArgs": [
                                    {"jsonPath": "$.fallback", "nullValue": "NULL_VALUE"}
                                ],
                            }
                        }
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    }
    payloads: list[dict[str, Any] | RuntimeError] = [shell, partial_values, null_value]
    if interrupted:
        payloads.append(RuntimeError("stream interrupted"))
    client = genai.Client(api_key="test", vertexai=True)
    transport = FakeTransport([payloads])
    client._api_client.request_streamed = transport.request_streamed
    client._api_client.async_request_streamed = transport.async_request_streamed
    wrapped = wrap_google_genai(client)
    chunks: list[Any] = []

    if async_mode:
        import asyncio

        async def run() -> list[Any]:
            stream = await wrapped.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents=USER_CONTENT
            )
            return [chunk async for chunk in stream]

        if interrupted:
            with pytest.raises(RuntimeError, match="stream interrupted"):
                asyncio.run(run())
        else:
            chunks = asyncio.run(run())
    else:
        stream = wrapped.models.generate_content_stream(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
        if interrupted:
            with pytest.raises(RuntimeError, match="stream interrupted"):
                list(stream)
        else:
            chunks = list(stream)

    assert len(recorded) == 2
    if not interrupted:
        decoded_partial_args = chunks[1].candidates[0].content.parts[0].function_call.partial_args
        assert decoded_partial_args == [
            types.PartialArg.model_validate(part) for part in partial_args
        ]
    span = only_span(memory)
    assert attrs(span)["gen_ai.provider.name"] == "gcp.vertex_ai"
    assert span.status.status_code == (StatusCode.ERROR if interrupted else StatusCode.UNSET)


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_streaming_bounds_payload_without_dropping_terminal_metadata(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    source_text = "x" * (100 * 1100)
    source_chunks: list[dict[str, Any]] = [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "x" * 100}]},
                }
            ],
        }
        for _ in range(1100)
    ]
    source_chunks.append(
        {
            "responseId": "bounded-stream",
            "modelVersion": "gemini-2.5-flash-001",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "LOW",
                        }
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            },
            "promptFeedback": {
                "blockReason": "SAFETY",
                "blockReasonMessage": "bounded metadata",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "HIGH",
                    }
                ],
            },
        }
    )
    client, _transport = client_with_transport([source_chunks])
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        async def run() -> tuple[list[Any], Any]:
            stream = await wrapped.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents=USER_CONTENT
            )
            return [chunk async for chunk in stream], stream

        collected, observed_stream = asyncio.run(run())
    else:
        observed_stream = wrapped.models.generate_content_stream(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
        collected = list(observed_stream)

    state = observed_stream._state
    retained_text = "".join(
        part.get("text", "")
        for candidate in state.candidates.values()
        for part in candidate.parts
        if isinstance(part, dict)
    )
    assert len(collected) == len(source_chunks)
    assert state.budget.truncated is True
    assert state.budget.bytes_used <= state.budget.max_bytes
    assert source_text.startswith(retained_text)
    assert len(retained_text) < len(source_text)
    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "bounded-stream"
    assert a["gen_ai.response.model"] == "gemini-2.5-flash-001"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["STOP"]
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 2
    assert a["gen_ai.usage.total_tokens"] == 7
    assert a["google_genai.response.block_reason"] == "SAFETY"
    assert a["google_genai.response.block_reason_message"] == "bounded metadata"
    assert "google_genai.response.safety_ratings" not in a
    assert "google_genai.response.prompt_safety_ratings" not in a
    output = json.loads(str(a["gen_ai.output.messages"]))
    assert output[0]["parts"][0]["text"] == retained_text


def test_streaming_rejects_oversized_grounding_without_dropping_terminal_metadata(
    memory: SimpleNamespace,
) -> None:
    chunks = [
        {
            "responseId": "grounded-stream",
            "modelVersion": "gemini-2.5-flash-001",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                    "groundingMetadata": {"searchEntryPoint": {"renderedContent": "x" * 100_000}},
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            },
        }
    ]
    client, _transport = client_with_transport([chunks])
    wrapped = wrap_google_genai(client)

    stream = wrapped.models.generate_content_stream(model="gemini-2.5-flash", contents=USER_CONTENT)
    list(stream)

    assert stream._state.budget.truncated is True
    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "grounded-stream"
    assert a["gen_ai.response.model"] == "gemini-2.5-flash-001"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["STOP"]
    assert a["gen_ai.usage.input_tokens"] == 5
    assert a["gen_ai.usage.output_tokens"] == 2
    assert a["gen_ai.usage.total_tokens"] == 7
    assert "google_genai.response.grounding_metadata" not in a


def test_streaming_budgets_inline_data_at_native_size(memory: SimpleNamespace) -> None:
    encoded_data = "eHh4" * 16_400
    chunks = [
        {
            "responseId": "inline-data-stream",
            "modelVersion": "gemini-2.5-flash-001",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "application/octet-stream",
                                    "data": encoded_data,
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
        }
    ]
    client, _transport = client_with_transport([chunks])
    wrapped = wrap_google_genai(client)

    stream = wrapped.models.generate_content_stream(model="gemini-2.5-flash", contents=USER_CONTENT)
    list(stream)

    assert len(encoded_data) == 65_600
    assert stream._state.budget.truncated is True
    assert stream._state.candidates[0].parts == []
    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "inline-data-stream"
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["STOP"]
    assert "gen_ai.output.messages" not in a


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_streaming_thought_parts_stay_separate(memory: SimpleNamespace, async_mode: bool) -> None:
    chunks = [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "thought", "thought": True}]},
                }
            ],
            "responseId": "thought_stream",
        },
        {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": " answer"}]}}
            ]
        },
        stream_chunks()[2],
    ]
    client, _transport = client_with_transport([chunks])
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        async def run() -> None:
            stream = await wrapped.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents=USER_CONTENT
            )
            async for _chunk in stream:
                pass

        asyncio.run(run())
    else:
        list(
            wrapped.models.generate_content_stream(model="gemini-2.5-flash", contents=USER_CONTENT)
        )

    output = json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"]))
    assert output == [
        {
            "role": "model",
            "parts": [{"text": "thought", "thought": True}, {"text": " answer"}],
        }
    ]


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_streaming_mid_stream_error(memory: SimpleNamespace, async_mode: bool) -> None:
    class StreamError(RuntimeError):
        pass

    def broken_stream(*args: Any, **kwargs: Any) -> Iterator[types.HttpResponse]:
        yield types.HttpResponse(headers={}, body=json.dumps(stream_chunks()[0]))
        raise StreamError("stream broke")

    async def broken_async_stream(*args: Any, **kwargs: Any) -> AsyncIterator[types.HttpResponse]:
        async def async_generator() -> AsyncIterator[types.HttpResponse]:
            for item in broken_stream(*args, **kwargs):
                yield item

        return async_generator()

    client = genai.Client(api_key="test")
    client._api_client.request_streamed = broken_stream
    client._api_client.async_request_streamed = broken_async_stream
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        async def run() -> None:
            stream = await wrapped.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents=USER_CONTENT
            )
            with pytest.raises(StreamError):
                async for _chunk in stream:
                    pass

        asyncio.run(run())
    else:
        stream = wrapped.models.generate_content_stream(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
        with pytest.raises(StreamError):
            list(stream)

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "StreamError"
    assert json.loads(str(attrs(span)["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello"}]}
    ]


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_streaming_early_break(memory: SimpleNamespace, async_mode: bool) -> None:
    client, _transport = client_with_transport([stream_chunks()])
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        async def run() -> None:
            stream = await wrapped.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents=USER_CONTENT
            )
            async for chunk in stream:
                assert chunk.text == "Hello"
                break

        asyncio.run(run())
    else:
        stream = wrapped.models.generate_content_stream(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
        for chunk in stream:
            assert chunk.text == "Hello"
            break

    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello"}]}
    ]


def test_sync_stream_early_break_closes_inner_iterator(memory: SimpleNamespace) -> None:
    closed = False

    class ClosingIterator:
        def __iter__(self) -> ClosingIterator:
            return self

        def __next__(self) -> Any:
            return types.GenerateContentResponse.model_validate(stream_chunks()[0])

        def close(self) -> None:
            nonlocal closed
            closed = True

    client = genai.Client(api_key="test")
    client.models.generate_content_stream = lambda **_kwargs: ClosingIterator()  # type: ignore[method-assign]
    wrapped = wrap_google_genai(client)
    for chunk in wrapped.models.generate_content_stream(
        model="gemini-2.5-flash", contents=USER_CONTENT
    ):
        assert chunk.text == "Hello"
        break
    assert closed is True
    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello"}]}
    ]


def test_async_stream_early_break_closes_inner_iterator(memory: SimpleNamespace) -> None:
    import asyncio

    closed = False

    class ClosingAsyncIterator:
        def __aiter__(self) -> ClosingAsyncIterator:
            return self

        async def __anext__(self) -> Any:
            return types.GenerateContentResponse.model_validate(stream_chunks()[0])

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    async def stream(**_kwargs: Any) -> ClosingAsyncIterator:
        return ClosingAsyncIterator()

    client = genai.Client(api_key="test")
    client.aio.models.generate_content_stream = stream  # type: ignore[method-assign]
    wrapped = wrap_google_genai(client)

    async def run() -> None:
        async_stream = await wrapped.aio.models.generate_content_stream(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
        async for chunk in async_stream:
            assert chunk.text == "Hello"
            break

    asyncio.run(run())
    assert closed is True
    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello"}]}
    ]


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_api_error_records_error_and_reraises(memory: SimpleNamespace, async_mode: bool) -> None:
    error = errors.APIError(
        400,
        {"error": {"message": "bad", "status": "INVALID_ARGUMENT", "code": 400}},
    )
    client, _transport = client_with_transport([error])
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        with pytest.raises(errors.APIError):
            asyncio.run(
                wrapped.aio.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
            )
    else:
        with pytest.raises(errors.APIError):
            wrapped.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR


def test_blocked_prompt_records_block_attrs_without_output(memory: SimpleNamespace) -> None:
    payload = {"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "blocked"}}
    client, _transport = client_with_transport([payload])
    wrapped = wrap_google_genai(client)
    response = wrapped.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
    assert response.candidates is None
    a = attrs(only_span(memory))
    assert a["google_genai.response.block_reason"] == "SAFETY"
    assert a["google_genai.response.block_reason_message"] == "blocked"
    assert "gen_ai.output.messages" not in a
    assert only_span(memory).status.status_code == StatusCode.UNSET


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_embed_content_maps_embedding_attrs(memory: SimpleNamespace, async_mode: bool) -> None:
    client, _transport = client_with_transport([embed_payload()])
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        result = asyncio.run(
            wrapped.aio.models.embed_content(model="gemini-embedding-001", contents=["embed me"])
        )
    else:
        result = wrapped.models.embed_content(model="gemini-embedding-001", contents=["embed me"])

    assert len(result.embeddings[0].values) == 3
    a = attrs(only_span(memory))
    assert only_span(memory).name == "embeddings gemini-embedding-001"
    assert a["gen_ai.operation.name"] == "embeddings"
    assert a["google_genai.response.embedding_count"] == 1
    assert a["google_genai.response.embedding_dimensions"] == 3
    assert a["gen_ai.usage.input_tokens"] == 6
    assert a["google_genai.usage.billable_characters"] == 12
    assert "gen_ai.output.messages" not in a


def test_chats_emit_spans_with_history(memory: SimpleNamespace) -> None:
    wrapped_before, transport_before = client_with_transport([happy_payload()])
    wrapped_before = wrap_google_genai(wrapped_before)
    chat_before = wrapped_before.chats.create(
        model="gemini-2.5-flash",
        config={"systemInstruction": SYSTEM_INSTRUCTION},
        history=[{"role": "user", "parts": [{"text": "hello"}]}],
    )
    chat_before.send_message("follow up")
    assert len(transport_before.calls) == 1

    client_after, transport_after = client_with_transport([happy_payload()])
    chat_after = client_after.chats.create(
        model="gemini-2.5-flash",
        config={"systemInstruction": SYSTEM_INSTRUCTION},
        history=[{"role": "user", "parts": [{"text": "hello"}]}],
    )
    wrap_google_genai(client_after)
    chat_after.send_message("follow up")
    assert len(transport_after.calls) == 1

    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 2
    expected_input = [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "user", "parts": [{"text": "follow up"}]},
    ]
    for span in spans:
        assert span.name == "chat gemini-2.5-flash"
        assert json.loads(str(attrs(span)["gen_ai.input.messages"])) == expected_input


def test_vertex_provider(memory: SimpleNamespace) -> None:
    _client, _transport = client_with_transport([happy_payload()])
    wrapped = wrap_google_genai(genai.Client(api_key="test", vertexai=True))
    wrapped._api_client.request = _transport.request
    wrapped.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
    assert attrs(only_span(memory))["gen_ai.provider.name"] == "gcp.vertex_ai"


def test_double_wrap_is_idempotent(memory: SimpleNamespace) -> None:
    client, _transport = client_with_transport([happy_payload()])
    wrapped_once = wrap_google_genai(client)
    create_once = wrapped_once.models.generate_content
    wrapped_twice = wrap_google_genai(wrapped_once)
    assert wrapped_twice is client
    assert wrapped_once.models.generate_content is create_once
    wrapped_once.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_fail_open_without_init() -> None:
    telemetry_dev.shutdown()
    uninstrument_google_genai()

    def handler(*args: Any, **kwargs: Any) -> types.HttpResponse:
        return types.HttpResponse(headers={}, body=json.dumps(happy_payload()))

    wrapped_client = wrap_google_genai(genai.Client(api_key="test"))
    wrapped_client._api_client.request = handler
    wrapped_response = wrapped_client.models.generate_content(
        model="gemini-2.5-flash", contents=USER_CONTENT
    )

    instrument_google_genai()
    try:
        instrumented = genai.Client(api_key="test")
        instrumented._api_client.request = handler
        instrumented_response = instrumented.models.generate_content(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
    finally:
        uninstrument_google_genai()
        telemetry_dev.shutdown()

    plain = genai.Client(api_key="test")
    plain._api_client.request = handler
    plain_response = plain.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)

    assert wrapped_response.text == "Telemetry works."
    assert instrumented_response.text == "Telemetry works."
    assert plain_response.text == "Telemetry works."


def test_automatic_function_calling_one_span_and_history(memory: SimpleNamespace) -> None:
    calls: list[dict[str, Any]] = []

    def get_weather(location: str) -> str:
        calls.append({"location": location})
        return f"sunny in {location}"

    get_weather.__annotations__["location"] = str

    queue = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Paris"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "responseId": "afc1",
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
                "toolUsePromptTokenCount": 2,
            },
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "It is sunny in Paris."}]},
                    "finishReason": "STOP",
                }
            ],
            "responseId": "afc2",
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 10,
                "totalTokenCount": 30,
                "toolUsePromptTokenCount": 3,
            },
            "automaticFunctionCallingHistory": [
                {"role": "user", "parts": [{"text": "weather?"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "Paris"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "get_weather",
                                "response": {"result": "sunny in Paris"},
                            }
                        }
                    ],
                },
            ],
        },
    ]
    client, transport = client_with_transport(queue)
    wrapped = wrap_google_genai(client)
    response = wrapped.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "weather?"}]}],
        config=types.GenerateContentConfig(tools=[get_weather]),
    )
    assert len(transport.calls) == 2
    assert calls == [{"location": "Paris"}]
    span = only_span(memory)
    a = attrs(span)
    assert a["google_genai.automatic_function_calling"] is True
    assert json.loads(str(a["gen_ai.input.messages"])) == _native_history(response)
    assert a["gen_ai.usage.input_tokens"] == 30
    assert a["gen_ai.usage.output_tokens"] == 15
    assert a["gen_ai.usage.total_tokens"] == 45
    assert a["google_genai.usage.tool_use_prompt_tokens"] == 5


def test_instrument_and_uninstrument_global(memory: SimpleNamespace) -> None:
    _client, transport = client_with_transport(
        [happy_payload(), happy_payload(), happy_payload(), happy_payload()]
    )
    instrument_google_genai()
    try:
        instrumented = genai.Client(api_key="test")
        instrumented._api_client.request = transport.request
        instrumented._api_client.async_request = transport.async_request
        instrumented.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)

        import asyncio

        asyncio.run(
            instrumented.aio.models.generate_content(
                model="gemini-2.5-flash", contents=USER_CONTENT
            )
        )
        assert len(memory.span_exporter.get_finished_spans()) == 2
    finally:
        uninstrument_google_genai()
    uninstrumented = genai.Client(api_key="test")
    uninstrumented._api_client.request = transport.request
    uninstrumented.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
    uninstrumented._api_client.async_request = transport.async_request
    asyncio.run(
        uninstrumented.aio.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
    )
    assert len(memory.span_exporter.get_finished_spans()) == 2


def test_uninstrument_ends_pending_global_video_span(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from google.genai.models import Models

    def generate_videos(_self: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"name": "operations/pending", "done": False}

    monkeypatch.setattr(Models, "generate_videos", generate_videos)
    instrument_google_genai()
    client = genai.Client(api_key="test")

    result = client.models.generate_videos(model="veo-3.0")
    assert cast(Any, result)["done"] is False
    assert memory.span_exporter.get_finished_spans() == ()

    uninstrument_google_genai()

    span = only_span(memory)
    assert span.status.status_code is StatusCode.ERROR
    event = next(event for event in span.events if event.name == "exception")
    assert "instrumentation removed" in str((event.attributes or {})["exception.message"])


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("poll_fails", [False, True], ids=["terminal", "exception"])
def test_uninstrument_claims_callback_while_video_poll_is_blocked(
    memory: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
    poll_fails: bool,
) -> None:
    callback_calls = 0
    original_end_once = google_genai_instrumentation._end_once

    def counted_end_once(handle: telemetry_dev.SpanHandle) -> Any:
        end = original_end_once(handle)

        def counted_end(**fields: Any) -> None:
            nonlocal callback_calls
            callback_calls += 1
            end(**fields)

        return counted_end

    monkeypatch.setattr(google_genai_instrumentation, "_end_once", counted_end_once)
    failure = RuntimeError("blocked poll failed")

    if async_mode:
        import asyncio

        from google.genai.models import AsyncModels
        from google.genai.operations import AsyncOperations

        async def run() -> None:
            started = asyncio.Event()
            release = asyncio.Event()

            async def generate_videos(_self: Any, **_kwargs: Any) -> dict[str, Any]:
                return {"name": "operations/blocked", "done": False}

            async def get_operation(_self: Any, operation: Any) -> dict[str, Any]:
                started.set()
                await release.wait()
                if poll_fails:
                    raise failure
                return {"name": operation["name"], "done": True}

            monkeypatch.setattr(AsyncModels, "generate_videos", generate_videos)
            monkeypatch.setattr(AsyncOperations, "get", get_operation)
            instrument_google_genai()
            client = genai.Client(api_key="test")
            submitted = await client.aio.models.generate_videos(model="veo-3.0")
            poll = asyncio.create_task(client.aio.operations.get(submitted))
            await started.wait()
            uninstrument_google_genai()
            release.set()
            if poll_fails:
                with pytest.raises(RuntimeError, match="blocked poll failed") as caught:
                    await poll
                assert caught.value is failure
            else:
                assert cast(Any, await poll)["done"] is True

        asyncio.run(run())
    else:
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        from google.genai.models import Models
        from google.genai.operations import Operations

        started = Event()
        release = Event()

        def generate_videos(_self: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"name": "operations/blocked", "done": False}

        def get_operation(_self: Any, operation: Any) -> dict[str, Any]:
            started.set()
            assert release.wait(timeout=5)
            if poll_fails:
                raise failure
            return {"name": operation["name"], "done": True}

        monkeypatch.setattr(Models, "generate_videos", generate_videos)
        monkeypatch.setattr(Operations, "get", get_operation)
        instrument_google_genai()
        client = genai.Client(api_key="test")
        submitted = client.models.generate_videos(model="veo-3.0")
        with ThreadPoolExecutor(max_workers=1) as executor:
            poll = executor.submit(client.operations.get, submitted)
            assert started.wait(timeout=5)
            uninstrument_google_genai()
            release.set()
            if poll_fails:
                with pytest.raises(RuntimeError, match="blocked poll failed") as caught:
                    poll.result(timeout=5)
                assert caught.value is failure
            else:
                assert cast(Any, poll.result(timeout=5))["done"] is True

    assert callback_calls == 1
    span = only_span(memory)
    assert span.status.status_code is StatusCode.ERROR
    assert "google_genai.response.operation_done" not in attrs(span)
    exceptions = [event for event in span.events if event.name == "exception"]
    assert len(exceptions) == 1
    assert "instrumentation removed" in str((exceptions[0].attributes or {})["exception.message"])


def test_uninstrument_ends_global_video_submission_that_finishes_late(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from google.genai.models import AsyncModels

    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def generate_videos(_self: Any, **_kwargs: Any) -> dict[str, Any]:
            started.set()
            await release.wait()
            return {"name": "operations/late", "done": False}

        monkeypatch.setattr(AsyncModels, "generate_videos", generate_videos)
        instrument_google_genai()
        client = genai.Client(api_key="test")
        task = asyncio.create_task(client.aio.models.generate_videos(model="veo-3.0"))
        await started.wait()
        uninstrument_google_genai()
        assert memory.span_exporter.get_finished_spans() == ()
        release.set()
        result = await task
        assert cast(Any, result)["done"] is False

    asyncio.run(run())

    span = only_span(memory)
    assert span.status.status_code is StatusCode.ERROR
    event = next(event for event in span.events if event.name == "exception")
    assert "instrumentation removed" in str((event.attributes or {})["exception.message"])


def test_uninstrument_preserves_later_class_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai.models import Models

    original = Models.generate_content
    instrument_google_genai()

    def later_patch(self: object, *args: Any, **kwargs: Any) -> str:
        return "patched"

    try:
        monkeypatch.setattr(Models, "generate_content", later_patch)
        uninstrument_google_genai()
        assert Models.generate_content is later_patch
    finally:
        Models.generate_content = original  # type: ignore[method-assign]
        uninstrument_google_genai()


def test_wrap_shadows_global_instrumentation_and_survives_uninstrument(
    memory: SimpleNamespace,
) -> None:
    _client, transport = client_with_transport([happy_payload(), happy_payload()])
    instrument_google_genai()
    try:
        wrapped = wrap_google_genai(genai.Client(api_key="test"))
        wrapped._api_client.request = transport.request
        wrapped.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
        assert len(memory.span_exporter.get_finished_spans()) == 1
    finally:
        uninstrument_google_genai()
    wrapped.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
    assert len(memory.span_exporter.get_finished_spans()) == 2


def test_sync_stream_close_ends_partial_span(memory: SimpleNamespace) -> None:
    client, _transport = client_with_transport([stream_chunks()])
    wrapped = wrap_google_genai(client)
    stream = wrapped.models.generate_content_stream(model="gemini-2.5-flash", contents=USER_CONTENT)
    iterator = iter(stream)
    first = next(iterator)
    assert first.text == "Hello"
    stream.close()
    assert json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello"}]}
    ]


def test_sync_stream_supports_next_send_close_protocol(memory: SimpleNamespace) -> None:
    client, _transport = client_with_transport([stream_chunks()])
    wrapped = wrap_google_genai(client)
    stream = wrapped.models.generate_content_stream(model="gemini-2.5-flash", contents=USER_CONTENT)
    assert next(stream).text == "Hello"
    assert next(stream).text == " world"
    next(stream)
    with pytest.raises(StopIteration):
        next(stream)
    span = only_span(memory)
    a = attrs(span)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello world"}]}
    ]
    assert a["gen_ai.usage.total_tokens"] == 7


def test_sync_stream_throw_records_error_and_partial_output(memory: SimpleNamespace) -> None:
    client, _transport = client_with_transport([stream_chunks()])
    wrapped = wrap_google_genai(client)
    stream = wrapped.models.generate_content_stream(model="gemini-2.5-flash", contents=USER_CONTENT)
    assert next(stream).text == "Hello"
    with pytest.raises(RuntimeError):
        stream.throw(RuntimeError("injected"))
    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "RuntimeError"
    assert json.loads(str(attrs(span)["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello"}]}
    ]


def test_async_stream_supports_anext_protocol(memory: SimpleNamespace) -> None:
    import asyncio

    client, _transport = client_with_transport([stream_chunks()])
    wrapped = wrap_google_genai(client)

    async def run() -> None:
        stream = await wrapped.aio.models.generate_content_stream(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
        assert (await anext(stream)).text == "Hello"
        assert (await anext(stream)).text == " world"
        await anext(stream)
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    asyncio.run(run())
    span = only_span(memory)
    assert json.loads(str(attrs(span)["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello world"}]}
    ]


def test_blocked_prompt_records_prompt_safety_ratings(memory: SimpleNamespace) -> None:
    payload = {
        "promptFeedback": {
            "blockReason": "SAFETY",
            "safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "HIGH"}],
        }
    }
    client, _transport = client_with_transport([payload])
    wrapped = wrap_google_genai(client)
    wrapped.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
    a = attrs(only_span(memory))
    assert a["google_genai.response.block_reason"] == "SAFETY"
    ratings = json.loads(str(a["google_genai.response.prompt_safety_ratings"]))
    assert ratings[0]["category"] == "HARM_CATEGORY_HATE_SPEECH"
    assert ratings[0]["probability"] == "HIGH"


def test_mapper_failures_are_fail_open(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import telemetry_dev_google_genai as integration

    def boom(*args: Any) -> Any:
        raise ValueError("mapper broke")

    monkeypatch.setattr(integration, "_generate_request_fields", boom)
    monkeypatch.setattr(integration, "_generate_response_fields", boom)
    client, _transport = client_with_transport([happy_payload()])
    wrapped = wrap_google_genai(client)
    response = wrapped.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
    assert response.text == "Telemetry works."
    span = only_span(memory)
    assert span.name == "chat gemini-2.5-flash"
    assert span.status.status_code == StatusCode.UNSET


def test_tool_definitions_include_new_builtin_markers() -> None:
    from telemetry_dev_google_genai import _tool_definitions

    encoded = _tool_definitions(
        [
            {"google_maps": {}},
            {"retrieval": {}},
            {"enterprise_web_search": {}},
            {"parallel_ai_search": {}},
            {"mcp_servers": [{"url": "http://localhost"}]},
        ]
    )
    assert encoded is not None
    types_seen = {entry["type"] for entry in json.loads(encoded)}
    assert types_seen == {
        "googleMaps",
        "retrieval",
        "enterpriseWebSearch",
        "parallelAiSearch",
        "mcpServers",
    }


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_streaming_afc_accumulates_usage_and_records_history(
    memory: SimpleNamespace, async_mode: bool
) -> None:
    calls: list[dict[str, Any]] = []

    def get_weather(location: str) -> str:
        calls.append({"location": location})
        return f"sunny in {location}"

    get_weather.__annotations__["location"] = str

    turn_one = [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Paris"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "responseId": "afc_s1",
            "modelVersion": "gemini-2.5-flash-001",
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
                "toolUsePromptTokenCount": 2,
            },
        }
    ]
    turn_two = [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "It is sunny in Paris."}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "responseId": "afc_s2",
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 10,
                "totalTokenCount": 30,
                "toolUsePromptTokenCount": 3,
            },
        }
    ]
    client, transport = client_with_transport([turn_one, turn_two])
    wrapped = wrap_google_genai(client)

    if async_mode:
        import asyncio

        async def run() -> list[Any]:
            stream = await wrapped.aio.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=[{"role": "user", "parts": [{"text": "weather?"}]}],
                config=types.GenerateContentConfig(tools=[get_weather]),
            )
            return [chunk async for chunk in stream]

        chunks = asyncio.run(run())
    else:
        stream = wrapped.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": "weather?"}]}],
            config=types.GenerateContentConfig(tools=[get_weather]),
        )
        chunks = list(stream)

    assert calls == [{"location": "Paris"}]
    assert len(transport.calls) == 2
    assert len(chunks) == 1
    span = only_span(memory)
    a = attrs(span)
    assert a["gen_ai.usage.input_tokens"] == 30
    assert a["gen_ai.usage.output_tokens"] == 15
    assert a["gen_ai.usage.total_tokens"] == 45
    assert a["google_genai.usage.tool_use_prompt_tokens"] == 5
    assert a["google_genai.automatic_function_calling"] is True
    history = json.loads(str(a["gen_ai.input.messages"]))
    assert any("functionResponse" in json.dumps(item) for item in history)
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "It is sunny in Paris."}]}
    ]


def test_stream_state_folds_usage_and_output_across_response_ids() -> None:
    from telemetry_dev_google_genai import _record_chunk, _stream_fields, _StreamState

    state = _StreamState()
    _record_chunk(
        {
            "responseId": "r1",
            "candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "call"}]}}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        },
        state,
    )
    _record_chunk(
        {
            "responseId": "r2",
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": "answer"}]}}
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 10,
                "totalTokenCount": 30,
            },
        },
        state,
    )
    fields = _stream_fields(state)
    assert fields["usage"] == {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45}
    assert fields["output"] == [
        {"role": "model", "parts": [{"text": "call"}]},
        {"role": "model", "parts": [{"text": "answer"}]},
    ]


def test_stream_state_drops_over_budget_afc_history_but_still_folds_turn() -> None:
    from telemetry_dev_google_genai import _record_chunk, _stream_fields, _StreamState

    state = _StreamState()
    _record_chunk(
        {
            "responseId": "r1",
            "candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "call"}]}}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        },
        state,
    )
    _record_chunk(
        {
            "responseId": "r1",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "x" * 100_000}]},
                }
            ],
        },
        state,
    )
    _record_chunk(
        {
            "responseId": "r2",
            "modelVersion": "gemini-2.5-flash-001",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "answer"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 10,
                "totalTokenCount": 30,
            },
            "automaticFunctionCallingHistory": [
                {"role": "user", "parts": [{"text": "x" * 100_000}]}
            ],
        },
        state,
    )

    fields = _stream_fields(state)
    assert state.budget.truncated is True
    assert state.afc_history is None
    assert fields["response_id"] == "r2"
    assert fields["response_model"] == "gemini-2.5-flash-001"
    assert fields["finish_reason"] == "STOP"
    assert fields["usage"] == {
        "input_tokens": 30,
        "output_tokens": 15,
        "total_tokens": 45,
    }
    assert fields["output"] == [{"role": "model", "parts": [{"text": "call"}]}]
