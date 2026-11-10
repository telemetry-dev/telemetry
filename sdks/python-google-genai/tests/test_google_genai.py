# Tests monkeypatch google-genai's protected transport seam and inspect dynamic pydantic output.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportPrivateUsage=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import copy
import gc
import json
import os
import signal
import threading
import time
import traceback
import tracemalloc
import weakref
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from io import StringIO
from types import SimpleNamespace
from typing import Any, cast

import conftest
import pytest
import telemetry_dev
from conftest import STREAM_LIMIT
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
MIB_TEXT = "x" * (1024 * 1024)
# Large enough that the 16 MiB bound of a media capture is reached after a few hundred.
REFERENCE_PADDING = "r" * 100_000


def redact_media(_value: Any, context: telemetry_dev.MaskContext) -> str:
    return f"redacted:{context.key}"


def identity_mask(value: Any, _context: Any) -> Any:
    return value


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


def client_with_transport(queue: list[Any], *, vertexai: bool = False) -> tuple[Any, FakeTransport]:
    client = genai.Client(api_key="test", vertexai=vertexai)
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
    # google-genai 2.25 serves Imagen only through Vertex AI.
    client, _transport = client_with_transport(
        [{"predictions": [{"bytesBase64Encoded": "AAAA", "mimeType": "image/png"}]}],
        vertexai=True,
    )
    wrapped = wrap_google_genai(client)

    response = wrapped.models.generate_images(model="imagen-4.0-generate-001", prompt="A graph")

    assert response.generated_images
    a = attrs(only_span(memory))
    assert a["gen_ai.operation.name"] == "generate_content"
    assert a["gen_ai.output.type"] == "image"
    assert a["gen_ai.provider.name"] == "gcp.vertex_ai"
    assert "AAAA" not in json.dumps(a)


@pytest.mark.parametrize("max_attribute_length", [0, 100])
def test_media_input_capture_ignores_a_small_configured_cap(
    make: Any, max_attribute_length: int
) -> None:
    env = make(max_attribute_length=max_attribute_length)
    client, _transport = client_with_transport([])

    def generate_images(**_kwargs: Any) -> dict[str, Any]:
        return {"generatedImages": [{"image": {"gcsUri": "gs://bucket/output.png"}}]}

    client.models.generate_images = generate_images
    wrap_google_genai(client).models.generate_images(
        model="imagen-4.0-generate-001", prompt="p" * 1_000
    )

    a = attrs(only_span(env))
    assert "telemetry.dev.capture.truncated" not in a
    exported = str(a.get("gen_ai.input.messages", ""))
    assert len(exported) <= max_attribute_length
    if max_attribute_length:
        assert exported.endswith("...[truncated]")


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
    # Each reference costs more than its padding, so one past this many cannot fit the bound.
    max_reads = telemetry_dev.MAX_RECORD_BYTES // len(REFERENCE_PADDING) + 1

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
        if self.reads > self.max_reads:
            raise AssertionError("reference image traversal exceeded the capture budget")
        return {"reference_image": {"uri": f"gs://bucket/{REFERENCE_PADDING}-{index}.png"}}


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
    # Each reference costs its padding plus under 100 bytes, so the 168th does not fit.
    assert references.reads == references.max_reads


def test_media_capture_projects_cyclic_labels(memory: SimpleNamespace) -> None:
    client, _transport = client_with_transport([])

    def generate_images(**_kwargs: Any) -> dict[str, Any]:
        return {"generatedImages": []}

    client.models.generate_images = generate_images
    labels: dict[str, Any] = {"ordinary": "value"}
    labels["cycle"] = labels

    wrap_google_genai(client).models.generate_images(model="imagen-4.0", config={"labels": labels})

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.input.messages"])) == {
        "labels": {"ordinary": "value", "cycle": None}
    }
    assert "telemetry.dev.capture.truncated" not in a


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
    assert attrs(span)["error.type"] == "INTERNAL"
    assert dict(span.events[0].attributes or {})["exception.message"] == "generation failed"


SUPPORT_PROMPT = telemetry_dev.ChatPrompt(
    name="support/reply",
    version=12,
    label="production",
    messages=[
        {"role": "system", "content": "You answer for the {{team}} desk."},
        {"role": "user", "content": "Where is order {{order}}?"},
    ],
    model=None,
    parameters={},
    tools=[],
    response_format={"type": "text"},
    is_fallback=False,
)


@pytest.mark.parametrize(
    ("contents", "capture_input", "older_core"),
    [
        ("changed in place", True, False),
        ("changed in place", False, False),
        ("new", True, False),
        ("changed in place", True, True),
    ],
)
def test_generate_content_outside_with_prompt_records_the_prompt_of_compiled_messages(
    make: Any,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
    capture_input: bool,
    older_core: bool,
) -> None:
    memory = make(capture_input=capture_input)
    if older_core:
        monkeypatch.delattr(telemetry_dev, "prompt_attributes_for")
    system, user = SUPPORT_PROMPT.compile(team="billing", order="#4")
    parts = [{"text": user["content"]}]
    # Gemini takes parts: a message changed in place keeps its tag, a new one has none.
    if contents == "new":
        content: dict[str, Any] = {"role": "user", "parts": parts}
    else:
        del user["content"]
        user["parts"] = parts
        content = user
    client, transport = client_with_transport([happy_payload()])

    wrap_google_genai(client).models.generate_content(
        model="gemini-2.5-flash",
        contents=[content],
        config=types.GenerateContentConfig(system_instruction=system["content"]),
    )

    assert cast(Any, transport.calls[0])["contents"] == [
        {"role": "user", "parts": [{"text": "Where is order #4?"}]}
    ]
    a = attrs(only_span(memory))
    assert a["gen_ai.request.model"] == "gemini-2.5-flash"
    assert {key: value for key, value in a.items() if key.startswith("td.prompt.")} == (
        {}
        if older_core or contents == "new"
        else {
            "td.prompt.name": "support/reply",
            "td.prompt.version": 12,
            "td.prompt.label": "production",
            **({"td.prompt.variables": '{"team":"billing","order":"#4"}'} if capture_input else {}),
        }
    )


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
    assert a["telemetry.dev.capture.truncated"] is True


def _instrumentation_reports(reported: list[BaseException]) -> list[BaseException]:
    # The in-memory metric reader reports at the first span with chunk timing that only the
    # built-in OTLP exporter exports the interval metric.
    return [cause for cause in reported if "time_per_output_chunk" not in str(cause)]


def _streamed_text(
    memory: SimpleNamespace, texts: list[str], *, async_mode: bool = False
) -> tuple[Any, dict[str, object]]:
    chunks: list[dict[str, Any]] = [
        {"candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": text}]}}]}
        for text in texts
    ]
    chunks.append(
        {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": []}, "finishReason": "STOP"}
            ]
        }
    )
    client, _transport = client_with_transport([chunks])
    wrapped = wrap_google_genai(client)
    if async_mode:
        import asyncio

        async def run() -> tuple[Any, list[Any]]:
            stream = await wrapped.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents=USER_CONTENT
            )
            return stream, [chunk async for chunk in stream]

        stream, collected = asyncio.run(run())
    else:
        stream = wrapped.models.generate_content_stream(
            model="gemini-2.5-flash", contents=USER_CONTENT
        )
        collected = list(stream)
    assert len(collected) == len(chunks)
    return stream._state, attrs(only_span(memory))


def test_a_history_and_ratings_that_each_chunk_repeats_are_charged_once(
    memory: SimpleNamespace,
) -> None:
    # After the first AFC round, the SDK sets the whole history on each chunk, and each chunk
    # carries the prompt and candidate ratings again; each replaces the last one. Nine copies of
    # the 10,000-byte history, or of either 6,000-byte rating list, pass the 48 KiB stream bound.
    def get_weather(location: str) -> str:
        """Get the weather."""
        return "sunny"

    ratings = SAFETY_RATINGS[:1] * 30
    call = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "P"}}}],
                }
            }
        ]
    }
    answer: list[Any] = [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": f"{index} "}]},
                    "safetyRatings": ratings,
                }
            ],
            "promptFeedback": {"safetyRatings": ratings},
        }
        for index in range(9)
    ]
    answer[-1]["candidates"][0]["finishReason"] = "STOP"
    client, _ = client_with_transport([[call], answer])
    prompt = "h" * 10_000

    list(
        wrap_google_genai(client).models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=types.GenerateContentConfig(tools=[get_weather]),
        )
    )

    a = attrs(only_span(memory))
    assert "telemetry.dev.capture.truncated" not in a
    assert "get_weather" in str(a["gen_ai.input.messages"])
    assert "8 " in str(a["gen_ai.output.messages"])
    assert json.loads(cast(str, a["google_genai.response.safety_ratings"])) == [
        {"candidateIndex": 0, "ratings": ratings}
    ]


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
    def mask(value: Any, context: telemetry_dev.MaskContext) -> Any:
        if not redact or context.key != "gen_ai.output.messages":
            return value
        return [{"role": "model", "parts": [{"text": "[redacted]"}]}]

    memory = make(max_attribute_length=max_attribute_length, mask=mask)
    text = ("東" * 400) if not redact else ("x" * 5_000)

    state, a = _streamed_text(memory, [text])

    assert state.budget.truncated is False
    assert state.candidates[0].parts == [{"text": text}]
    assert "telemetry.dev.capture.truncated" not in a
    if expected is None:
        assert a.get("gen_ai.output.messages", "") == ""
    else:
        output = json.loads(str(a["gen_ai.output.messages"]))
        assert output[0]["parts"][0]["text"] == expected


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("max_attribute_length", [0, 100, 1024 * 1024])
def test_stream_beyond_the_stream_bound_is_flagged_and_withheld_from_a_mask(
    make: Any, masked: bool, max_attribute_length: int
) -> None:
    seen: list[str] = []

    def mask(value: Any, _context: Any) -> Any:
        seen.append(json.dumps(value, default=str))
        return value

    options: dict[str, Any] = {"max_attribute_length": max_attribute_length}
    if masked:
        options["mask"] = mask
    memory = make(**options)

    state, a = _streamed_text(memory, ["x" * 100] * 1_100)

    retained = "".join(part["text"] for part in state.candidates[0].parts)
    assert state.budget.truncated is True
    # Cut at the stream bound, within one 100-byte delta of it.
    assert STREAM_LIMIT - 200 < state.budget.bytes_used <= STREAM_LIMIT
    assert 0 < len(retained) < 1_100 * 100
    assert a["telemetry.dev.capture.truncated"] is True
    if masked:
        assert "gen_ai.output.messages" not in a
        assert not any("x" * 100 in value for value in seen)
    elif max_attribute_length > len(retained):
        output = json.loads(str(a["gen_ai.output.messages"]))
        assert output[0]["parts"][0]["text"] == retained
    else:
        assert len(str(a.get("gen_ai.output.messages", ""))) <= max_attribute_length


@pytest.mark.parametrize(
    ("length", "cut"), [(64 * 1024, False), (telemetry_dev.MAX_RECORD_BYTES, True)]
)
def test_with_the_default_bounds_a_stream_is_cut_only_at_the_record_bound(
    make: Any, length: int, cut: bool
) -> None:
    conftest.other_stream.release()
    reported: list[BaseException] = []
    memory = make(max_stream_bytes=64 * 1024 * 1024, on_error=reported.append)
    texts = [
        MIB_TEXT[: min(len(MIB_TEXT), length - start)] for start in range(0, length, len(MIB_TEXT))
    ]

    _state, a = _streamed_text(memory, texts)

    # A mapping failure would also flag the capture and keep the same 15 parts, so this is the
    # assertion that tells it from the cut at the bound.
    failures = _instrumentation_reports(reported)
    assert not failures, "".join(
        line for cause in failures for line in traceback.format_exception(cause)
    )
    text = json.loads(str(a["gen_ai.output.messages"]))[0]["parts"][0]["text"]
    assert a.get("telemetry.dev.capture.truncated") is (True if cut else None)
    # Each part costs its text plus 52 bytes, so the 16th 1 MiB part does not fit and is
    # rejected whole.
    assert len(text) == (telemetry_dev.MAX_RECORD_BYTES - len(MIB_TEXT) if cut else length)


def test_a_stream_of_small_deltas_holds_about_the_bytes_it_is_charged(make: Any) -> None:
    conftest.other_stream.release()
    memory = make(max_stream_bytes=64 * 1024 * 1024)
    # Reads at which memory is measured, after the memory a stream takes once. No multiple of
    # 1,000, 1,024, 2,000, 2,048, 4,096, 5,000, 8,192 or 10,000 lies between them, so with
    # batches of any of these sizes, or none, the 650 deltas read between them stay separate
    # strings; with batches of 256 they are joined.
    checkpoints = (7_325, 7_975)
    chunk = {
        "candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": "abcd"}]}}]
    }
    client, _transport = client_with_transport([[chunk] * checkpoints[-1]])
    # With automatic function calling, the SDK itself appends each chunk to the request contents
    # it keeps.
    stream = wrap_google_genai(client).models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=USER_CONTENT,
        config={"automatic_function_calling": {"disable": True}},
    )

    held: list[int] = []
    tracemalloc.start()
    try:
        for read, _chunk in enumerate(stream, 1):
            if read in checkpoints:
                # What the stream does not hold, such as decoding garbage in reference cycles,
                # is freed first.
                gc.collect()
                held.append(tracemalloc.get_traced_memory()[0])
    finally:
        tracemalloc.stop()

    # Each delta is decoded into a new string and charged at least its 4 bytes.
    # Before Python 3.12, StringIO keeps each written string apart, about 15 times that, unless
    # the deltas are joined first. Measured between two reads, so that what the stream holds
    # whatever its length does not count. The allowance covers the spare room of a StringIO
    # from Python 3.12 and the memory the core's accounting keeps, and what 256 waiting deltas
    # hold, about 16 KB, but not 650 more.
    charged = 4 * (checkpoints[1] - checkpoints[0])
    assert held[1] - held[0] < 1.3 * charged + 16 * 1024
    output = json.loads(str(attrs(only_span(memory))["gen_ai.output.messages"]))
    assert output[0]["parts"][0]["text"] == "abcd" * checkpoints[-1]


@pytest.mark.parametrize("handler", ["read", "interrupt"])
@pytest.mark.parametrize("step", ["append", "join", "write"])
def test_a_signal_handler_inside_a_delta_buffer_write_keeps_each_delta_once(
    step: str, handler: str
) -> None:
    # A signal handler runs on the thread it interrupts, so one that closes the stream reads
    # the buffer inside a write, and SIGINT raises KeyboardInterrupt there: here right after a
    # delta is appended, right after the waiting deltas are read to be joined onto the tail,
    # and right after text is written to the StringIO.
    buffer = google_genai_instrumentation._DeltaBuffer()
    reads: list[str] = []
    fired = False

    def run_handler() -> None:
        nonlocal fired
        if not fired:
            fired = True
            if handler == "interrupt":
                raise KeyboardInterrupt
            reads.append(buffer.getvalue())

    class Pending(list[str]):
        def append(self, text: str, /) -> None:
            super().append(text)
            if step == "append":
                run_handler()

        def __iter__(self) -> Iterator[str]:
            yield from super().__iter__()
            if step == "join":
                run_handler()

    class Written(StringIO):
        def write(self, text: str, /) -> int:
            count = super().write(text)
            if step == "write":
                run_handler()
            return count

    buffer._buffer = Written()
    # Distinct deltas, so that a delta written twice shows; 10 characters, so that the first
    # batch of 256 waits as the tail and the second is written to the StringIO with it.
    deltas = [f"{index:010d}" for index in range(600)]
    interrupted: list[tuple[int, str]] = []

    def write_all() -> None:
        for index, delta in enumerate(deltas):
            if index == 256:
                # Hooked once the tail holds the first batch, which a join must not lose or
                # leave for a read to write as well.
                buffer._pending = Pending()
            try:
                buffer.write(delta)
            except KeyboardInterrupt:
                interrupted.append((index, buffer.getvalue()))

    # Without a reentrant lock the read waits forever for its own thread.
    writer = threading.Thread(target=write_all, daemon=True)
    writer.start()
    writer.join(10)
    assert not writer.is_alive(), "the read inside the write waited for the lock it holds"

    text = "".join(deltas)
    assert buffer.getvalue() == text
    if handler == "interrupt":
        # The write that the exception came in had taken its delta.
        [(index, value)] = interrupted
        assert value == "".join(deltas[: index + 1])
    else:
        assert reads
        assert all(text.startswith(value) for value in reads)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process:DeprecationWarning")
@pytest.mark.parametrize("read_first", [False, True])
def test_a_delta_buffer_locked_by_another_thread_at_a_fork_works_in_the_child(
    read_first: bool,
) -> None:
    buffer = google_genai_instrumentation._DeltaBuffer()
    buffer.write("ab")
    locked = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with buffer._lock:
            locked.set()
            release.wait()

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert locked.wait(10)
        pid = os.fork()
        if pid == 0:
            # The thread that holds the lock does not run in the child, so it never releases it.
            code = 1
            try:
                if read_first:
                    buffer.getvalue()
                buffer.write("cd")
                code = 0 if buffer.getvalue() == "abcd" else 1
            finally:
                os._exit(code)
    finally:
        release.set()
        holder.join()

    deadline = time.monotonic() + 10
    waited, status = os.waitpid(pid, os.WNOHANG)
    while waited == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
        waited, status = os.waitpid(pid, os.WNOHANG)
    if waited == 0:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    assert waited == pid, "the child waited on the lock it copied held"
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.parametrize("extra_indices", [0, 5_000])
def test_stream_candidate_indices_are_charged_against_the_bound(
    make: Any, extra_indices: int
) -> None:
    seen: list[str] = []

    def mask(value: Any, _context: Any) -> Any:
        seen.append(json.dumps(value, default=str))
        return value

    memory = make(mask=mask)
    chunks: list[dict[str, Any]] = [
        {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": "first"}]}},
                {"index": 1, "content": {"role": "model", "parts": [{"text": "second"}]}},
            ]
        },
        *(
            {
                "candidates": [
                    {
                        "index": index,
                        "content": {"role": "model", "parts": [{"text": "y"}]},
                        "finishReason": "SAFETY",
                    }
                    for index in range(start, start + 100)
                ]
            }
            for start in range(2, 2 + extra_indices, 100)
        ),
        {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": []}, "finishReason": "STOP"},
                {
                    "index": 1,
                    "content": {"role": "model", "parts": []},
                    "finishReason": "MAX_TOKENS",
                },
            ]
        },
    ]
    client, _transport = client_with_transport([chunks])
    stream = wrap_google_genai(client).models.generate_content_stream(
        model="gemini-2.5-flash", contents=USER_CONTENT
    )
    assert len(list(stream)) == len(chunks)

    state = stream._state
    a = attrs(only_span(memory))
    finish_reasons = list(cast(Any, a["gen_ai.response.finish_reasons"]))
    assert finish_reasons[:2] == ["STOP", "MAX_TOKENS"]
    if extra_indices == 0:
        assert finish_reasons == ["STOP", "MAX_TOKENS"]
        assert "telemetry.dev.capture.truncated" not in a
        output = json.loads(str(a["gen_ai.output.messages"]))
        assert [message["parts"][0]["text"] for message in output] == ["first", "second"]
        return
    max_items = google_genai_instrumentation._STREAM_CAPTURE_MAX_ITEMS  # pyright: ignore[reportPrivateUsage]
    assert len(state.candidates) <= max_items
    assert len(state.finish_reasons) <= len(state.candidates)
    assert len(finish_reasons) == len(state.finish_reasons) < 2 + extra_indices
    assert state.budget.truncated is True
    assert a["telemetry.dev.capture.truncated"] is True
    assert "gen_ai.output.messages" not in a
    assert not any('"first"' in value for value in seen)


@pytest.mark.filterwarnings("ignore:.* is not a valid FinishReason:UserWarning")
@pytest.mark.parametrize("spent", [False, True], ids=["within-budget", "budget-spent"])
def test_stream_finish_reasons_are_capped_at_a_fixed_length(make: Any, spent: bool) -> None:
    memory = make()
    max_length = 128
    chunks: list[dict[str, Any]] = [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "x" * (60_000 if spent else 10)}],
                    },
                },
                {"index": 1, "content": {"role": "model", "parts": [{"text": "second"}]}},
            ]
        },
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finishReason": "R" * 10_000,
                },
                {"index": 1, "content": {"role": "model", "parts": []}, "finishReason": "STOP"},
            ]
        },
    ]
    client, _transport = client_with_transport([chunks])
    stream = wrap_google_genai(client).models.generate_content_stream(
        model="gemini-2.5-flash", contents=USER_CONTENT
    )
    assert len(list(stream)) == len(chunks)

    a = attrs(only_span(memory))
    assert list(cast(Any, a["gen_ai.response.finish_reasons"])) == ["R" * max_length, "STOP"]
    assert all(len(reason) <= max_length for reason in stream._state.finish_reasons.values())
    assert (a.get("telemetry.dev.capture.truncated") is True) is spent


def test_masked_stream_closed_early_withholds_partial_output(make: Any) -> None:
    memory = make(mask=identity_mask)
    client, _transport = client_with_transport([stream_chunks()])
    stream = wrap_google_genai(client).models.generate_content_stream(
        model="gemini-2.5-flash", contents=USER_CONTENT
    )
    for _chunk in stream:
        break

    a = attrs(only_span(memory))
    assert "gen_ai.output.messages" not in a
    assert a["telemetry.dev.capture.truncated"] is True


def _streamed_afc(memory: SimpleNamespace, *, async_mode: bool) -> dict[str, object]:
    def get_weather(location: str) -> str:
        """Get the weather."""
        return f"sunny in {location}"

    call = {"functionCall": {"name": "get_weather", "args": {"location": "Paris"}}}
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
    # From google-genai 2.25 the wrapper also sees the function-call turn; distinct response ids
    # keep it a separate message on every version.
    # The final answer comes in several chunks, so a failure that cut the output would show.
    answer: list[dict[str, Any]] = [
        {
            "responseId": "afc-2",
            "candidates": [{"index": 0, "content": {"role": "model", "parts": [{"text": text}]}}],
        }
        for text in ("It is ", "sunny", " in Paris.")
    ]
    answer[-1]["candidates"][0]["finishReason"] = "STOP"
    answer[-1]["usageMetadata"] = usage
    answer[-1]["promptFeedback"] = {"blockReason": "OTHER"}
    turns = [
        [
            {
                "responseId": "afc-1",
                "candidates": [{"index": 0, "content": {"role": "model", "parts": [call]}}],
            }
        ],
        answer,
    ]
    client, _transport = client_with_transport(turns)
    wrapped = wrap_google_genai(client)
    request: dict[str, Any] = {
        "model": "gemini-2.5-flash",
        "contents": USER_CONTENT,
        "config": types.GenerateContentConfig(tools=[get_weather]),
    }
    if async_mode:
        import asyncio

        async def run() -> None:
            stream = await wrapped.aio.models.generate_content_stream(**request)
            async for _chunk in stream:
                pass

        asyncio.run(run())
    else:
        list(wrapped.models.generate_content_stream(**request))
    return attrs(only_span(memory))


def _carries_afc_history(chunk: Any, *_args: Any) -> bool:
    return bool(chunk.automatic_function_calling_history)


def _is_function_call_turn(response: Any, *_args: Any) -> bool:
    return response.response_id == "afc-1"


@pytest.mark.parametrize(
    ("mapper", "failing_call", "masked", "kept", "cut", "history"),
    [
        ("_record_chunk_inner", 2, False, "a", True, None),
        # The last text chunk: a later part would be rejected, which withholds the output on
        # its own.
        ("_record_chunk_inner", 3, True, None, True, None),
        # Inside the parts loop: the part is charged but lost.
        ("_append_part", 2, False, "a", True, None),
        ("_stream_fields_inner", 1, False, None, True, None),
        # Without a mask the metadata is kept whole, as response content, so its loss is
        # flagged; a mask keeps only its structure, whose loss is not.
        ("_record_candidate_metadata", 2, False, "abc", True, None),
        ("_record_candidate_metadata", 2, True, "abc", False, None),
        # Each chunk fails, and the stream reports once.
        ("_record_chunk_usage", None, False, "abc", False, None),
        ("_record_prompt_feedback", None, False, "abc", True, None),
        # The history is the only record of the tool call and its result, so its loss is
        # flagged, also with a mask; it is not output, so the answer is kept whole and not
        # withheld. Only chunks with a history fail.
        ("_record_afc_history", _carries_afc_history, False, "It is sunny in Paris.", True, False),
        ("_record_afc_history", _carries_afc_history, True, "It is sunny in Paris.", True, False),
        ("_generate_usage", None, False, "It is sunny in Paris.", False, True),
        # Only the function-call turn's usage fails to map, where only the AFC usage reads it.
        (
            "_generate_response_fields",
            _is_function_call_turn,
            False,
            "It is sunny in Paris.",
            False,
            True,
        ),
        # The chunk timings fail for each chunk, and the first-chunk update fails; neither is
        # output. The span's end calls update too, so only the first call fails.
        ("_chunk_has_output", None, False, "abc", False, None),
        ("SpanHandle.update", 1, False, "abc", False, None),
    ],
    ids=[
        "chunk",
        "masked-chunk",
        "part",
        "end",
        "candidate-metadata",
        "masked-candidate-metadata",
        "usage",
        "prompt-feedback",
        "afc-history",
        "masked-afc-history",
        "afc-usage",
        "afc-turn-usage",
        "chunk-timing",
        "first-chunk",
    ],
)
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_a_stream_mapping_failure_is_reported_and_flags_only_lost_output(
    make: Any,
    monkeypatch: pytest.MonkeyPatch,
    mapper: str,
    failing_call: int | Callable[..., bool] | None,
    masked: bool,
    kept: str | None,
    cut: bool,
    history: bool | None,
    async_mode: bool,
) -> None:
    owner: Any = google_genai_instrumentation
    if mapper == "SpanHandle.update":
        owner, mapper = telemetry_dev.SpanHandle, "update"
    original = getattr(owner, mapper)
    calls = 0

    def failing(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if (
            failing_call is None
            or calls == failing_call
            or (callable(failing_call) and failing_call(*args))
        ):
            raise ValueError("mapper broke")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, mapper, failing)
    reported: list[BaseException] = []
    options: dict[str, Any] = {"on_error": reported.append}
    if masked:
        options["mask"] = identity_mask
    memory = make(**options)

    if history is None:
        _state, a = _streamed_text(memory, ["a", "b", "c"], async_mode=async_mode)
    else:
        a = _streamed_afc(memory, async_mode=async_mode)

    assert [str(cause) for cause in _instrumentation_reports(reported)] == ["mapper broke"]
    assert a.get("telemetry.dev.capture.truncated") is (True if cut else None)
    if kept is None:
        assert "gen_ai.output.messages" not in a
    else:
        # The part after the lost one is not kept, so the output is a prefix of the stream.
        assert json.loads(str(a["gen_ai.output.messages"]))[-1]["parts"] == [{"text": kept}]
    if history is not None:
        # Each metadata step has its own guard, so a failed one skips none of the others.
        assert a["google_genai.response.block_reason"] == "OTHER"
        assert ("google_genai.automatic_function_calling" in a) is history
        messages = json.dumps(json.loads(str(a["gen_ai.input.messages"])))
        assert ("functionResponse" in messages) is history


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


# Every content string carries "private" or the answer's "Grounded" so a leak is easy to spot.
GROUNDING_METADATA: dict[str, Any] = {
    "webSearchQueries": ["private query"],
    "searchEntryPoint": {"renderedContent": "<div>private query</div>"},
    "groundingChunks": [
        {
            "web": {
                "uri": "https://private.example/page",
                "title": "private page",
                "domain": "private.example",
            }
        },
        {
            "retrievedContext": {
                "uri": "gs://private/doc.pdf",
                "title": "private doc",
                "text": "private retrieved text",
                "ragChunk": {
                    "text": "private rag text",
                    "pageSpan": {"firstPage": 1, "lastPage": 2},
                },
            }
        },
    ],
    "groundingSupports": [
        {
            "segment": {"startIndex": 0, "endIndex": 15, "partIndex": 0, "text": "Grounded answer"},
            "groundingChunkIndices": [1, 0],
            "confidenceScores": [0.9, 0.4],
        }
    ],
    "retrievalMetadata": {"googleSearchDynamicRetrievalScore": 0.7},
}
URL_CONTEXT_METADATA: dict[str, Any] = {
    "urlMetadata": [
        {
            "retrievedUrl": "https://private.example/url",
            "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS",
        },
        {"retrievedUrl": "https://private.example/other", "urlRetrievalStatus": "private status"},
    ]
}


SAFETY_RATINGS: list[dict[str, Any]] = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "probability": "NEGLIGIBLE",
        "probabilityScore": 0.1,
        "severity": "HARM_SEVERITY_LOW",
        "severityScore": 0.2,
        "blocked": False,
        "overwrittenThreshold": "BLOCK_ONLY_HIGH",
    },
    {"category": "private category", "probability": "LOW", "severity": "private severity"},
    {"category": "HARM_CATEGORY_NOT_YET_DEFINED"},
]
SAFETY_RATINGS_STRUCTURE = [SAFETY_RATINGS[0], {"probability": "LOW"}, SAFETY_RATINGS[2]]


def redact_grounded(value: Any, _context: Any) -> Any:
    return json.loads(json.dumps(value).replace("Grounded", "REDACTED"))


def capture_options(capture: str) -> dict[str, Any]:
    return {
        "off": {"capture_output": False},
        "masked": {"mask": redact_grounded},
        "on": {},
    }[capture]


async def generate(response: Any, *, is_async: bool, stream: bool) -> None:
    client, _transport = client_with_transport([response])
    wrapped = wrap_google_genai(client)
    request = {"model": "gemini-2.5-flash", "contents": "go"}
    if is_async and stream:
        async for _chunk in await wrapped.aio.models.generate_content_stream(**request):
            pass
    elif is_async:
        await wrapped.aio.models.generate_content(**request)
    elif stream:
        list(wrapped.models.generate_content_stream(**request))
    else:
        wrapped.models.generate_content(**request)


@pytest.mark.filterwarnings("ignore:.* is not a valid")
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("capture", ["off", "masked", "on"])
async def test_grounding_content_is_kept_only_when_output_is_captured_unmasked(
    make: Any, is_async: bool, stream: bool, capture: str
) -> None:
    memory = make(**capture_options(capture))
    content = {"role": "model", "parts": [{"text": "Grounded answer."}]}
    metadata = {
        "finishReason": "STOP",
        "safetyRatings": SAFETY_RATINGS,
        "groundingMetadata": GROUNDING_METADATA,
        "urlContextMetadata": URL_CONTEXT_METADATA,
    }
    response: Any = (
        [
            {"candidates": [{"index": 0, "content": content}]},
            {"candidates": [{"index": 0, **metadata}]},
        ]
        if stream
        else {"candidates": [{"index": 0, "content": content, **metadata}]}
    )
    await generate(response, is_async=is_async, stream=stream)

    a = attrs(only_span(memory))
    grounding = json.loads(cast(str, a["google_genai.response.grounding_metadata"]))
    url_context = json.loads(cast(str, a["google_genai.response.url_context_metadata"]))
    safety_ratings = json.loads(cast(str, a["google_genai.response.safety_ratings"]))
    if capture == "on":
        assert grounding == GROUNDING_METADATA
        assert url_context == URL_CONTEXT_METADATA
        assert safety_ratings == [{"candidateIndex": 0, "ratings": SAFETY_RATINGS}]
        return

    assert "private" not in json.dumps(a)
    assert "Grounded" not in json.dumps(a)
    assert grounding == {
        "groundingChunks": [{"web": {}}, {"retrievedContext": {}}],
        "groundingSupports": [
            {
                "segment": {"startIndex": 0, "endIndex": 15, "partIndex": 0},
                "groundingChunkIndices": [1, 0],
                "confidenceScores": [0.9, 0.4],
            }
        ],
        "retrievalMetadata": {"googleSearchDynamicRetrievalScore": 0.7},
    }
    assert url_context == {
        "urlMetadata": [{"urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"}, {}]
    }
    assert safety_ratings == [{"candidateIndex": 0, "ratings": SAFETY_RATINGS_STRUCTURE}]
    if capture == "masked":
        assert json.loads(cast(str, a["gen_ai.output.messages"])) == [
            {"role": "model", "parts": [{"text": "REDACTED answer."}]}
        ]


@pytest.mark.filterwarnings("ignore:.* is not a valid")
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("capture", ["off", "masked", "on"])
async def test_prompt_feedback_text_is_kept_only_when_output_is_captured_unmasked(
    make: Any, is_async: bool, stream: bool, capture: str
) -> None:
    memory = make(**capture_options(capture))
    response = {
        "promptFeedback": {
            "blockReason": "SAFETY",
            "blockReasonMessage": "private block message",
            "safetyRatings": SAFETY_RATINGS,
        }
    }
    await generate([response] if stream else response, is_async=is_async, stream=stream)

    span = only_span(memory)
    a = attrs(span)
    assert a["google_genai.response.block_reason"] == "SAFETY"
    assert "gen_ai.output.messages" not in a
    assert span.status.status_code == StatusCode.UNSET
    prompt_ratings = json.loads(cast(str, a["google_genai.response.prompt_safety_ratings"]))
    if capture == "on":
        assert a["google_genai.response.block_reason_message"] == "private block message"
        assert prompt_ratings == SAFETY_RATINGS
        return

    assert "private" not in json.dumps(a)
    assert "google_genai.response.block_reason_message" not in a
    assert prompt_ratings == SAFETY_RATINGS_STRUCTURE


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

    a = attrs(only_span(memory))
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "model", "parts": [{"text": "Hello"}]}
    ]
    assert a["telemetry.dev.capture.truncated"] is True


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
    error = errors.ClientError(
        429,
        {"error": {"message": "slow down", "status": "RESOURCE_EXHAUSTED", "code": 429}},
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
    # ClientError names no cause, so the HTTP status stands in for it.
    assert attrs(span)["error.type"] == "429"


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


def test_global_instrumentation_reports_vertex_provider(memory: SimpleNamespace) -> None:
    instrument_google_genai()
    client, _transport = client_with_transport([happy_payload()], vertexai=True)
    client.models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT)
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
def test_uninstrument_ends_video_span_while_poll_is_blocked(
    memory: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
    poll_fails: bool,
) -> None:
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


def test_sync_stream_supports_next_and_send_protocol(memory: SimpleNamespace) -> None:
    client, _transport = client_with_transport([stream_chunks()])
    wrapped = wrap_google_genai(client)
    stream = wrapped.models.generate_content_stream(model="gemini-2.5-flash", contents=USER_CONTENT)
    assert next(stream).text == "Hello"
    assert stream.send(None).text == " world"
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


@pytest.mark.parametrize(
    ("operation", "mapper"),
    [
        ("generate", "_generate_response_fields"),
        # Each AFC turn and the response fail to map, and the call reports once.
        ("generate-afc", "_generate_response_fields"),
        ("embed", "_embed_response_fields"),
        ("video-done", "_media_response_fields"),
        # A polled video operation ends its span where the handle is not kept.
        ("video-polled", "_media_response_fields"),
    ],
)
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_mapper_failures_are_fail_open_reported_and_flagged(
    make: Any, monkeypatch: pytest.MonkeyPatch, operation: str, mapper: str, async_mode: bool
) -> None:
    import asyncio

    import telemetry_dev_google_genai as integration

    def boom(*args: Any) -> Any:
        raise ValueError("mapper broke")

    monkeypatch.setattr(integration, "_generate_request_fields", boom)
    monkeypatch.setattr(integration, mapper, boom)
    reported: list[BaseException] = []
    memory = make(on_error=reported.append)
    video = {"name": "operations/video-1", "done": operation == "video-done"}
    payloads: list[Any] = [
        {"generate": happy_payload(), "embed": embed_payload()}.get(operation, video)
    ]
    if operation == "generate-afc":
        call_part = {"functionCall": {"name": "get_weather", "args": {"location": "Paris"}}}
        payloads = [
            {"candidates": [{"content": {"role": "model", "parts": [call_part]}}]},
            happy_payload(),
        ]
    client, _transport = client_with_transport(payloads)
    completed = types.GenerateVideosOperation.model_validate({**video, "done": True})

    def get_operation(operation: Any) -> Any:
        return completed

    async def async_get(operation: Any) -> Any:
        return completed

    client.operations.get = get_operation
    client.aio.operations.get = async_get
    wrapped = wrap_google_genai(client)
    models = wrapped.aio.models if async_mode else wrapped.models
    operations = wrapped.aio.operations if async_mode else wrapped.operations

    def call(result: Any) -> Any:
        return asyncio.run(result) if async_mode else result

    def get_weather(location: str) -> str:
        """Get the weather."""
        return f"sunny in {location}"

    if operation in {"generate", "generate-afc"}:
        config = (
            types.GenerateContentConfig(tools=[get_weather])
            if operation == "generate-afc"
            else None
        )
        response = call(
            models.generate_content(model="gemini-2.5-flash", contents=USER_CONTENT, config=config)
        )
        assert response.text == "Telemetry works."
    elif operation == "embed":
        call(models.embed_content(model="gemini-embedding-001", contents=["embed me"]))
    else:
        submitted = call(models.generate_videos(model="veo-3.0", prompt="a graph"))
        if operation == "video-polled":
            call(operations.get(submitted))

    assert [str(cause) for cause in _instrumentation_reports(reported)] == ["mapper broke"]
    span = only_span(memory)
    if operation in {"generate", "generate-afc"}:
        assert span.name == "chat gemini-2.5-flash"
    assert span.status.status_code == StatusCode.UNSET
    # The call succeeded, but its output and response fields are lost.
    assert attrs(span)["telemetry.dev.capture.truncated"] is True


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
    # The chunk sequence depends on the google-genai version (2.25 also yields the
    # function-call turn), so compare against the same stream without instrumentation.
    plain_client, _plain_transport = client_with_transport(copy.deepcopy([turn_one, turn_two]))
    plain_chunks = list(
        plain_client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=[{"role": "user", "parts": [{"text": "weather?"}]}],
            config=types.GenerateContentConfig(tools=[get_weather]),
        )
    )
    assert [chunk.model_dump(mode="json") for chunk in chunks] == [
        chunk.model_dump(mode="json") for chunk in plain_chunks
    ]
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


def test_stream_state_drops_over_budget_afc_history_but_still_folds_turn(
    memory: SimpleNamespace,
) -> None:
    # The client sizes the shared stream bound, which leaves this stream STREAM_LIMIT.
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


def _lock_and_take(name: str) -> tuple[Callable[[], Any], Callable[[], object]]:
    """A lock that a thread can hold at a fork: how to get it, and a call that takes it."""
    if name == "install":
        return lambda: google_genai_instrumentation._install_lock, uninstrument_google_genai
    tracker = google_genai_instrumentation._VideoOperationTracker()
    return tracker._lock, lambda: tracker.take(object(), "operations/1")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("name", ["install", "video operations"])
def test_a_forked_child_does_not_wait_for_a_lock_that_a_thread_of_the_parent_held(
    name: str,
) -> None:
    lock_of, take = _lock_and_take(name)
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with lock_of():
            held.set()
            release.wait()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    try:
        assert held.wait(10)
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                # The holder thread does not exist here, so this must not wait for it.
                take()
                code = 0
            finally:
                # The child must not return into pytest.
                os._exit(code)
    finally:
        release.set()
    holder.join(10)
    # A child that waits for the lock fails the test instead of hanging the suite.
    deadline = time.monotonic() + 10
    while (waited := os.waitpid(pid, os.WNOHANG)) == (0, 0):
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("The forked process did not end in 10 s")
        time.sleep(0.01)
    assert os.waitstatus_to_exitcode(waited[1]) == 0


def test_a_wrapped_client_registers_no_fork_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    # os.register_at_fork keeps each hook for good, so a hook for each wrapped client would grow
    # without bound in an app that wraps a client for each request.
    hooks: list[dict[str, object]] = []

    def register_at_fork(**hook: object) -> None:
        hooks.append(hook)

    monkeypatch.setattr(os, "register_at_fork", register_at_fork)
    for _ in range(3):
        client, _transport = client_with_transport([happy_payload()])
        wrap_google_genai(client)
    assert hooks == []
