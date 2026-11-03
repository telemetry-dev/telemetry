from __future__ import annotations

import copy
import dataclasses
import gc
import io
import json
import os
import signal
import threading
import time
import traceback
import tracemalloc
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import boto3
import conftest
import pytest
import telemetry_dev
from botocore.client import BaseClient
from botocore.exceptions import ClientError, EventStreamError
from botocore.response import StreamingBody
from botocore.stub import Stubber
from conftest import STREAM_LIMIT
from opentelemetry.trace import StatusCode

import telemetry_dev_bedrock._instrument as bedrock_instrument
import telemetry_dev_bedrock._streams as bedrock_streams
from telemetry_dev_bedrock import instrument_bedrock, uninstrument_bedrock, wrap_bedrock
from telemetry_dev_bedrock._registry import SERVICE_OPERATIONS

# A JSON string of this many characters alone passes the bound of a non-stream capture.
OVER_RECORD = "x" * telemetry_dev.MAX_RECORD_BYTES
MIB_TEXT = "x" * (1024 * 1024)


def _client(service: str) -> BaseClient:
    return cast(
        BaseClient,
        boto3.client(
            service,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        ),
    )


def _stub_api_call(
    client: BaseClient, responses: list[Any], monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake(operation_name: str, api_params: dict[str, Any]) -> Any:
        calls.append((operation_name, copy.deepcopy(api_params)))
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(client, "_make_api_call", fake)
    return calls


class FakeEventStream:
    def __init__(self, events: list[Any], error_at: int | None = None) -> None:
        self.events = events
        self.error_at = error_at
        self.closed = False

    def __iter__(self) -> Any:
        for index, event in enumerate(self.events):
            if self.error_at == index:
                raise EventStreamError(
                    error_response={
                        "Error": {"Code": "modelStreamErrorException", "Message": "boom"},
                        "ResponseMetadata": {
                            "RequestId": "stream-error-1",
                            "HTTPStatusCode": 424,
                            "RetryAttempts": 1,
                        },
                    },
                    operation_name="ConverseStream",
                )
            yield event
        if self.error_at == len(self.events):
            raise EventStreamError(
                error_response={
                    "Error": {"Code": "modelStreamErrorException", "Message": "boom"},
                    "ResponseMetadata": {
                        "RequestId": "stream-error-2",
                        "HTTPStatusCode": 424,
                        "RetryAttempts": 1,
                    },
                },
                operation_name="ConverseStream",
            )

    def close(self) -> None:
        self.closed = True


class FailingStreamingBody(StreamingBody):
    def read(self, *args: Any, **kwargs: Any) -> bytes:
        raise OSError("read failed")


class UnrewindableBody:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.reads = 0

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        self.reads += 1
        return self.raw


def identity_mask(value: Any, _context: Any) -> Any:
    return value


def _body(value: Any) -> bytes:
    return json.dumps(value).encode("utf-8")


def _json_attr(span: Any, key: str) -> Any:
    value = span.attributes[key]
    assert isinstance(value, str)
    return json.loads(value)


def _instrumentation_errors(errors: list[BaseException]) -> list[BaseException]:
    # The in-memory metric reader reports at the first span with chunk timing that only the
    # built-in OTLP exporter exports the interval metric.
    return [error for error in errors if "time_per_output_chunk" not in str(error)]


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
    ("messages", "older_core"),
    [("changed in place", False), ("new", False), ("changed in place", True)],
)
def test_converse_outside_with_prompt_records_the_prompt_of_compiled_messages(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, messages: str, older_core: bool
) -> None:
    if older_core:
        monkeypatch.delattr(telemetry_dev, "prompt_attributes_for")
    system, user = SUPPORT_PROMPT.compile(team="billing", order="#4")
    blocks = [{"text": user["content"]}]
    # Converse takes content blocks: a message changed in place keeps its tag, a new one has none.
    if messages == "new":
        message = {"role": "user", "content": blocks}
    else:
        user["content"] = blocks
        message = user
    client = _client("bedrock-runtime")
    # The stubber runs botocore's validation of the request.
    stubber = Stubber(client)
    stubber.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "Soon."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        },
    )

    with stubber:
        cast(Any, wrap_bedrock(client)).converse(
            modelId="anthropic.claude-3-haiku",
            system=[{"text": system["content"]}],
            messages=[message],
        )

    [span] = memory.span_exporter.get_finished_spans()
    a = dict(span.attributes or {})
    assert a["gen_ai.request.model"] == "anthropic.claude-3-haiku"
    assert {key: value for key, value in a.items() if key.startswith("td.prompt.")} == (
        {}
        if older_core or messages == "new"
        else {
            "td.prompt.name": "support/reply",
            "td.prompt.version": 12,
            "td.prompt.label": "production",
            "td.prompt.variables": '{"team":"billing","order":"#4"}',
        }
    )


def test_converse_happy_path(memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client("bedrock-runtime")
    request = {
        "modelId": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "hello"},
                    {"image": {"format": "png", "source": {"bytes": b"123"}}},
                    {
                        "document": {
                            "format": "txt",
                            "source": {"s3Location": {"uri": "s3://bucket/key"}},
                        }
                    },
                ],
            }
        ],
        "system": [{"text": "be terse"}],
        "inferenceConfig": {"temperature": 0.1, "topP": 0.9, "maxTokens": 10},
    }
    request_snapshot = copy.deepcopy(request)
    calls = _stub_api_call(
        client,
        [
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 3,
                    "outputTokens": 4,
                    "totalTokens": 7,
                    "cacheReadInputTokens": 1,
                    "cacheWriteInputTokens": 2,
                },
                "metrics": {"latencyMs": 12},
                "trace": {"promptRouter": {"invokedModelId": "routed"}},
                "ResponseMetadata": {
                    "RequestId": "req-1",
                    "RetryAttempts": 1,
                    "HTTPStatusCode": 200,
                    "TotalRetryDelay": 8,
                },
            }
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    client.converse(**request)  # type: ignore[attr-defined]

    assert request == request_snapshot
    assert calls == [("Converse", request_snapshot)]
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.name == "chat anthropic.claude-3-5-haiku-20241022-v1:0"
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.provider.name"] == "amazon-bedrock"
    assert span.attributes["gen_ai.response.id"] == "req-1"
    assert span.attributes["aws.request.attempts"] == 2
    assert span.attributes["aws.http.status_code"] == 200
    assert span.attributes["aws.request.total_retry_delay_ms"] == 8
    assert span.attributes["gen_ai.usage.input_tokens"] == 3 + 1 + 2
    assert span.attributes["gen_ai.usage.total_tokens"] == 3 + 1 + 2 + 4
    assert span.attributes["gen_ai.usage.cache_creation.input_tokens"] == 2
    assert span.attributes["td.metadata.server_latency_ms"] == "12"
    assert _json_attr(span, "gen_ai.input.messages") == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "hello"},
                {"type": "blob", "modality": "image", "mime_type": "image/png"},
                {
                    "type": "uri",
                    "uri": "s3://bucket/key",
                    "modality": "document",
                    "mime_type": "txt",
                },
            ],
        }
    ]
    assert _json_attr(span, "gen_ai.output.messages") == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "hi"}],
            "finish_reason": "end_turn",
        }
    ]


def test_rerank_captures_query_sources_model_and_results(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-agent-runtime")
    model_arn = "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0"
    request = {
        "queries": [{"type": "TEXT", "textQuery": {"text": "best observability platform"}}],
        "sources": [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": "Telemetry data"},
                },
            },
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "JSON",
                    "jsonDocument": {"title": "Other result"},
                },
            },
        ],
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "numberOfResults": 2,
                "modelConfiguration": {
                    "modelArn": model_arn,
                    "additionalModelRequestFields": {"max_tokens_per_doc": 256},
                },
            },
        },
    }
    request_snapshot = copy.deepcopy(request)
    results = [
        {
            "index": 0,
            "relevanceScore": 0.91,
            "document": {"type": "TEXT", "textDocument": {"text": "Telemetry data"}},
        },
        {"index": 1, "relevanceScore": 0.12},
    ]
    calls = _stub_api_call(
        client,
        [
            {
                "results": results,
                "nextToken": "page-2",
                "ResponseMetadata": {"RequestId": "rerank-1", "HTTPStatusCode": 200},
            }
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    client.rerank(**request)  # type: ignore[attr-defined]

    assert request == request_snapshot
    assert calls == [("Rerank", request_snapshot)]
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.name == "rerank cohere.rerank-v3-5:0"
    assert span.attributes["gen_ai.operation.name"] == "rerank"
    assert span.attributes["gen_ai.provider.name"] == "amazon-bedrock"
    assert span.attributes["gen_ai.request.model"] == model_arn
    assert span.attributes["gen_ai.response.id"] == "rerank-1"
    assert span.attributes["td.metadata.requested_result_count"] == "2"
    assert span.attributes["td.metadata.result_count"] == "2"
    assert span.attributes["td.metadata.has_next_token"] == "true"
    assert _json_attr(span, "gen_ai.input.messages") == {
        "queries": request["queries"],
        "sources": request["sources"],
        "additionalModelRequestFields": {"max_tokens_per_doc": 256},
    }
    assert _json_attr(span, "gen_ai.output.messages") == results


@pytest.mark.parametrize("max_attribute_length", [0, 100])
def test_rerank_capture_ignores_a_small_configured_cap(
    make: Any, monkeypatch: pytest.MonkeyPatch, max_attribute_length: int
) -> None:
    memory = make(max_attribute_length=max_attribute_length)
    client = _client("bedrock-agent-runtime")
    results = [{"index": 0, "relevanceScore": 0.5, "document": {"text": "r" * 1_000}}]
    _stub_api_call(client, [{"results": results}], monkeypatch)
    wrap_bedrock(client)

    client.rerank(  # type: ignore[attr-defined]
        queries=[{"type": "TEXT", "textQuery": {"text": "q" * 1_000}}],
        sources=[],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": "cohere.rerank-v3-5:0"}
            },
        },
    )

    span = memory.span_exporter.get_finished_spans()[0]
    assert "telemetry.dev.capture.truncated" not in span.attributes
    for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        exported = str(span.attributes.get(key, ""))
        assert len(exported) <= max_attribute_length
        if max_attribute_length:
            assert exported.endswith("...[truncated]")


def test_rerank_bounds_large_inputs_and_outputs_without_changing_request(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-agent-runtime")
    shared_source = {
        "type": "INLINE",
        "inlineDocumentSource": {
            "type": "TEXT",
            "textDocument": {"text": "x" * 10_000},
        },
    }
    # 2,000 sources of 10,000 bytes pass the 16 MiB bound of a non-stream capture.
    sources = [shared_source] * 2_000
    request = {
        "queries": [{"type": "TEXT", "textQuery": {"text": "best"}}],
        "sources": sources,
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": "cohere.rerank-v3-5:0"}
            },
        },
    }
    document = {"text": "x" * 10_000}
    results = [
        {"index": index, "relevanceScore": 0.5, "document": document} for index in range(2_000)
    ]
    calls = _stub_api_call(client, [{"results": results}], monkeypatch)
    wrap_bedrock(client)

    client.rerank(**request)  # type: ignore[attr-defined]

    assert calls[0][1]["sources"] == sources
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["telemetry.dev.capture.truncated"] is True
    assert span.attributes["td.metadata.result_count"] == "2000"
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes


def test_converse_stream_accumulates_and_errors(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
    _stub_api_call(
        client,
        [
            {
                "stream": FakeEventStream(
                    [
                        {"messageStart": {"role": "assistant"}},
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 0,
                                "delta": {"text": "Hello "},
                            }
                        },
                        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "world"}}},
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 0,
                                "delta": {"citation": {"title": "doc", "source": "kb"}},
                            }
                        },
                        {"messageStop": {"stopReason": "end_turn"}},
                        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
                    ]
                ),
                "ResponseMetadata": {"RequestId": "stream-1"},
            },
            {
                "stream": FakeEventStream(
                    [
                        {"messageStart": {"role": "assistant"}},
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 0,
                                "delta": {"text": "partial"},
                            }
                        },
                    ],
                    error_at=2,
                )
            },
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.converse_stream(modelId="m", messages=[])  # type: ignore[attr-defined]
    list(response["stream"])
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.response.id"] == "stream-1"
    assert span.attributes["gen_ai.response.time_to_first_chunk"] >= 0
    assert _json_attr(span, "gen_ai.output.messages")[0]["parts"][0] == {
        "type": "text",
        "content": "Hello world",
        "citations": [{"title": "doc", "source": "kb"}],
    }

    response = client.converse_stream(modelId="m", messages=[])  # type: ignore[attr-defined]
    with pytest.raises(EventStreamError):
        list(response["stream"])
    error_span = memory.span_exporter.get_finished_spans()[1]
    assert error_span.attributes["error.type"] == "EventStreamError"
    assert error_span.attributes["aws.error.code"] == "modelStreamErrorException"
    assert error_span.attributes["gen_ai.response.id"] == "stream-error-2"
    assert error_span.attributes["aws.http.status_code"] == 424
    assert error_span.attributes["aws.request.attempts"] == 2
    assert _json_attr(error_span, "gen_ai.output.messages")[0]["parts"][0]["content"] == "partial"
    assert error_span.attributes["telemetry.dev.capture.truncated"] is True


def test_converse_stream_bounds_retained_state_without_dropping_events(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        {"messageStart": {"role": "assistant"}},
        *[
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "x" * 100},
                }
            }
            for _ in range(1100)
        ],
        {"messageStop": {"stopReason": "max_tokens"}},
        {"metadata": {"usage": {"inputTokens": 7, "outputTokens": 11}}},
    ]
    client = _client("bedrock-runtime")
    _stub_api_call(client, [{"stream": FakeEventStream(events)}], monkeypatch)
    wrap_bedrock(client)

    response = client.converse_stream(modelId="m", messages=[])  # type: ignore[attr-defined]
    instrumented_stream = response["stream"]
    delivered = list(instrumented_stream)
    state = cast(Any, instrumented_stream)._state
    budget = state.budget

    assert delivered == events
    assert budget.truncated is True
    assert budget.bytes_used <= budget.max_bytes
    assert len(state.blocks[0]["text"]) < STREAM_LIMIT
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 11
    assert list(span.attributes["gen_ai.response.finish_reasons"]) == ["max_tokens"]
    assert span.attributes["telemetry.dev.capture.truncated"] is True


def _converse_text_stream(text: str, *, repeat: int = 1) -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        *[
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}
            for _ in range(repeat)
        ],
        {"messageStop": {"stopReason": "end_turn"}},
    ]


def _streamed_converse(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, events: list[dict[str, Any]]
) -> tuple[Any, Any]:
    client = _client("bedrock-runtime")
    _stub_api_call(client, [{"stream": FakeEventStream(events)}], monkeypatch)
    wrap_bedrock(client)
    response = client.converse_stream(modelId="m", messages=[])  # type: ignore[attr-defined]
    stream = response["stream"]
    assert list(stream) == events
    return cast(Any, stream)._state, memory.span_exporter.get_finished_spans()[0]


@pytest.mark.parametrize(
    ("max_attribute_length", "redact", "expected"),
    [
        (1_000, False, "東" * 400),
        (1_000, True, "[redacted]"),
        (100, False, "...[truncated]"),
        (0, False, None),
    ],
)
def test_configured_cap_applies_after_the_mask_not_to_stream_retention(
    make: Any,
    monkeypatch: pytest.MonkeyPatch,
    max_attribute_length: int,
    redact: bool,
    expected: str | None,
) -> None:
    def mask(value: Any, _context: Any) -> Any:
        if not redact:
            return value
        return [{"role": "assistant", "parts": [{"type": "text", "content": "[redacted]"}]}]

    memory = make(max_attribute_length=max_attribute_length, mask=mask)
    text = ("東" * 400) if not redact else ("x" * 5_000)

    state, span = _streamed_converse(memory, monkeypatch, _converse_text_stream(text))

    assert state.budget.truncated is False
    assert state.blocks[0]["text"] == text
    assert "telemetry.dev.capture.truncated" not in span.attributes
    exported = span.attributes.get("gen_ai.output.messages", "")
    if expected is None:
        assert exported == ""
    elif expected == "...[truncated]":
        assert isinstance(exported, str)
        assert len(exported) <= max_attribute_length
        assert exported.endswith(expected)
    else:
        assert _json_attr(span, "gen_ai.output.messages")[0]["parts"][0]["content"] == expected


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("max_attribute_length", [0, 100, 1024 * 1024])
def test_stream_beyond_the_stream_bound_is_flagged_and_withheld_from_a_mask(
    make: Any, monkeypatch: pytest.MonkeyPatch, masked: bool, max_attribute_length: int
) -> None:
    seen: list[Any] = []

    def mask(value: Any, _context: Any) -> Any:
        seen.append(value)
        return value

    options: dict[str, Any] = {"max_attribute_length": max_attribute_length}
    if masked:
        options["mask"] = mask
    memory = make(**options)

    state, span = _streamed_converse(
        memory, monkeypatch, _converse_text_stream("x" * 100, repeat=1_100)
    )

    assert state.budget.truncated is True
    # Cut at the stream bound, within one 100-byte delta of it.
    assert STREAM_LIMIT - 200 < state.budget.bytes_used <= STREAM_LIMIT
    retained = state.blocks[0]["text"]
    assert 0 < len(retained) < 1_100 * 100
    assert span.attributes["telemetry.dev.capture.truncated"] is True
    if masked:
        assert "gen_ai.output.messages" not in span.attributes
        assert not any(retained in json.dumps(value) for value in seen)
    elif max_attribute_length > len(retained):
        output = _json_attr(span, "gen_ai.output.messages")
        assert output[0]["parts"][0]["content"] == retained
    else:
        assert len(str(span.attributes.get("gen_ai.output.messages", ""))) <= max_attribute_length


@pytest.mark.parametrize(
    ("length", "cut"), [(64 * 1024, False), (telemetry_dev.MAX_RECORD_BYTES, True)]
)
def test_with_the_default_bounds_a_converse_stream_is_cut_only_at_the_record_bound(
    make: Any, monkeypatch: pytest.MonkeyPatch, length: int, cut: bool
) -> None:
    conftest.other_stream.release()
    errors: list[BaseException] = []
    memory = make(max_stream_bytes=64 * 1024 * 1024, on_error=errors.append)
    deltas = [
        MIB_TEXT[: min(len(MIB_TEXT), length - start)] for start in range(0, length, len(MIB_TEXT))
    ]
    events = [
        {"messageStart": {"role": "assistant"}},
        *[{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": d}}} for d in deltas],
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    _state, span = _streamed_converse(memory, monkeypatch, events)

    # A mapping failure would also flag the capture and keep the same 15 deltas, so this is the
    # assertion that tells it from the cut at the bound.
    failures = _instrumentation_errors(errors)
    assert not failures, "".join(
        line for cause in failures for line in traceback.format_exception(cause)
    )
    text = _json_attr(span, "gen_ai.output.messages")[0]["parts"][0]["content"]
    assert span.attributes.get("telemetry.dev.capture.truncated") is (True if cut else None)
    # Each delta costs its text plus 52 bytes, so the 16th 1 MiB delta does not fit and is
    # rejected whole.
    assert len(text) == (telemetry_dev.MAX_RECORD_BYTES - len(MIB_TEXT) if cut else length)


def test_a_converse_stream_of_small_deltas_holds_about_the_bytes_it_is_charged(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    conftest.other_stream.release()
    memory = make(max_stream_bytes=64 * 1024 * 1024)
    # Reads at which memory is measured, after the memory a stream takes once. No multiple of
    # 1,000, 1,024, 2,000, 2,048, 4,096, 5,000, 8,192 or 10,000 lies between them, so with
    # batches of any of these sizes, or none, the 650 deltas read between them stay separate
    # strings; with batches of 256 they are joined.
    checkpoints = (7_325, 7_975)

    def delta(index: int) -> str:
        return f"{index % 10_000:04d}"

    # Built while the stream is read, so that each delta is a new string, as the decoded event
    # stream gives, and tracemalloc sees it.
    def events() -> Iterator[dict[str, Any]]:
        for index in range(checkpoints[-1]):
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": delta(index)}}}

    client = _client("bedrock-runtime")
    _stub_api_call(client, [{"stream": events()}], monkeypatch)
    wrap_bedrock(client)
    stream = client.converse_stream(modelId="m", messages=[])["stream"]

    held: list[int] = []
    tracemalloc.start()
    try:
        for read, _event in enumerate(stream, 1):
            if read in checkpoints:
                # What the stream does not hold, such as garbage in reference cycles, is freed
                # first.
                gc.collect()
                held.append(tracemalloc.get_traced_memory()[0])
    finally:
        tracemalloc.stop()

    # Each delta is a new string, charged at least its 4 bytes.
    # Before Python 3.12, StringIO keeps each written string apart, about 15 times that, unless
    # the deltas are joined first. Measured between two reads, so that what the stream holds
    # whatever its length does not count. The allowance covers the spare room of a StringIO
    # from Python 3.12 and the memory the core's accounting keeps, and what 256 waiting deltas
    # hold, about 16 KB, but not 650 more.
    charged = 4 * (checkpoints[1] - checkpoints[0])
    assert held[1] - held[0] < 1.3 * charged + 16 * 1024
    span = memory.span_exporter.get_finished_spans()[0]
    text = _json_attr(span, "gen_ai.output.messages")[0]["parts"][0]["content"]
    assert text == "".join(delta(index) for index in range(checkpoints[-1]))


@pytest.mark.parametrize("handler", ["read", "interrupt"])
@pytest.mark.parametrize("step", ["append", "join", "write"])
def test_a_signal_handler_inside_a_delta_buffer_write_keeps_each_delta_once(
    step: str, handler: str
) -> None:
    # A signal handler runs on the thread it interrupts, so one that closes the stream reads
    # the buffer inside a write, and SIGINT raises KeyboardInterrupt there: here right after a
    # delta is appended, right after the waiting deltas are read to be joined onto the tail,
    # and right after text is written to the StringIO.
    buffer = bedrock_streams._DeltaBuffer()
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

    class Written(io.StringIO):
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
    buffer = bedrock_streams._DeltaBuffer()
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


def test_masked_converse_stream_closed_early_withholds_partial_output(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = make(mask=identity_mask)
    client = _client("bedrock-runtime")
    events = _converse_text_stream("partial", repeat=3)
    _stub_api_call(client, [{"stream": FakeEventStream(events)}], monkeypatch)
    wrap_bedrock(client)
    response = client.converse_stream(modelId="m", messages=[])  # type: ignore[attr-defined]
    stream = iter(response["stream"])
    next(stream)
    next(stream)
    cast(Any, stream).close()

    span = memory.span_exporter.get_finished_spans()[0]
    assert "gen_ai.output.messages" not in span.attributes
    assert span.attributes["telemetry.dev.capture.truncated"] is True


def test_chunk_timing_includes_reasoning_and_tool_arguments_not_usage(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[float | None] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        calls.append(timestamp_ms)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    events = [
        {"chunk": {"bytes": _body(payload)}}
        for payload in [
            {"message": {"usage": {"input_tokens": 2}}},
            {"delta": {"thinking": "reasoning"}},
            {"delta": {"signature": "signature"}},
            {"delta": {"partial_json": '{"city":'}},
            {"delta": {"partial_json": ""}},
        ]
    ]
    client = _client("bedrock-runtime")
    _stub_api_call(client, [{"body": FakeEventStream(events)}], monkeypatch)
    wrap_bedrock(client)
    response = cast(Any, client).invoke_model_with_response_stream(
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    assert list(response["body"]) == events
    assert len(calls) == 2


def test_invoke_model_stream_bounds_content_but_keeps_terminal_metadata(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        *[
            {"chunk": {"bytes": _body({"completion": f"{index:04d}" + "x" * 96})}}
            for index in range(1100)
        ],
        {
            "chunk": {
                "bytes": _body(
                    {
                        "stop_reason": "max_tokens",
                        "amazon-bedrock-invocationMetrics": {
                            "inputTokenCount": 7,
                            "outputTokenCount": 11,
                        },
                    }
                )
            }
        },
    ]
    client = _client("bedrock-runtime")
    _stub_api_call(client, [{"body": FakeEventStream(events)}], monkeypatch)
    wrap_bedrock(client)

    response = client.invoke_model_with_response_stream(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    instrumented_stream = response["body"]
    delivered = list(instrumented_stream)
    state = cast(Any, instrumented_stream)._state

    assert delivered == events
    assert state.budget.truncated is True
    assert state.budget.bytes_used <= state.budget.max_bytes
    span = memory.span_exporter.get_finished_spans()[0]
    retained_text = _json_attr(span, "gen_ai.output.messages")[0]["parts"][0]["content"]
    expected_prefix = "".join(f"{index:04d}" + "x" * 96 for index in range(len(state.chunks)))
    assert retained_text == expected_prefix
    assert len(retained_text) < 1100 * 100
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 11
    assert list(span.attributes["gen_ai.response.finish_reasons"]) == ["max_tokens"]


def test_unconsumed_stream_finishes_partial(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
    _stub_api_call(
        client,
        [{"stream": FakeEventStream([{"messageStart": {"role": "assistant"}}])}],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.converse_stream(modelId="m", messages=[])  # type: ignore[attr-defined]
    assert len(memory.span_exporter.get_finished_spans()) == 0
    del response
    gc.collect()

    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.operation.name"] == "chat"
    assert spans[0].attributes["telemetry.dev.capture.truncated"] is True


@pytest.mark.parametrize("kind", ["absent", "compaction", "at-bound", "bounded"])
def test_claude_invoke_compaction_usage(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    compaction = {
        "type": "compaction",
        "input_tokens": 11,
        "output_tokens": 7,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 5,
    }
    usage: dict[str, Any] = {
        "input_tokens": 20,
        "output_tokens": 9,
        "cache_read_input_tokens": 2,
        "cache_creation_input_tokens": 4,
    }
    if kind != "absent":
        usage["iterations"] = (
            [
                {
                    **compaction,
                    "input_tokens": 17,
                    "output_tokens": 13,
                    "cache_read_input_tokens": 6,
                    "cache_creation_input_tokens": 8,
                }
            ]
            + [{"type": "message", "input_tokens": 20}] * (999 if kind == "bounded" else 998)
            + [compaction]
            if kind in ("bounded", "at-bound")
            else [compaction, {"type": "message", "input_tokens": 20}]
        )
    events = [
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 1,
                    "cache_read_input_tokens": 1,
                    "cache_creation_input_tokens": 1,
                }
            },
        },
        {"type": "message_delta", "usage": usage, "delta": {"stop_reason": "end_turn"}},
        {
            "type": "message_stop",
            **(
                {}
                if kind == "compaction"
                else {
                    "amazon-bedrock-invocationMetrics": {
                        "inputTokenCount": 31,
                        "outputTokenCount": 16,
                    }
                }
            ),
        },
    ]
    raw = _body({"usage": usage})
    client = _client("bedrock-runtime")
    _stub_api_call(
        client,
        [
            {"body": StreamingBody(io.BytesIO(raw), len(raw)), "contentType": "application/json"},
            {"body": FakeEventStream([{"chunk": {"bytes": _body(event)}} for event in events])},
        ],
        monkeypatch,
    )
    wrapped = cast(Any, wrap_bedrock(client))
    result = wrapped.invoke_model(modelId="anthropic.claude", body=_body({"messages": []}))
    assert result["body"].read() == raw
    response = wrapped.invoke_model_with_response_stream(
        modelId="anthropic.claude", body=_body({"messages": []})
    )
    list(response["body"])
    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.attributes["gen_ai.usage.input_tokens"] == (
            26 if kind == "absent" else 76 if kind == "at-bound" else 45
        )
        assert span.attributes["gen_ai.usage.output_tokens"] == (
            9 if kind == "absent" else 29 if kind == "at-bound" else 16
        )
        assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == (
            2 if kind == "absent" else 11 if kind == "at-bound" else 5
        )
        assert span.attributes["gen_ai.usage.cache_creation.input_tokens"] == (
            4 if kind == "absent" else 17 if kind == "at-bound" else 9
        )
        if kind == "bounded":
            assert span.attributes["telemetry.dev.capture.truncated"] is True


INTERRUPTED_COMPACTION: dict[str, Any] = {
    "type": "compaction",
    "input_tokens": 11,
    "output_tokens": 7,
    "cache_read_input_tokens": 3,
    "cache_creation_input_tokens": 5,
}
INTERRUPTED_BASE: dict[str, Any] = {
    "input_tokens": 20,
    "cache_read_input_tokens": 2,
    "cache_creation_input_tokens": 4,
}


def _claude_invoke_stream(
    monkeypatch: pytest.MonkeyPatch, events: list[dict[str, Any]]
) -> Iterator[Any]:
    client = _client("bedrock-runtime")
    _stub_api_call(
        client,
        [{"body": FakeEventStream([{"chunk": {"bytes": _body(event)}} for event in events])}],
        monkeypatch,
    )
    response = cast(Any, wrap_bedrock(client)).invoke_model_with_response_stream(
        modelId="anthropic.claude", body=_body({"messages": []})
    )
    return iter(response["body"])


@pytest.mark.parametrize(
    "deltas,input_tokens,output_tokens",
    [
        ([], 45, 9),
        ([{"type": "message_delta", "delta": {}, "usage": {"output_tokens": 3}}], 45, 10),
        (
            [
                {
                    "type": "message_delta",
                    "delta": {},
                    "usage": {
                        **INTERRUPTED_BASE,
                        "output_tokens": 3,
                        "iterations": [
                            INTERRUPTED_COMPACTION,
                            {"type": "message"},
                            INTERRUPTED_COMPACTION,
                        ],
                    },
                }
            ],
            26 + 2 * 19,
            3 + 2 * 7,
        ),
    ],
    ids=["start_only", "delta_without_iterations", "delta_with_cumulative_iterations"],
)
def test_interrupted_claude_invoke_stream_keeps_message_start_compaction(
    memory: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    deltas: list[dict[str, Any]],
    input_tokens: int,
    output_tokens: int,
) -> None:
    start_usage = {
        **INTERRUPTED_BASE,
        "output_tokens": 2,
        "iterations": [INTERRUPTED_COMPACTION, {"type": "message", "input_tokens": 20}],
    }
    events = [
        {"type": "message_start", "message": {"usage": start_usage}},
        *deltas,
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
        {"type": "message_stop"},
    ]
    stream = _claude_invoke_stream(monkeypatch, events)
    for _ in range(1 + len(deltas)):
        next(stream)
    cast(Any, stream).close()

    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.usage.input_tokens"] == input_tokens
    assert span.attributes["gen_ai.usage.output_tokens"] == output_tokens
    assert span.attributes["telemetry.dev.capture.truncated"] is True


def test_message_start_compaction_is_never_added_to_invocation_metrics(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        {
            "type": "message_start",
            "message": {"usage": {**INTERRUPTED_BASE, "iterations": [INTERRUPTED_COMPACTION]}},
        },
        {
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": {"inputTokenCount": 31, "outputTokenCount": 16},
        },
    ]
    list(_claude_invoke_stream(monkeypatch, events))

    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.usage.input_tokens"] == 37
    assert span.attributes["gen_ai.usage.output_tokens"] == 16
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 2
    assert span.attributes["gen_ai.usage.cache_creation.input_tokens"] == 4


@pytest.mark.parametrize("count", [0, None, "9", True, float("inf"), float("nan"), 10**400])
def test_claude_final_usage_precedes_metrics(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, count: Any
) -> None:
    usage: dict[str, Any] = {
        "iterations": [
            {
                "type": "compaction",
                "input_tokens": 11.0,
                "output_tokens": 7.0,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 5,
            }
        ],
    }
    if count is not None:
        usage.update(input_tokens=count, output_tokens=count)
    events = [
        {"type": "message_delta", "usage": usage},
        {
            "type": "message_stop",
            "amazon-bedrock-invocationMetrics": {"inputTokenCount": 31, "outputTokenCount": 16},
        },
    ]
    client = _client("bedrock-runtime")
    raw = _body({"usage": usage})
    _stub_api_call(
        client,
        [
            {"body": StreamingBody(io.BytesIO(raw), len(raw)), "contentType": "application/json"},
            {"body": FakeEventStream([{"chunk": {"bytes": _body(event)}} for event in events])},
        ],
        monkeypatch,
    )
    wrapped = cast(Any, wrap_bedrock(client))
    result = wrapped.invoke_model(modelId="anthropic.claude", body=_body({"messages": []}))
    assert result["body"].read() == raw
    response = wrapped.invoke_model_with_response_stream(
        modelId="anthropic.claude", body=_body({"messages": []})
    )
    list(response["body"])
    nonstream, span = memory.span_exporter.get_finished_spans()
    assert nonstream.attributes["gen_ai.usage.input_tokens"] == 19
    assert nonstream.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["gen_ai.usage.input_tokens"] == (19 if count == 0 else 31)
    assert span.attributes["gen_ai.usage.output_tokens"] == (7 if count == 0 else 16)


def test_invoke_model_provider_native_and_streaming_body(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
    raw = _body({"completion": "answer", "stop_reason": "end_turn"})
    invoke_body = StreamingBody(io.BytesIO(raw), len(raw))
    mistral_raw = _body({"outputs": [{"text": "mistral", "stop_reason": "stop"}]})
    _stub_api_call(
        client,
        [
            {
                "body": invoke_body,
                "contentType": "application/json",
                "ResponseMetadata": {
                    "RequestId": "invoke-1",
                    "HTTPHeaders": {
                        "x-amzn-bedrock-input-token-count": "9",
                        "x-amzn-bedrock-output-token-count": "4",
                    },
                    "HTTPStatusCode": 200,
                    "RetryAttempts": 1,
                    "TotalRetryDelay": 6,
                },
            },
            {
                "body": FakeEventStream(
                    [
                        {
                            "chunk": {
                                "bytes": _body(
                                    {
                                        "type": "message_start",
                                        "message": {
                                            "usage": {
                                                "input_tokens": 5,
                                                "cache_read_input_tokens": 1,
                                            }
                                        },
                                    }
                                )
                            }
                        },
                        {
                            "chunk": {
                                "bytes": _body(
                                    {
                                        "type": "content_block_delta",
                                        "delta": {"text": "hi"},
                                    }
                                )
                            }
                        },
                        {"chunk": {"bytes": _body({"completion": " legacy"})}},
                        {
                            "chunk": {
                                "bytes": _body(
                                    {
                                        "type": "message_delta",
                                        "delta": {"stop_reason": "end_turn"},
                                    }
                                )
                            }
                        },
                        {
                            "chunk": {
                                "bytes": _body(
                                    {
                                        "type": "message_delta",
                                        "delta": {"stop_reason": "end_turn"},
                                        "usage": {"output_tokens": 2},
                                    }
                                )
                            }
                        },
                    ]
                )
            },
            {
                "body": StreamingBody(
                    io.BytesIO(mistral_raw),
                    len(mistral_raw),
                ),
                "contentType": "application/json",
                "ResponseMetadata": {
                    "HTTPHeaders": {
                        "x-amzn-bedrock-input-token-count": "11",
                        "x-amzn-bedrock-output-token-count": "12",
                    }
                },
            },
            {
                "body": FailingStreamingBody(io.BytesIO(b"{}"), 2),
                "contentType": "application/json",
                "ResponseMetadata": {"RequestId": "invoke-read-failed"},
            },
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10, "top_k": 3}),
    )
    assert invoke_body.tell() == 0
    assert response["body"].read() == raw
    stream_response = client.invoke_model_with_response_stream(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    list(stream_response["body"])
    seekable_body = io.BytesIO(_body({"prompt": "hello", "temperature": 0.2}))
    mistral_response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="mistral.mistral-large",
        contentType="application/json",
        body=seekable_body,
    )
    assert seekable_body.tell() == 0
    assert json.loads(mistral_response["body"].read())["outputs"][0]["text"] == "mistral"
    failed_read_response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    assert isinstance(failed_read_response["body"], StreamingBody)
    with pytest.raises(OSError, match="read failed"):
        failed_read_response["body"].read()

    invoke_span, stream_span, mistral_span, failed_read_span = (
        memory.span_exporter.get_finished_spans()
    )
    assert invoke_span.attributes["gen_ai.request.top_k"] == 3
    assert invoke_span.attributes["gen_ai.usage.input_tokens"] == 9
    assert invoke_span.attributes["gen_ai.response.id"] == "invoke-1"
    assert invoke_span.attributes["aws.http.status_code"] == 200
    assert invoke_span.attributes["aws.request.attempts"] == 2
    assert invoke_span.attributes["aws.request.total_retry_delay_ms"] == 6
    assert (
        _json_attr(stream_span, "gen_ai.output.messages")[0]["parts"][0]["content"] == "hi legacy"
    )
    assert stream_span.attributes["gen_ai.usage.input_tokens"] == 5 + 1
    assert stream_span.attributes["gen_ai.usage.output_tokens"] == 2
    assert stream_span.attributes["gen_ai.usage.cache_read.input_tokens"] == 1
    assert list(stream_span.attributes["gen_ai.response.finish_reasons"]) == ["end_turn"]
    assert mistral_span.attributes["gen_ai.request.temperature"] == 0.2
    assert mistral_span.attributes["gen_ai.usage.input_tokens"] == 11
    assert mistral_span.attributes["gen_ai.usage.output_tokens"] == 12
    assert failed_read_span.attributes["gen_ai.response.id"] == "invoke-read-failed"
    assert _json_attr(stream_span, "gen_ai.output.messages")[0]["finish_reason"] == "end_turn"


def test_invoke_stream_decodes_once_and_delivers_malformed_payload(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        {"chunk": {"bytes": _body({"delta": {"text": "hi"}})}},
        {"chunk": {"bytes": b"not-json"}},
    ]
    client = _client("bedrock-runtime")
    _stub_api_call(client, [{"body": FakeEventStream(events)}], monkeypatch)
    loads = bedrock_streams.json.loads
    calls = 0

    def counting_loads(value: Any) -> Any:
        nonlocal calls
        calls += 1
        return loads(value)

    monkeypatch.setattr(bedrock_streams.json, "loads", counting_loads)
    wrap_bedrock(client)
    response = client.invoke_model_with_response_stream(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    before = calls

    assert list(response["body"]) == events
    assert calls - before == 2
    assert len(memory.span_exporter.get_finished_spans()) == 1


def test_invoke_model_streaming_body_preserves_read_all_and_context_manager(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _body({"completion": "answer", "stop_reason": "end_turn"})
    unknown_length_source = StreamingBody(io.BytesIO(raw), None)
    context_raw = io.BytesIO(raw)
    context_source = StreamingBody(context_raw, len(raw))
    _stub_api_call(
        client := _client("bedrock-runtime"),
        [
            {
                "body": unknown_length_source,
                "contentType": "application/json",
                "ResponseMetadata": {"RequestId": "read-all"},
            },
            {
                "body": context_source,
                "contentType": "application/json",
                "ResponseMetadata": {
                    "RequestId": "context",
                    "HTTPHeaders": {
                        "x-amzn-bedrock-input-token-count": "9",
                        "x-amzn-bedrock-output-token-count": "4",
                    },
                },
            },
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    assert response["body"].read(-1) == raw
    read_all_span = memory.span_exporter.get_finished_spans()[0]
    assert read_all_span.attributes["gen_ai.response.id"] == "read-all"
    assert _json_attr(read_all_span, "gen_ai.output.messages") == {
        "completion": "answer",
        "stop_reason": "end_turn",
    }
    assert list(read_all_span.attributes["gen_ai.response.finish_reasons"]) == ["end_turn"]

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    with response["body"] as entered:
        assert entered is not context_raw
        assert entered.read() == raw
        assert entered.tell() == len(raw)

    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[1].attributes["gen_ai.response.id"] == "context"
    assert _json_attr(spans[1], "gen_ai.output.messages") == {
        "completion": "answer",
        "stop_reason": "end_turn",
    }
    assert spans[1].attributes["gen_ai.usage.input_tokens"] == 9
    assert spans[1].attributes["gen_ai.usage.output_tokens"] == 4
    assert list(spans[1].attributes["gen_ai.response.finish_reasons"]) == ["end_turn"]


def test_invoke_model_context_stream_iteration_and_readline_are_captured(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _body({"completion": "answer", "stop_reason": "end_turn"})
    _stub_api_call(
        client := _client("bedrock-runtime"),
        [
            {
                "body": StreamingBody(io.BytesIO(raw), len(raw)),
                "contentType": "application/json",
                "ResponseMetadata": {"RequestId": "iterated"},
            },
            {
                "body": StreamingBody(io.BytesIO(raw), len(raw)),
                "contentType": "application/json",
                "ResponseMetadata": {"RequestId": "readline"},
            },
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    iterated = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    chunks: list[bytes] = []
    with iterated["body"] as stream:
        for chunk in stream:
            chunks.append(chunk)
    assert b"".join(chunks) == raw

    line_response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    with line_response["body"] as stream:
        assert stream.readline() == raw

    iterated_span, readline_span = memory.span_exporter.get_finished_spans()
    assert iterated_span.attributes["gen_ai.response.id"] == "iterated"
    assert readline_span.attributes["gen_ai.response.id"] == "readline"
    assert _json_attr(iterated_span, "gen_ai.output.messages")["completion"] == "answer"
    assert _json_attr(readline_span, "gen_ai.output.messages")["completion"] == "answer"


def test_invoke_model_context_readlines_hint_at_eof_finishes_capture(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _body({"completion": "answer", "stop_reason": "end_turn"})
    _stub_api_call(
        client := _client("bedrock-runtime"),
        [
            {
                "body": StreamingBody(io.BytesIO(raw), None),
                "contentType": "application/json",
                "ResponseMetadata": {"RequestId": "readlines-hint"},
            }
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    with response["body"] as stream:
        assert stream.readlines(len(raw)) == [raw]

    [span] = memory.span_exporter.get_finished_spans()
    assert span.attributes["gen_ai.response.id"] == "readlines-hint"
    assert _json_attr(span, "gen_ai.output.messages") == {
        "completion": "answer",
        "stop_reason": "end_turn",
    }
    assert list(span.attributes["gen_ai.response.finish_reasons"]) == ["end_turn"]


def test_invoke_model_context_readinto_rejects_readonly_before_consuming_pending_byte(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_line = b'{"completion":"answer",\n'
    rest = b'"stop_reason":"end_turn"}'
    raw = first_line + rest
    _stub_api_call(
        client := _client("bedrock-runtime"),
        [
            {
                "body": StreamingBody(io.BytesIO(raw), None),
                "contentType": "application/json",
                "ResponseMetadata": {"RequestId": "readlines-probe"},
            }
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    with response["body"] as stream:
        assert stream.readlines(len(first_line)) == [first_line]
        amount_read = cast(Any, stream)._body._amount_read
        with pytest.raises(TypeError):
            stream.readinto(bytes(1))
        assert cast(Any, stream)._body._amount_read == amount_read
        assert stream.read() == rest

    [span] = memory.span_exporter.get_finished_spans()
    assert span.attributes["gen_ai.response.id"] == "readlines-probe"
    assert span.attributes["error.type"] == "TypeError"


def test_invoke_model_context_readinto_writes_pending_and_delegated_bytes(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_line = b'{"completion":"answer",\n'
    rest = b'"stop_reason":"end_turn"}'
    raw = first_line + rest
    _stub_api_call(
        client := _client("bedrock-runtime"),
        [
            {
                "body": StreamingBody(io.BytesIO(raw), None),
                "contentType": "application/json",
                "ResponseMetadata": {"RequestId": "readinto"},
            }
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    with response["body"] as stream:
        assert stream.readlines(len(first_line)) == [first_line]
        pending = bytearray(1)
        assert stream.readinto(pending) == len(pending)
        assert pending == rest[:1]
        delegated = bytearray(len(rest) - 1)
        assert stream.readinto(delegated) == len(delegated)
        assert delegated == rest[1:]
        assert stream.read() == b""

    [span] = memory.span_exporter.get_finished_spans()
    assert span.attributes["gen_ai.response.id"] == "readinto"
    assert _json_attr(span, "gen_ai.output.messages")["completion"] == "answer"


def test_invoke_model_streaming_body_offers_readinto_only_when_botocore_does(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
    raw = _body({"completion": "answer"})
    _stub_api_call(
        client,
        [{"body": StreamingBody(io.BytesIO(raw), len(raw)), "contentType": "application/json"}],
        monkeypatch,
    )
    wrap_bedrock(client)

    body = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )["body"]

    assert hasattr(body, "readinto") == hasattr(StreamingBody, "readinto")
    assert body.read() == raw


@pytest.mark.skipif(
    not hasattr(StreamingBody, "readinto"), reason="botocore added readinto in 1.39"
)
@pytest.mark.parametrize("extra", [0, 1])
def test_invoke_model_streaming_body_bounds_capture_and_preserves_readinto(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, extra: int
) -> None:
    client = _client("bedrock-runtime")
    # The body takes the stream bound: one of STREAM_LIMIT bytes is kept, one more byte is not.
    padding = STREAM_LIMIT + extra - len(_body({"completion": ""}))
    raw = _body({"completion": "x" * padding})
    assert len(raw) == STREAM_LIMIT + extra
    source = StreamingBody(io.BytesIO(raw), len(raw))
    _stub_api_call(
        client,
        [{"body": source, "contentType": "application/json"}],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    body = response["body"]
    assert source.tell() == 0

    delivered = bytearray()
    buffer = bytearray(4096)
    while amount := body.readinto(buffer):
        delivered.extend(buffer[:amount])

    budget = cast(Any, body)._budget
    assert bytes(delivered) == raw
    assert budget.bytes_used == STREAM_LIMIT
    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 1
    if not extra:
        assert budget.truncated is False
        assert "telemetry.dev.capture.truncated" not in spans[0].attributes
        assert "gen_ai.output.messages" in spans[0].attributes
        return
    assert budget.truncated is True
    assert spans[0].attributes["telemetry.dev.capture.truncated"] is True
    assert "gen_ai.output.messages" not in spans[0].attributes


class _FailAfterFirstReadBody(StreamingBody):
    def __init__(self, raw: bytes) -> None:
        super().__init__(io.BytesIO(raw), len(raw))
        self._reads = 0

    def read(self, amt: int | None = None) -> bytes:
        self._reads += 1
        if self._reads > 1:
            raise OSError("read failed")
        return super().read(amt)


@pytest.mark.parametrize("ending", ["close", "error"])
def test_invoke_model_body_read_partially_is_flagged_below_the_bound(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    client = _client("bedrock-runtime")
    raw = _body({"completion": "short answer", "stop_reason": "end_turn"})
    source = (
        _FailAfterFirstReadBody(raw)
        if ending == "error"
        else StreamingBody(io.BytesIO(raw), len(raw))
    )
    _stub_api_call(client, [{"body": source, "contentType": "application/json"}], monkeypatch)
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    body = response["body"]
    assert body.read(8) == raw[:8]
    if ending == "close":
        body.close()
    else:
        with pytest.raises(OSError, match="read failed"):
            body.read(8)

    assert cast(Any, body)._budget.truncated is False
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["telemetry.dev.capture.truncated"] is True
    assert "gen_ai.output.messages" not in span.attributes
    if ending == "error":
        assert span.status.status_code.name == "ERROR"


@pytest.mark.parametrize("max_attribute_length", [0, 100])
def test_invoke_model_body_capture_ignores_a_small_configured_cap(
    make: Any, monkeypatch: pytest.MonkeyPatch, max_attribute_length: int
) -> None:
    memory = make(max_attribute_length=max_attribute_length)
    client = _client("bedrock-runtime")
    raw = _body({"completion": "x" * 5_000, "stop_reason": "end_turn"})
    _stub_api_call(
        client,
        [{"body": StreamingBody(io.BytesIO(raw), len(raw)), "contentType": "application/json"}],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    assert response["body"].read() == raw

    span = memory.span_exporter.get_finished_spans()[0]
    assert cast(Any, response["body"])._budget.truncated is False
    assert "telemetry.dev.capture.truncated" not in span.attributes
    assert list(span.attributes["gen_ai.response.finish_reasons"]) == ["end_turn"]
    exported = str(span.attributes.get("gen_ai.output.messages", ""))
    assert len(exported) <= max_attribute_length
    if max_attribute_length:
        assert exported.endswith("...[truncated]")


def test_invoke_model_nova_sampling_fields(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
    raw = _body({"output": {"message": {}}})
    _stub_api_call(
        client,
        [{"body": StreamingBody(io.BytesIO(raw), len(raw))}],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="amazon.nova-pro-v1:0",
        contentType="application/json",
        body=_body(
            {
                "messages": [],
                "inferenceConfig": {
                    "temperature": 0.4,
                    "topP": 0.9,
                    "topK": 20,
                    "maxTokens": 500,
                    "stopSequences": ["stop"],
                },
            }
        ),
    )
    response["body"].read()

    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.request.temperature"] == 0.4
    assert span.attributes["gen_ai.request.top_p"] == 0.9
    assert span.attributes["gen_ai.request.top_k"] == 20
    assert span.attributes["gen_ai.request.max_tokens"] == 500
    assert span.attributes["gen_ai.request.stop_sequences"] == ("stop",)


def test_invoke_model_stream_titan_counts_finish_and_unrewindable_body(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
    unrewindable_body = UnrewindableBody(_body({"inputText": "hello"}))
    _stub_api_call(
        client,
        [
            {
                "body": FakeEventStream(
                    [
                        {
                            "chunk": {
                                "bytes": _body(
                                    {
                                        "outputText": "Hello ",
                                        "inputTextTokenCount": 7,
                                    }
                                )
                            }
                        },
                        {
                            "chunk": {
                                "bytes": _body(
                                    {
                                        "outputText": "world",
                                        "totalOutputTextTokenCount": 3,
                                        "completionReason": "FINISHED",
                                    }
                                )
                            }
                        },
                    ]
                )
            },
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    titan_response = client.invoke_model_with_response_stream(  # type: ignore[attr-defined]
        modelId="amazon.titan-text-express-v1",
        contentType="application/json",
        body=unrewindable_body,
    )
    list(titan_response["body"])

    assert unrewindable_body.reads == 0
    titan_span = memory.span_exporter.get_finished_spans()[0]
    assert (
        _json_attr(titan_span, "gen_ai.output.messages")[0]["parts"][0]["content"] == "Hello world"
    )
    assert titan_span.attributes["gen_ai.usage.input_tokens"] == 7
    assert titan_span.attributes["gen_ai.usage.output_tokens"] == 3
    assert list(titan_span.attributes["gen_ai.response.finish_reasons"]) == ["FINISHED"]


@pytest.mark.parametrize(
    ("chunks", "content", "usage"),
    [
        (
            [
                {
                    "text": "choice",
                    "finish_reason": "COMPLETE",
                    "meta": {"billed_units": {"input_tokens": 8, "output_tokens": 3}},
                }
            ],
            "choice",
            (8, 3),
        ),
        (
            [
                {"generations": [{"text": "Cohere "}]},
                {"generations": [{"text": "done", "finish_reason": "COMPLETE"}]},
            ],
            "Cohere done",
            None,
        ),
    ],
    ids=["text", "generations"],
)
def test_invoke_model_stream_cohere_generations_finish(
    memory: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[dict[str, Any]],
    content: str,
    usage: tuple[int, int] | None,
) -> None:
    client = _client("bedrock-runtime")
    _stub_api_call(
        client,
        [{"body": FakeEventStream([{"chunk": {"bytes": _body(chunk)}} for chunk in chunks])}],
        monkeypatch,
    )
    wrap_bedrock(client)

    cohere_response = client.invoke_model_with_response_stream(  # type: ignore[attr-defined]
        modelId="cohere.command-text-v14",
        contentType="application/json",
        body=_body({"prompt": "choose"}),
    )
    list(cohere_response["body"])

    cohere_span = memory.span_exporter.get_finished_spans()[0]
    output = _json_attr(cohere_span, "gen_ai.output.messages")[0]
    assert output["parts"][0]["content"] == content
    assert output["finish_reason"] == "COMPLETE"
    assert list(cohere_span.attributes["gen_ai.response.finish_reasons"]) == ["COMPLETE"]
    if usage is not None:
        assert cohere_span.attributes["gen_ai.usage.input_tokens"] == usage[0]
        assert cohere_span.attributes["gen_ai.usage.output_tokens"] == usage[1]


def test_embeddings_apply_guardrail_and_error(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
    embedding_raw = _body({"embedding": [0.1]})
    error = ClientError(
        error_response={
            "Error": {"Code": "ThrottlingException", "Message": "slow down"},
            "ResponseMetadata": {
                "RequestId": "err-1",
                "RetryAttempts": 1,
                "HTTPStatusCode": 429,
                "TotalRetryDelay": 25,
            },
        },
        operation_name="Converse",
    )
    _stub_api_call(
        client,
        [
            {
                "body": StreamingBody(io.BytesIO(embedding_raw), len(embedding_raw)),
                "contentType": "application/json",
                "ResponseMetadata": {"HTTPHeaders": {"x-amzn-bedrock-input-token-count": "6"}},
            },
            {
                "action": "GUARDRAIL_INTERVENED",
                "actionReason": "blocked",
                "outputs": [],
                "ResponseMetadata": {
                    "RequestId": "guardrail-1",
                    "HTTPStatusCode": 200,
                    "RetryAttempts": 2,
                    "TotalRetryDelay": 4,
                },
            },
            error,
        ],
        monkeypatch,
    )
    wrap_bedrock(client)

    embedding_response = client.invoke_model(  # type: ignore[attr-defined]
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        body=_body({"inputText": "hello"}),
    )
    embedding_response["body"].read()
    client.apply_guardrail(  # type: ignore[attr-defined]
        guardrailIdentifier="gr-1", guardrailVersion="1", source="INPUT", content=[]
    )
    with pytest.raises(ClientError):
        client.converse(modelId="m", messages=[])  # type: ignore[attr-defined]

    embedding, guardrail, failed = memory.span_exporter.get_finished_spans()
    assert embedding.attributes["gen_ai.operation.name"] == "embeddings"
    assert embedding.attributes["gen_ai.output.type"] == "embedding"
    assert embedding.attributes["gen_ai.usage.input_tokens"] == 6
    assert guardrail.name == "apply_guardrail gr-1"
    assert guardrail.attributes["td.metadata.guardrail_action"] == "GUARDRAIL_INTERVENED"
    assert guardrail.attributes["gen_ai.response.id"] == "guardrail-1"
    assert guardrail.attributes["aws.http.status_code"] == 200
    assert guardrail.attributes["aws.request.attempts"] == 3
    assert guardrail.attributes["aws.request.total_retry_delay_ms"] == 4
    assert failed.attributes["aws.error.code"] == "ThrottlingException"
    assert failed.attributes["gen_ai.response.id"] == "err-1"
    assert failed.attributes["aws.http.status_code"] == 429
    assert failed.attributes["aws.request.total_retry_delay_ms"] == 25


def test_agent_stream_keeps_return_control_after_budget_exhaustion(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    return_control = {"invocationId": "inv-after-truncation"}
    events = [
        *[{"chunk": {"bytes": b"x" * 100}} for _ in range(1100)],
        {"returnControl": return_control},
    ]
    client = _client("bedrock-agent-runtime")
    _stub_api_call(
        client,
        [{"completion": FakeEventStream(events), "sessionId": "sess-1"}],
        monkeypatch,
    )
    wrap_bedrock(client)

    response = client.invoke_agent(  # type: ignore[attr-defined]
        agentId="agent-1", agentAliasId="alias", sessionId="sess-1", inputText="hi"
    )
    instrumented_stream = response["completion"]

    assert list(instrumented_stream) == events
    state = cast(Any, instrumented_stream)._state
    assert state.budget.truncated is True
    assert state.return_control == return_control
    span = memory.span_exporter.get_finished_spans()[0]
    assert _json_attr(span, "gen_ai.output.messages") == {"returnControl": return_control}
    assert span.attributes["td.metadata.return_control"] == "true"
    assert span.attributes["telemetry.dev.capture.truncated"] is True


# Return control has the 16 MiB bound of a non-stream capture and no item limit.
@pytest.mark.parametrize(
    ("inputs", "kept"),
    [
        ([{"apiInvocationInput": {"actionGroup": OVER_RECORD}}], False),
        ([{"apiInvocationInput": {"actionGroup": "x" * (STREAM_LIMIT + 1)}}], True),
        ([{"function": "f"} for _ in range(1_500)], True),
    ],
    ids=["over-record-bytes", "over-stream-bytes", "many-items"],
)
def test_masked_agent_stream_drops_return_control_only_beyond_the_record_bound(
    make: Any, monkeypatch: pytest.MonkeyPatch, inputs: list[Any], kept: bool
) -> None:
    seen: list[str] = []

    def mask(value: Any, _context: Any) -> Any:
        seen.append(json.dumps(value))
        return value

    memory = make(mask=mask)
    return_control = {"invocationId": "inv-oversized", "invocationInputs": inputs}
    events = [{"chunk": {"bytes": b"Short answer"}}, {"returnControl": return_control}]
    client = _client("bedrock-agent-runtime")
    _stub_api_call(
        client, [{"completion": FakeEventStream(events), "sessionId": "sess-1"}], monkeypatch
    )
    wrap_bedrock(client)

    response = client.invoke_agent(  # type: ignore[attr-defined]
        agentId="agent-1", agentAliasId="alias", sessionId="sess-1", inputText="hi"
    )
    instrumented_stream = response["completion"]
    assert list(instrumented_stream) == events

    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["td.metadata.return_control"] == "true"
    if kept:
        assert cast(Any, instrumented_stream)._state.return_control == return_control
        assert "telemetry.dev.capture.truncated" not in span.attributes
        assert _json_attr(span, "gen_ai.output.messages") == {"returnControl": return_control}
        return
    assert cast(Any, instrumented_stream)._state.return_control is None
    assert span.attributes["telemetry.dev.capture.truncated"] is True
    assert "gen_ai.output.messages" not in span.attributes
    assert not any("inv-oversized" in value or "Short answer" in value for value in seen)


@pytest.mark.parametrize("later_text", [False, True])
def test_masked_agent_stream_withholds_text_only_when_text_was_rejected(
    make: Any, monkeypatch: pytest.MonkeyPatch, later_text: bool
) -> None:
    seen: list[str] = []

    def mask(value: Any, _context: Any) -> Any:
        seen.append(json.dumps(value))
        return value

    memory = make(mask=mask)
    events: list[dict[str, Any]] = [
        {"chunk": {"bytes": b"Complete answer"}},
        {"trace": {"sessionId": "t" * 50_000}},
    ]
    if later_text:
        events.append({"chunk": {"bytes": b" and more"}})
    client = _client("bedrock-agent-runtime")
    _stub_api_call(
        client, [{"completion": FakeEventStream(events), "sessionId": "sess-1"}], monkeypatch
    )
    wrap_bedrock(client, capture_agent_trace=True)

    response = client.invoke_agent(  # type: ignore[attr-defined]
        agentId="agent-1", agentAliasId="alias", sessionId="sess-1", inputText="hi"
    )
    assert list(response["completion"]) == events

    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["telemetry.dev.capture.truncated"] is True
    if later_text:
        assert "gen_ai.output.messages" not in span.attributes
        assert not any("Complete answer" in value for value in seen)
    else:
        output = _json_attr(span, "gen_ai.output.messages")
        assert output[0]["parts"][0]["content"] == "Complete answer"
        assert any("Complete answer" in value for value in seen)


_COMPLETE = {"chunk": {"bytes": b"Complete answer"}}
_AFTER = {"chunk": {"bytes": b" after"}}
_UNDECODABLE = {"chunk": {"bytes": b"\xff"}}
# "café ok" split inside the é.
_CAFE = "café ok".encode()


@pytest.mark.parametrize(
    ("events", "failing", "masked", "kept", "cut", "reported"),
    [
        (
            [_COMPLETE, _UNDECODABLE, _AFTER],
            None,
            False,
            "Complete answer",
            True,
            UnicodeDecodeError,
        ),
        # The last text event: a later text event would be rejected, which withholds the output
        # from a mask on its own.
        ([_COMPLETE, _UNDECODABLE], None, True, None, True, UnicodeDecodeError),
        (
            [_COMPLETE, {"trace": {"trace": {}}}, _AFTER],
            (bedrock_streams.AgentStreamState, "_collect_trace"),
            False,
            "Complete answer after",
            False,
            ValueError,
        ),
        # The captured trace content is lost, as a trace dropped by the bound is, but no text
        # is: it is kept, also from a mask.
        (
            [_COMPLETE, {"trace": {"trace": {}}}, _AFTER],
            (bedrock_streams, "_project_trace"),
            True,
            "Complete answer after",
            True,
            ValueError,
        ),
        ([_COMPLETE], (bedrock_streams.AgentStreamState, "finish"), False, None, True, ValueError),
        # A character split between two chunks is decoded whole.
        (
            [{"chunk": {"bytes": _CAFE[:4]}}, {"chunk": {"bytes": _CAFE[4:]}}],
            None,
            False,
            "café ok",
            False,
            None,
        ),
        # The stream ends inside the é, so that character is lost.
        (
            [_COMPLETE, {"chunk": {"bytes": _CAFE[:4]}}],
            None,
            False,
            "Complete answercaf",
            True,
            UnicodeDecodeError,
        ),
    ],
    ids=[
        "text",
        "masked-text",
        "trace-summary",
        "masked-trace-content",
        "end",
        "split-character",
        "ends-inside-character",
    ],
)
def test_an_agent_stream_mapping_failure_is_reported_and_flags_only_lost_output(
    make: Any,
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    failing: tuple[Any, str] | None,
    masked: bool,
    kept: str | None,
    cut: bool,
    reported: type[BaseException] | None,
) -> None:
    if failing is not None:

        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("mapper broke")

        monkeypatch.setattr(*failing, fail)
    errors: list[BaseException] = []
    options: dict[str, Any] = {"on_error": errors.append}
    if masked:
        options["mask"] = identity_mask
    memory = make(**options)
    client = _client("bedrock-agent-runtime")
    _stub_api_call(
        client, [{"completion": FakeEventStream(events), "sessionId": "sess-1"}], monkeypatch
    )
    wrap_bedrock(client, capture_agent_trace=True)

    # boto3 generates service methods at runtime, so invoke_agent is not on the static type.
    response = client.invoke_agent(  # type: ignore[attr-defined]
        agentId="agent-1", agentAliasId="alias", sessionId="sess-1", inputText="hi"
    )
    assert list(response["completion"]) == events

    assert [type(error) for error in _instrumentation_errors(errors)] == (
        [reported] if reported else []
    )
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes.get("telemetry.dev.capture.truncated") is (True if cut else None)
    if kept is None:
        assert "gen_ai.output.messages" not in span.attributes
    else:
        # A lost text event keeps no later text, so the output is a prefix of the stream; a lost
        # trace loses no text.
        output = _json_attr(span, "gen_ai.output.messages")
        assert output[0]["parts"][0]["content"] == kept


def _converse_delta(text: str) -> dict[str, Any]:
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}


def _invoke_chunk(text: str) -> dict[str, Any]:
    return {"chunk": {"bytes": _body({"completion": text})}}


@pytest.mark.parametrize(
    ("operation", "failing", "events", "kept", "cut", "reported"),
    [
        (
            "converse_stream",
            (bedrock_streams, "usage_from_converse", None),
            # The usage comes before more text, so a usage failure that cut the output would
            # show.
            [
                {"messageStart": {"role": "assistant"}},
                _converse_delta("Hello"),
                {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2}}},
                _converse_delta(" world"),
                {"messageStop": {"stopReason": "end_turn"}},
            ],
            "Hello world",
            False,
            ValueError,
        ),
        (
            "converse_stream",
            (bedrock_streams.ConverseStreamState, "_append", 2),
            [
                {"messageStart": {"role": "assistant"}},
                _converse_delta("Hello"),
                _converse_delta(" lost"),
                _converse_delta(" after"),
                {"messageStop": {"stopReason": "end_turn"}},
            ],
            "Hello",
            True,
            ValueError,
        ),
        (
            "invoke_model_with_response_stream",
            # Each chunk fails, and the stream reports once.
            (bedrock_streams.InvokeModelStreamState, "_collect_anthropic_usage", None),
            [_invoke_chunk("Hello"), _invoke_chunk(" world")],
            "Hello world",
            False,
            ValueError,
        ),
        (
            "invoke_model_with_response_stream",
            None,
            [_invoke_chunk("Hello"), {"chunk": {"bytes": b"\xff"}}, _invoke_chunk(" after")],
            "Hello",
            True,
            UnicodeDecodeError,
        ),
        # The chunk-timing fallback reads the chunk again and fails the same way, which must not
        # reach the caller. Nesting past the recursion limit fails json.loads with an error that
        # is not a JSONDecodeError on every supported Python.
        (
            "invoke_model_with_response_stream",
            None,
            [_invoke_chunk("Hello"), {"chunk": {"bytes": b"[" * 100_000}}, _invoke_chunk(" after")],
            "Hello",
            True,
            RecursionError,
        ),
        (
            "invoke_model_with_response_stream",
            None,
            # The later undecodable chunk fails too, and the stream reports once.
            [
                _invoke_chunk("Hello"),
                {"chunk": {"bytes": b'{"completion": "lost'}},
                _invoke_chunk(" after"),
                {"chunk": {"bytes": b"\xff"}},
            ],
            "Hello",
            True,
            json.JSONDecodeError,
        ),
        # A chunk timing or first-chunk update that fails loses only that timing.
        (
            "converse_stream",
            (telemetry_dev.SpanHandle, "record_output_chunk", None),
            [
                {"messageStart": {"role": "assistant"}},
                _converse_delta("Hello"),
                _converse_delta(" world"),
                {"messageStop": {"stopReason": "end_turn"}},
            ],
            "Hello world",
            False,
            ValueError,
        ),
        (
            "converse_stream",
            (telemetry_dev.SpanHandle, "update", 1),
            [
                {"messageStart": {"role": "assistant"}},
                _converse_delta("Hello"),
                _converse_delta(" world"),
                {"messageStop": {"stopReason": "end_turn"}},
            ],
            "Hello world",
            False,
            ValueError,
        ),
        (
            "invoke_model_with_response_stream",
            None,
            # A blank chunk carries no output, so it is neither lost nor flagged.
            [_invoke_chunk("Hello"), {"chunk": {"bytes": b"\n"}}, _invoke_chunk(" after")],
            "Hello after",
            False,
            None,
        ),
    ],
    ids=[
        "converse-usage",
        "converse-delta",
        "invoke-usage",
        "invoke-undecodable",
        "invoke-unparsable",
        "invoke-not-json",
        "converse-chunk-timing",
        "converse-first-chunk",
        "invoke-blank",
    ],
)
def test_a_runtime_stream_mapping_failure_is_reported_and_flags_only_lost_output(
    make: Any,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failing: tuple[Any, str, int | None] | None,
    events: list[dict[str, Any]],
    kept: str,
    cut: bool,
    reported: type[BaseException] | None,
) -> None:
    if failing is not None:
        owner, name, failing_call = failing
        original = getattr(owner, name)
        calls = 0

        def fail(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if failing_call is None or calls == failing_call:
                raise ValueError("mapper broke")
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, fail)
    errors: list[BaseException] = []
    memory = make(on_error=errors.append)
    client = _client("bedrock-runtime")
    # boto3 generates service methods at runtime, so BaseClient's static type does not have them.
    if operation == "converse_stream":
        _stub_api_call(client, [{"stream": FakeEventStream(events)}], monkeypatch)
        wrap_bedrock(client)
        stream = cast(Any, client).converse_stream(modelId="m", messages=[])["stream"]
    else:
        _stub_api_call(client, [{"body": FakeEventStream(events)}], monkeypatch)
        wrap_bedrock(client)
        stream = cast(Any, client).invoke_model_with_response_stream(
            modelId="anthropic.claude-3-haiku",
            contentType="application/json",
            body=_body({"messages": [], "max_tokens": 10}),
        )["body"]
    assert list(stream) == events

    assert [type(error) for error in _instrumentation_errors(errors)] == (
        [reported] if reported else []
    )
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes.get("telemetry.dev.capture.truncated") is (True if cut else None)
    # Text after a lost delta or chunk is not kept, so the output is a prefix of the stream.
    output = _json_attr(span, "gen_ai.output.messages")
    assert output[0]["parts"][0]["content"] == kept


@pytest.mark.parametrize(
    ("operation", "event_fails"),
    [("Converse", False), ("ConverseStream", False), ("ConverseStream", True)],
    ids=["converse", "converse-stream", "converse-stream-and-event"],
)
def test_a_response_mapping_failure_is_reported_and_ends_the_span(
    make: Any, monkeypatch: pytest.MonkeyPatch, operation: str, event_fails: bool
) -> None:
    def fail(*_args: Any) -> Any:
        raise ValueError("mapper broke")

    if event_fails:
        # A failure of the stream's events after one of its response fields is reported once.
        monkeypatch.setattr(bedrock_streams.ConverseStreamState, "_append", fail)
    operations = SERVICE_OPERATIONS["bedrock-runtime"]
    monkeypatch.setitem(
        operations, operation, dataclasses.replace(operations[operation], response_fields=fail)
    )
    errors: list[BaseException] = []
    memory = make(on_error=errors.append)
    client = _client("bedrock-runtime")
    metadata = {"RequestId": "req-1", "HTTPStatusCode": 200}
    events = [
        {"messageStart": {"role": "assistant"}},
        _converse_delta("Hello"),
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    response: dict[str, Any] = (
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello"}]}},
            "stopReason": "end_turn",
            "ResponseMetadata": metadata,
        }
        if operation == "Converse"
        else {"stream": FakeEventStream(events), "ResponseMetadata": metadata}
    )
    _stub_api_call(client, [response], monkeypatch)
    wrap_bedrock(client)

    # boto3 generates service methods at runtime, so BaseClient's static type does not have them.
    if operation == "Converse":
        cast(Any, client).converse(modelId="m", messages=[])
    else:
        assert list(cast(Any, client).converse_stream(modelId="m", messages=[])["stream"]) == events

    assert [str(error) for error in _instrumentation_errors(errors)] == ["mapper broke"]
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.status.status_code is not StatusCode.ERROR
    assert span.attributes["gen_ai.response.id"] == "req-1"
    if operation == "Converse":
        # The call succeeded, but its response fields and output are lost.
        assert span.attributes["telemetry.dev.capture.truncated"] is True
        assert "gen_ai.output.messages" not in span.attributes
    elif event_fails:
        assert span.attributes["telemetry.dev.capture.truncated"] is True
    else:
        # A stream records its output from its events, so only the response's own fields are lost.
        assert "telemetry.dev.capture.truncated" not in span.attributes
        output = _json_attr(span, "gen_ai.output.messages")
        assert output[0]["parts"][0]["content"] == "Hello"


@pytest.mark.parametrize(
    ("raw", "reported"),
    [
        (b'{"completion": "\xff"}', UnicodeDecodeError),
        (b'{"completion": "cut', json.JSONDecodeError),
        # A blank body holds no output, so nothing is lost.
        (b"   ", None),
    ],
    ids=["not-utf-8", "not-json", "blank"],
)
def test_an_invoke_model_body_that_fails_to_map_is_reported_and_flagged(
    make: Any, monkeypatch: pytest.MonkeyPatch, raw: bytes, reported: type[BaseException] | None
) -> None:
    errors: list[BaseException] = []
    memory = make(on_error=errors.append)
    client = _client("bedrock-runtime")
    _stub_api_call(
        client,
        [{"body": StreamingBody(io.BytesIO(raw), len(raw)), "contentType": "application/json"}],
        monkeypatch,
    )
    wrap_bedrock(client)

    # boto3 generates service methods at runtime, so BaseClient's static type does not have them.
    response = cast(Any, client).invoke_model(
        modelId="anthropic.claude-3-haiku",
        contentType="application/json",
        body=_body({"messages": [], "max_tokens": 10}),
    )
    assert response["body"].read() == raw

    assert [type(error) for error in _instrumentation_errors(errors)] == (
        [reported] if reported else []
    )
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes.get("telemetry.dev.capture.truncated") is (True if reported else None)
    assert "gen_ai.output.messages" not in span.attributes


def _pii_assessment(match: str) -> dict[str, Any]:
    return {
        "sensitiveInformationPolicy": {
            "piiEntities": [{"match": match, "type": "EMAIL", "action": "NONE"}]
        }
    }


def _agent_trace_events(prompt: str) -> list[dict[str, Any]]:
    # Content markers end in _TEXT; UNKNOWN_SECRET sits only in fields the trace shape doesn't know.
    unknown = "UNKNOWN_SECRET"
    return [
        {
            "trace": {
                "agentId": "agent-1",
                "callerChain": [{"agentAliasArn": "arn:alias"}, {"futureCaller": unknown}],
                "futureEnvelopeField": unknown,
                "trace": {
                    "orchestrationTrace": {
                        "modelInvocationInput": {
                            "traceId": "t-1",
                            "type": "ORCHESTRATION",
                            "text": prompt,
                        },
                        "invocationInput": {
                            "traceId": "t-1",
                            "invocationType": "KNOWLEDGE_BASE",
                            "knowledgeBaseLookupInput": {
                                "text": "QUERY_TEXT",
                                "knowledgeBaseId": "kb-1",
                            },
                        },
                        "modelInvocationOutput": {
                            "traceId": "t-1",
                            "rawResponse": {"content": "MODEL_TEXT"},
                            "metadata": {
                                "usage": {"inputTokens": 3, "outputTokens": 4},
                                "totalTimeMs": 12,
                                "clientRequestId": "req-1",
                                "newContentField": "MODEL_META_TEXT",
                            },
                        },
                        "rationale": {"traceId": "t-1", "text": "RATIONALE_TEXT"},
                        "observation": {
                            "traceId": "t-1",
                            "type": "KNOWLEDGE_BASE",
                            "knowledgeBaseLookupOutput": {
                                "retrievedReferences": [{"content": {"text": "REFERENCE_TEXT"}}]
                            },
                        },
                        "futurePart": {"text": unknown},
                    }
                },
            }
        },
        {
            "trace": {
                "trace": {
                    "preProcessingTrace": {
                        "modelInvocationInput": {
                            "traceId": "pre",
                            "type": "PRE_PROCESSING",
                            "text": "PRE_TEXT",
                        },
                        "modelInvocationOutput": {
                            "traceId": "pre",
                            "parsedResponse": {"isValid": True, "rationale": "PRE_RATIONALE_TEXT"},
                            "metadata": {"usage": {"inputTokens": 10, "outputTokens": 20}},
                        },
                    }
                }
            }
        },
        {
            "trace": {
                "trace": {
                    "postProcessingTrace": {
                        "modelInvocationInput": {
                            "traceId": "post",
                            "type": "POST_PROCESSING",
                            "text": "POST_TEXT",
                        },
                        "modelInvocationOutput": {
                            "traceId": "post",
                            "parsedResponse": {"text": "POST_OUT_TEXT"},
                            "metadata": {"usage": {"inputTokens": 1000, "outputTokens": 2000}},
                        },
                    }
                }
            }
        },
        {
            "trace": {
                "trace": {
                    "routingClassifierTrace": {
                        "modelInvocationInput": {
                            "traceId": "route",
                            "type": "ROUTING",
                            "text": "ROUTE_TEXT",
                        },
                        "invocationInput": {
                            "traceId": "route",
                            "invocationType": "AGENT_COLLABORATOR",
                            "agentCollaboratorInvocationInput": {
                                "input": {"text": "COLLAB_IN_TEXT"}
                            },
                        },
                        "modelInvocationOutput": {
                            "traceId": "route",
                            "rawResponse": {"content": "ROUTE_OUT_TEXT"},
                            "metadata": {"usage": {"inputTokens": 10000, "outputTokens": 20000}},
                        },
                        "observation": {
                            "traceId": "route",
                            "type": "AGENT_COLLABORATOR",
                            "agentCollaboratorInvocationOutput": {
                                "output": {"text": "COLLAB_OUT_TEXT"}
                            },
                        },
                    }
                }
            }
        },
        {
            "trace": {
                "trace": {
                    "customOrchestrationTrace": {
                        "traceId": "custom",
                        "event": {"text": "CUSTOM_TEXT"},
                        "futureField": unknown,
                    }
                }
            }
        },
        {
            "trace": {
                "trace": {
                    "guardrailTrace": {
                        "traceId": "g-1",
                        "action": "NONE",
                        "metadata": {"totalTimeMs": 5, "newContentField": unknown},
                        "inputAssessments": [_pii_assessment("INPUT_PII_TEXT")],
                        "outputAssessments": [_pii_assessment("OUTPUT_PII_TEXT")],
                    }
                }
            }
        },
        {
            "trace": {
                "trace": {
                    "failureTrace": {
                        "traceId": "f-1",
                        "failureCode": 500,
                        "failureReason": "FAILURE_TEXT",
                        "metadata": {"clientRequestId": "req-f", "newContentField": unknown},
                    }
                }
            }
        },
        {
            "trace": {
                "trace": {
                    "futureTrace": {
                        "text": unknown,
                        "guardrailTrace": {"action": unknown},
                        "failureTrace": {"failureCode": 400},
                        "modelInvocationOutput": {
                            "metadata": {"usage": {"inputTokens": 100, "outputTokens": 100}}
                        },
                    }
                }
            }
        },
        {"trace": {"trace": {"$unknown": ["futureTrace", {"text": unknown}]}}},
        {"chunk": {"bytes": b"ANSWER_TEXT"}},
    ]


@pytest.mark.parametrize(
    ("options", "keeps_input", "keeps_output"),
    [
        ({}, True, True),
        ({"capture_input": False}, False, True),
        ({"capture_output": False}, True, False),
        ({"capture_input": False, "capture_output": False}, False, False),
        (
            {
                "mask": lambda value, _context: json.loads(
                    json.dumps(value).replace("_TEXT", "_MASKED")
                )
            },
            False,
            False,
        ),
    ],
    ids=["all on", "input off", "output off", "both off", "masked"],
)
def test_agent_trace_content_follows_the_capture_flags(
    make: Any,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    keeps_input: bool,
    keeps_output: bool,
) -> None:
    memory = make(**options)
    # Withheld content past the stream bound must not spend it.
    prompt = "PROMPT_TEXT" if keeps_input else f"PROMPT_TEXT {'p' * 60_000}"
    events = _agent_trace_events(prompt)
    orchestration, pre, post, routing, custom, guardrail, failure = (
        event["trace"]["trace"] for event in events[:7]
    )
    client = _client("bedrock-agent-runtime")
    _stub_api_call(client, [{"completion": FakeEventStream(events)}], monkeypatch)
    wrap_bedrock(client, capture_agent_trace=True)

    response = client.invoke_agent(  # type: ignore[attr-defined]
        agentId="agent-1", agentAliasId="alias", sessionId="sess-1"
    )
    assert list(response["completion"]) == _agent_trace_events(prompt)

    def present(**fields: Any) -> dict[str, Any]:
        return {key: value for key, value in fields.items() if value is not None}

    def input_part(value: Any, withheld: Any = None) -> Any:
        return value if keeps_input else withheld

    def output_part(value: Any, withheld: Any = None) -> Any:
        return value if keeps_output else withheld

    keeps_failure_reason = keeps_input and keeps_output
    span = memory.span_exporter.get_finished_spans()[0]
    assert "telemetry.dev.capture.truncated" not in span.attributes
    assert span.attributes["gen_ai.usage.input_tokens"] == 3 + 10 + 1000 + 10000
    assert span.attributes["gen_ai.usage.output_tokens"] == 4 + 20 + 2000 + 20000
    assert span.attributes["td.metadata.guardrail_action"] == "NONE"
    assert span.attributes["td.metadata.failure"] == "true"
    assert span.attributes["td.metadata.failure_code"] == "500"
    assert span.attributes.get("td.metadata.failure_reason") == (
        "FAILURE_TEXT" if keeps_failure_reason else None
    )
    orchestration_trace = orchestration["orchestrationTrace"]
    assert _json_attr(span, "td.metadata.agent_trace") == [
        {
            "agentId": "agent-1",
            "callerChain": [{"agentAliasArn": "arn:alias"}, {}],
            "trace": {
                "orchestrationTrace": {
                    "modelInvocationInput": input_part(
                        orchestration_trace["modelInvocationInput"],
                        {"traceId": "t-1", "type": "ORCHESTRATION"},
                    ),
                    "invocationInput": input_part(
                        orchestration_trace["invocationInput"],
                        {"traceId": "t-1", "invocationType": "KNOWLEDGE_BASE"},
                    ),
                    "modelInvocationOutput": output_part(
                        orchestration_trace["modelInvocationOutput"],
                        {
                            "traceId": "t-1",
                            "metadata": {
                                "usage": {"inputTokens": 3, "outputTokens": 4},
                                "totalTimeMs": 12,
                                "clientRequestId": "req-1",
                            },
                        },
                    ),
                    "rationale": output_part(orchestration_trace["rationale"], {"traceId": "t-1"}),
                    "observation": output_part(
                        orchestration_trace["observation"],
                        {"traceId": "t-1", "type": "KNOWLEDGE_BASE"},
                    ),
                }
            },
        },
        {
            "trace": {
                "preProcessingTrace": {
                    "modelInvocationInput": input_part(
                        pre["preProcessingTrace"]["modelInvocationInput"],
                        {"traceId": "pre", "type": "PRE_PROCESSING"},
                    ),
                    "modelInvocationOutput": output_part(
                        pre["preProcessingTrace"]["modelInvocationOutput"],
                        {
                            "traceId": "pre",
                            "metadata": {"usage": {"inputTokens": 10, "outputTokens": 20}},
                        },
                    ),
                }
            }
        },
        {
            "trace": {
                "postProcessingTrace": {
                    "modelInvocationInput": input_part(
                        post["postProcessingTrace"]["modelInvocationInput"],
                        {"traceId": "post", "type": "POST_PROCESSING"},
                    ),
                    "modelInvocationOutput": output_part(
                        post["postProcessingTrace"]["modelInvocationOutput"],
                        {
                            "traceId": "post",
                            "metadata": {"usage": {"inputTokens": 1000, "outputTokens": 2000}},
                        },
                    ),
                }
            }
        },
        {
            "trace": {
                "routingClassifierTrace": {
                    "modelInvocationInput": input_part(
                        routing["routingClassifierTrace"]["modelInvocationInput"],
                        {"traceId": "route", "type": "ROUTING"},
                    ),
                    "invocationInput": input_part(
                        routing["routingClassifierTrace"]["invocationInput"],
                        {"traceId": "route", "invocationType": "AGENT_COLLABORATOR"},
                    ),
                    "modelInvocationOutput": output_part(
                        routing["routingClassifierTrace"]["modelInvocationOutput"],
                        {
                            "traceId": "route",
                            "metadata": {"usage": {"inputTokens": 10000, "outputTokens": 20000}},
                        },
                    ),
                    "observation": output_part(
                        routing["routingClassifierTrace"]["observation"],
                        {"traceId": "route", "type": "AGENT_COLLABORATOR"},
                    ),
                }
            }
        },
        {
            "trace": {
                "customOrchestrationTrace": present(
                    traceId="custom",
                    event=output_part(custom["customOrchestrationTrace"]["event"]),
                )
            }
        },
        {
            "trace": {
                "guardrailTrace": present(
                    traceId="g-1",
                    action="NONE",
                    metadata={"totalTimeMs": 5},
                    inputAssessments=input_part(guardrail["guardrailTrace"]["inputAssessments"]),
                    outputAssessments=output_part(guardrail["guardrailTrace"]["outputAssessments"]),
                )
            }
        },
        {
            "trace": {
                "failureTrace": present(
                    traceId="f-1",
                    failureCode=500,
                    failureReason=failure["failureTrace"]["failureReason"]
                    if keeps_failure_reason
                    else None,
                    metadata={"clientRequestId": "req-f"},
                )
            }
        },
        {"trace": {}},
        {"trace": {}},
    ]
    exported = [str(value) for value in span.attributes.values()]
    assert any("_TEXT" in value for value in exported) is (keeps_input or keeps_output)
    assert not any("UNKNOWN_SECRET" in value for value in exported)


@pytest.mark.parametrize(
    ("options", "reason"),
    [({"mask": lambda value, _context: value}, None), ({}, "FAILURE_TEXT")],
    ids=["masked", "all on"],
)
def test_agent_failure_metadata_follows_the_capture_flags_without_trace_capture(
    make: Any, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], reason: str | None
) -> None:
    memory = make(**options)
    events = [
        {
            "trace": {
                "trace": {"failureTrace": {"failureCode": 500, "failureReason": "FAILURE_TEXT"}}
            }
        }
    ]
    client = _client("bedrock-agent-runtime")
    _stub_api_call(client, [{"completion": FakeEventStream(events)}], monkeypatch)
    wrap_bedrock(client)

    response = client.invoke_agent(  # type: ignore[attr-defined]
        agentId="agent-1", agentAliasId="alias", sessionId="sess-1"
    )
    list(response["completion"])

    span = memory.span_exporter.get_finished_spans()[0]
    assert "td.metadata.agent_trace" not in span.attributes
    assert span.attributes["td.metadata.failure"] == "true"
    assert span.attributes["td.metadata.failure_code"] == "500"
    assert span.attributes.get("td.metadata.failure_reason") == reason


def test_agent_retrieve_rag_flow_and_global_instrument(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-agent-runtime")
    _stub_api_call(
        client,
        [
            {
                "completion": FakeEventStream(
                    [
                        {"chunk": {"bytes": b"hello"}},
                        {
                            "trace": {
                                "trace": {
                                    "orchestrationTrace": {
                                        "modelInvocationOutput": {
                                            "metadata": {
                                                "usage": {"inputTokens": 3, "outputTokens": 4}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        {"returnControl": {"invocationId": "inv-1"}},
                    ]
                ),
                "sessionId": "sess-1",
            },
            {"retrievalResults": [{"content": {"text": "doc"}}], "guardrailAction": "NONE"},
            {"output": {"text": "answer"}, "citations": [{}], "sessionId": "sess-2"},
            {
                "stream": FakeEventStream([{"output": {"text": "rag"}}, {"citation": {}}]),
                "sessionId": "sess-stream",
            },
            {
                "responseStream": FakeEventStream(
                    [
                        {"flowOutputEvent": {"content": {"document": {"value": 1}}}},
                        {
                            "flowMultiTurnInputRequestEvent": {
                                "content": {"document": {"prompt": "more"}}
                            }
                        },
                        {"flowCompletionEvent": {"completionReason": "SUCCESS"}},
                    ]
                )
            },
        ],
        monkeypatch,
    )
    wrap_bedrock(client, capture_agent_trace=True)

    list(
        client.invoke_agent(
            agentId="agent-1", agentAliasId="alias", sessionId="sess-1", inputText="hi"
        )["completion"]
    )  # type: ignore[attr-defined]
    client.retrieve(knowledgeBaseId="kb-1", retrievalQuery={"text": "q"})  # type: ignore[attr-defined]
    client.retrieve_and_generate(input={"text": "q"})  # type: ignore[attr-defined]
    list(client.retrieve_and_generate_stream(input={"text": "q"})["stream"])  # type: ignore[attr-defined]
    list(
        client.invoke_flow(flowIdentifier="flow-1", flowAliasIdentifier="alias", inputs=[])[
            "responseStream"
        ]
    )  # type: ignore[attr-defined]

    spans = memory.span_exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 3
    assert spans[0].attributes["gen_ai.provider.name"] == "amazon-bedrock"
    assert spans[0].attributes["td.metadata.return_control"] == "true"
    assert spans[0].attributes["gen_ai.usage.output_tokens"] == 4
    assert spans[0].attributes["gen_ai.usage.total_tokens"] == 7
    assert spans[1].attributes["td.metadata.citation_count"] == "1"
    assert spans[1].attributes["gen_ai.provider.name"] == "amazon-bedrock"
    assert spans[2].attributes["td.metadata.citation_count"] == "1"
    assert spans[3].attributes["td.metadata.citation_count"] == "1"
    assert spans[3].attributes["td.metadata.bedrock_session_id"] == "sess-stream"
    assert _json_attr(spans[4], "gen_ai.output.messages") == [
        {"document": {"value": 1}},
        {"document": {"prompt": "more"}},
    ]
    assert list(spans[4].attributes["gen_ai.response.finish_reasons"]) == ["SUCCESS"]
    assert spans[4].attributes["gen_ai.provider.name"] == "amazon-bedrock"

    uninstrument_bedrock()
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_api_call(
        self: BaseClient, operation_name: str, api_params: dict[str, Any]
    ) -> dict[str, Any]:
        calls.append((operation_name, dict(api_params)))
        return {"output": {"message": {"content": []}}}

    monkeypatch.setattr(BaseClient, "_make_api_call", fake_api_call)
    instrument_bedrock()
    instrument_bedrock()
    global_client = _client("bedrock-runtime")
    global_client.converse(modelId="m", messages=[])  # type: ignore[attr-defined]
    assert calls == [("Converse", {"modelId": "m", "messages": []})]
    assert len(memory.span_exporter.get_finished_spans()) == 6


def test_global_uninstrument_survives_lifo_wrapper_stacking(
    memory: SimpleNamespace,
) -> None:
    uninstrument_bedrock()
    original_api_call = BaseClient._make_api_call
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_api_call(
        self: BaseClient, operation_name: str, api_params: dict[str, Any]
    ) -> dict[str, Any]:
        calls.append((operation_name, dict(api_params)))
        return {"output": {"message": {"content": []}}}

    try:
        BaseClient._make_api_call = fake_api_call  # type: ignore[method-assign]
        instrument_bedrock()
        bedrock_wrapper = BaseClient._make_api_call

        def outer_wrapper(self: BaseClient, operation_name: str, api_params: dict[str, Any]) -> Any:
            return bedrock_wrapper(self, operation_name, api_params)

        BaseClient._make_api_call = outer_wrapper  # type: ignore[method-assign]
        uninstrument_bedrock()
        BaseClient._make_api_call = bedrock_wrapper  # type: ignore[method-assign]
        uninstrument_bedrock()

        client = _client("bedrock-runtime")
        client.converse(modelId="m", messages=[])  # type: ignore[attr-defined]

        assert calls == [("Converse", {"modelId": "m", "messages": []})]
        assert len(memory.span_exporter.get_finished_spans()) == 0
    finally:
        BaseClient._make_api_call = original_api_call  # type: ignore[method-assign]
        uninstrument_bedrock()


def test_fail_open_without_init(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry_dev.shutdown()
    client = _client("bedrock-runtime")
    _stub_api_call(client, [{"output": {"message": {"content": []}}}], monkeypatch)
    wrap_bedrock(client)
    assert client.converse(modelId="m", messages=[]) == {"output": {"message": {"content": []}}}  # type: ignore[attr-defined]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_does_not_wait_for_an_install_lock_that_a_thread_of_the_parent_held() -> (
    None
):
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        # A private name: the test must hold the real install lock across the fork.
        with bedrock_instrument._install_lock:  # pyright: ignore[reportPrivateUsage]
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
                uninstrument_bedrock()
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
