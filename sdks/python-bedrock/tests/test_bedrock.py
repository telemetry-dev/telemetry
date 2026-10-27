from __future__ import annotations

import copy
import gc
import io
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import boto3
import pytest
import telemetry_dev
from botocore.client import BaseClient
from botocore.exceptions import ClientError, EventStreamError
from botocore.response import StreamingBody

import telemetry_dev_bedrock._streams as bedrock_streams
from telemetry_dev_bedrock import instrument_bedrock, uninstrument_bedrock, wrap_bedrock


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
            "textDocument": {"text": "x" * 1_000},
        },
    }
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
    results = [
        {"index": index, "relevanceScore": 0.5, "document": {"text": "x" * 1_000}}
        for index in range(2_000)
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
    assert len(state.blocks[0]["text"]) < 64 * 1024
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
def test_stream_beyond_the_fixed_bound_is_flagged_and_withheld_from_a_mask(
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
    assert state.budget.max_bytes == bedrock_streams._STREAM_CAPTURE_MAX_BYTES  # pyright: ignore[reportPrivateUsage]
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
def test_invoke_model_streaming_body_bounds_capture_and_preserves_readinto(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
    raw = _body({"completion": "x" * 100_000})
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
    assert budget.truncated is True
    assert budget.bytes_used <= budget.max_bytes
    spans = memory.span_exporter.get_finished_spans()
    assert len(spans) == 1
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


@pytest.mark.parametrize("shape", ["bytes", "items"])
def test_masked_agent_stream_drops_return_control_beyond_its_bound(
    make: Any, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    seen: list[str] = []

    def mask(value: Any, _context: Any) -> Any:
        seen.append(json.dumps(value))
        return value

    memory = make(mask=mask)
    inputs: list[Any] = (
        [{"apiInvocationInput": {"actionGroup": "x" * 60_000}}]
        if shape == "bytes"
        else [{"function": "f"} for _ in range(1_500)]
    )
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

    assert cast(Any, instrumented_stream)._state.return_control is None
    span = memory.span_exporter.get_finished_spans()[0]
    assert span.attributes["telemetry.dev.capture.truncated"] is True
    assert span.attributes["td.metadata.return_control"] == "true"
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
    # Withheld content past the fixed bound must not spend it.
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
