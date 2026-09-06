from __future__ import annotations

import copy
import gc
import io
import json
from types import SimpleNamespace
from typing import Any, cast

import boto3
import pytest
import telemetry_dev
from botocore.client import BaseClient
from botocore.exceptions import ClientError, EventStreamError
from botocore.response import StreamingBody

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
    assert not hasattr(state, "text")
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
    assert stream_span.attributes["gen_ai.usage.input_tokens"] == 5
    assert stream_span.attributes["gen_ai.usage.output_tokens"] == 2
    assert stream_span.attributes["gen_ai.usage.cache_read.input_tokens"] == 1
    assert list(stream_span.attributes["gen_ai.response.finish_reasons"]) == ["end_turn"]
    assert mistral_span.attributes["gen_ai.request.temperature"] == 0.2
    assert mistral_span.attributes["gen_ai.usage.input_tokens"] == 11
    assert mistral_span.attributes["gen_ai.usage.output_tokens"] == 12
    assert failed_read_span.attributes["gen_ai.response.id"] == "invoke-read-failed"
    assert _json_attr(stream_span, "gen_ai.output.messages")[0]["finish_reason"] == "end_turn"


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
    assert len(memory.span_exporter.get_finished_spans()) == 1


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


def test_invoke_model_stream_cohere_generations_finish(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client("bedrock-runtime")
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
                                        "text": "choice",
                                        "finish_reason": "COMPLETE",
                                        "meta": {
                                            "billed_units": {
                                                "input_tokens": 8,
                                                "output_tokens": 3,
                                            }
                                        },
                                    }
                                )
                            }
                        }
                    ]
                )
            },
        ],
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
    assert _json_attr(cohere_span, "gen_ai.output.messages")[0]["parts"][0]["content"] == "choice"
    assert _json_attr(cohere_span, "gen_ai.output.messages")[0]["finish_reason"] == "COMPLETE"
    assert cohere_span.attributes["gen_ai.usage.input_tokens"] == 8
    assert cohere_span.attributes["gen_ai.usage.output_tokens"] == 3
    assert list(cohere_span.attributes["gen_ai.response.finish_reasons"]) == ["COMPLETE"]


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
