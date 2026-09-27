from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import litellm
import pytest
import telemetry_dev
from litellm.exceptions import MidStreamFallbackError
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

import telemetry_dev_litellm
from telemetry_dev_litellm import instrument_litellm, uninstrument_litellm, wrap_router

llm = cast(Any, litellm)

CHAT_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hi"},
]


def attrs(span: ReadableSpan) -> dict[str, object]:
    return dict(span.attributes or {})


def only_span(env: SimpleNamespace) -> ReadableSpan:
    spans = env.span_exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def json_attr(value: object) -> Any:
    return json.loads(str(value))


def number_attr(value: object) -> int | float:
    assert isinstance(value, int | float)
    return value


def finished_spans(env: SimpleNamespace) -> list[ReadableSpan]:
    return list(env.span_exporter.get_finished_spans())


def test_completion_maps_model_messages_usage_finish_provider_and_sampling(
    memory: SimpleNamespace,
) -> None:
    completion = telemetry_dev_litellm.completion(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        temperature=0.5,
        top_p=1.0,
        top_k=5,
        max_tokens=100,
        seed=7,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        response_format={"type": "json_object"},
        stop=["END"],
        mock_response="Hi",
    )

    assert completion.choices[0].message.content == "Hi"
    span = only_span(memory)
    assert span.name == "chat gpt-4o-mini"
    a = attrs(span)
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o-mini"
    assert a["gen_ai.response.model"] == "gpt-4o-mini"
    assert isinstance(a["gen_ai.response.id"], str)
    assert tuple(cast(Any, a["gen_ai.response.finish_reasons"])) == ("stop",)
    assert a["gen_ai.request.temperature"] == 0.5
    assert a["gen_ai.request.top_p"] == 1.0
    assert a["gen_ai.request.max_tokens"] == 100
    assert tuple(cast(Any, a["gen_ai.request.stop_sequences"])) == ("END",)
    assert a["gen_ai.request.seed"] == 7
    assert a["gen_ai.request.frequency_penalty"] == 0.1
    assert a["gen_ai.request.presence_penalty"] == 0.2
    assert a["gen_ai.request.top_k"] == 5
    assert a["gen_ai.output.type"] == "json"
    assert number_attr(a["gen_ai.usage.input_tokens"]) > 0
    assert number_attr(a["gen_ai.usage.output_tokens"]) > 0
    assert number_attr(a["gen_ai.usage.total_tokens"]) > 0
    assert number_attr(a["gen_ai.usage.cost"]) > 0
    assert json_attr(a["gen_ai.input.messages"]) == CHAT_MESSAGES
    assert json_attr(a["gen_ai.output.messages"]) == [{"content": "Hi", "role": "assistant"}]


def test_completion_tool_calls_preserved(memory: SimpleNamespace) -> None:
    instrument_litellm()
    llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "weather"}],
        mock_tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
            }
        ],
    )

    output = json_attr(attrs(only_span(memory))["gen_ai.output.messages"])
    assert output[0]["content"] == "This is a mock request"
    assert output[0]["tool_calls"] == [
        {
            "function": {"arguments": '{"city":"SF"}', "name": "get_weather"},
            "id": "call_1",
            "type": "function",
        }
    ]
    assert tuple(cast(Any, attrs(only_span(memory))["gen_ai.response.finish_reasons"])) == ("stop",)


def test_completion_multiple_choices(memory: SimpleNamespace) -> None:
    instrument_litellm()
    llm.completion(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        n=2,
        mock_response="A",
    )

    a = attrs(only_span(memory))
    assert len(json_attr(a["gen_ai.output.messages"])) == 2
    assert tuple(cast(Any, a["gen_ai.response.finish_reasons"])) == ("stop", "stop")


def test_completion_metadata_param_maps_to_td_metadata(memory: SimpleNamespace) -> None:
    instrument_litellm()
    llm.completion(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        metadata={"tenant": "acme"},
        mock_response="ok",
    )

    assert attrs(only_span(memory))["td.metadata.tenant"] == "acme"


def test_completion_litellm_metadata_maps_to_td_metadata(memory: SimpleNamespace) -> None:
    instrument_litellm()
    llm.completion(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        litellm_metadata={"tenant": "acme"},
        mock_response="ok",
    )

    assert attrs(only_span(memory))["td.metadata.tenant"] == "acme"


def test_completion_cache_write_tokens_map_to_cache_creation(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_completion(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            id="resp-cache-write",
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message={"content": "Hi", "role": "assistant"},
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4, cache_write_tokens=3),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
            _hidden_params={"response_cost": 0.0},
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    instrument_litellm()

    llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES)

    a = attrs(only_span(memory))
    assert a["gen_ai.usage.cache_read.input_tokens"] == 4
    assert a["gen_ai.usage.cache_creation.input_tokens"] == 3


def test_completion_error_records_error_span(memory: SimpleNamespace) -> None:
    instrument_litellm()

    with pytest.raises(llm.RateLimitError):
        llm.completion(
            model="gpt-4o-mini",
            messages=CHAT_MESSAGES,
            mock_response="litellm.RateLimitError",
        )

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == "RateLimitError"
    assert a["gen_ai.provider.name"] == "openai"


def test_completion_timeout_error(memory: SimpleNamespace) -> None:
    instrument_litellm()

    with pytest.raises(llm.Timeout):
        llm.completion(
            model="gpt-4o-mini",
            messages=CHAT_MESSAGES,
            mock_timeout=True,
            timeout=0.01,
        )

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    assert attrs(span)["error.type"] == "Timeout"


def test_completion_response_mapping_base_exception_escapes(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_completion(*args: Any, **kwargs: Any) -> Any:
        class Response:
            @property
            def choices(self) -> Any:
                raise KeyboardInterrupt

        return Response()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    instrument_litellm()

    with pytest.raises(KeyboardInterrupt):
        llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES)

    span = only_span(memory)
    assert span.status.status_code is StatusCode.ERROR
    assert attrs(span)["error.type"] == "KeyboardInterrupt"


def test_stream_context_body_error_does_not_mark_span_error(memory: SimpleNamespace) -> None:
    instrument_litellm()
    stream = llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        mock_response="Hello world streaming",
    )

    with pytest.raises(ValueError), stream:
        next(stream)
        raise ValueError("consumer error")

    assert only_span(memory).status.status_code is not StatusCode.ERROR


def test_streaming_accumulates_content_ttfc_and_usage(memory: SimpleNamespace) -> None:
    instrument_litellm()
    stream = llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        mock_response="Hello world streaming",
    )

    chunks = list(stream)

    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello world streaming"
    )
    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"content": "Hello world streaming", "role": "assistant"}
    ]
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)
    assert tuple(cast(Any, a["gen_ai.response.finish_reasons"])) == ("stop",)
    assert number_attr(a["gen_ai.usage.input_tokens"]) > 0
    assert number_attr(a["gen_ai.usage.output_tokens"]) > 0


def test_streaming_bounds_retained_chunks_without_dropping_output(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
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
    chunks = [
        SimpleNamespace(
            id=None,
            model=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="x" * 100),
                    finish_reason=None,
                )
            ],
        )
        for _ in range(1100)
    ]
    chunks.extend(
        [
            SimpleNamespace(
                id=None,
                model=None,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(
                            content=None,
                            function_call=SimpleNamespace(name="lookup", arguments=""),
                        ),
                        finish_reason=None,
                    )
                ],
            ),
            SimpleNamespace(
                id=None,
                model=None,
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(
                            content=None,
                            function_call=SimpleNamespace(name=None, arguments='{"city":'),
                        ),
                        finish_reason=None,
                    )
                ],
            ),
        ]
    )
    chunks.append(
        SimpleNamespace(
            id="bounded-stream",
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="tail"),
                    finish_reason="length",
                ),
                SimpleNamespace(
                    index=1,
                    delta=SimpleNamespace(content=None),
                    finish_reason="content_filter",
                ),
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            ),
        )
    )

    def fake_completion(*args: Any, **kwargs: Any) -> Any:
        return iter(chunks)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    instrument_litellm()
    stream = llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True)
    delivered = list(stream)
    budget = stream._budget

    assert delivered == chunks
    assert budget.truncated is True
    assert budget.bytes_used <= budget.max_bytes
    assert len(stream._chunks) < len(chunks)
    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "bounded-stream"
    assert a["gen_ai.response.model"] == "gpt-4o-mini"
    assert tuple(cast(Any, a["gen_ai.response.finish_reasons"])) == (
        "length",
        "content_filter",
    )
    assert a["gen_ai.usage.input_tokens"] == 11
    assert a["gen_ai.usage.output_tokens"] == 7
    assert a["gen_ai.usage.total_tokens"] == 18
    assert len(calls) == 1102


def test_streaming_exposes_litellm_stream_attributes(memory: SimpleNamespace) -> None:
    instrument_litellm()
    stream = llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        mock_response="Attribute stream",
    )

    assert stream.model == "gpt-4o-mini"
    assert stream.custom_llm_provider == "openai"

    chunks = list(stream)

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Attribute stream"
    assert len(finished_spans(memory)) == 1


def test_streaming_iterator_identity_survives_disposable_iterator_close(
    memory: SimpleNamespace,
) -> None:
    instrument_litellm()
    stream = llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        mock_response="Hello world streaming",
    )

    iterator = iter(stream)
    chunks = [next(iterator)]
    if iterator is not stream:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    assert finished_spans(memory) == []
    assert iterator is stream
    chunks.append(next(stream))
    chunks.extend(list(stream))

    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello world streaming"
    )
    assert json_attr(attrs(only_span(memory))["gen_ai.output.messages"]) == [
        {"content": "Hello world streaming", "role": "assistant"}
    ]


def test_completion_positional_sampling_and_stream_args_map_request_attrs(
    memory: SimpleNamespace,
) -> None:
    instrument_litellm()
    stream = llm.completion(
        "gpt-4o-mini",
        [{"role": "user", "content": "stream"}],
        None,
        0.25,
        0.75,
        None,
        True,
        None,
        ["END"],
        None,
        64,
        mock_response="Positional stream",
    )

    chunks = list(stream)

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Positional stream"
    a = attrs(only_span(memory))
    assert a["gen_ai.request.temperature"] == 0.25
    assert a["gen_ai.request.top_p"] == 0.75
    assert a["gen_ai.request.max_tokens"] == 64
    assert tuple(cast(Any, a["gen_ai.request.stop_sequences"])) == ("END",)
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"content": "Positional stream", "role": "assistant"}
    ]
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)


def test_streaming_early_break_ends_span_once_with_partial_output(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    instrument_litellm()
    builder_calls = 0
    original_builder = llm.stream_chunk_builder

    def counting_builder(*args: Any, **kwargs: Any) -> Any:
        nonlocal builder_calls
        builder_calls += 1
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(litellm, "stream_chunk_builder", counting_builder)
    stream = llm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        mock_response="Hello world streaming",
    )

    for chunk in stream:
        assert chunk.choices[0].delta.content == "Hel"
        break
    stream.close()
    stream.close()

    assert builder_calls == 1

    spans = finished_spans(memory)
    assert len(spans) == 1
    assert json_attr(attrs(spans[0])["gen_ai.output.messages"]) == [
        {"content": "Hel", "role": "assistant"}
    ]


def test_streaming_mid_stream_error_records_error(memory: SimpleNamespace) -> None:
    instrument_litellm()
    stream = llm.completion(
        model="anthropic/claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        mock_response="Exception: mock_streaming_error",
    )

    with pytest.raises(MidStreamFallbackError):
        list(stream)

    span = only_span(memory)
    assert span.status.status_code == StatusCode.ERROR
    a = attrs(span)
    assert a["error.type"] == "MidStreamFallbackError"
    assert a["gen_ai.provider.name"] == "anthropic"


async def test_async_completion_and_stream(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps: list[float] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        assert timestamp_ms is not None
        timestamps.append(timestamp_ms)
        return original(handle, timestamp_ms)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    instrument_litellm()
    completion = await llm.acompletion(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        mock_response="async ok",
    )
    stream = await llm.acompletion(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        stream=True,
        mock_response="Async stream",
    )
    chunks = [chunk async for chunk in stream]

    assert completion.choices[0].message.content == "async ok"
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Async stream"
    non_stream_span, stream_span = finished_spans(memory)
    assert attrs(non_stream_span)["gen_ai.response.model"] == "gpt-4o-mini"
    assert json_attr(attrs(stream_span)["gen_ai.output.messages"]) == [
        {"content": "Async stream", "role": "assistant"}
    ]
    assert timestamps


async def test_async_streaming_bounds_retained_chunks_without_dropping_output(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [
        SimpleNamespace(
            id=None,
            model=None,
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content="x" * 100),
                    finish_reason=None,
                )
            ],
        )
        for _ in range(1100)
    ]
    chunks.append(
        SimpleNamespace(
            id="bounded-async-stream",
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=13,
                completion_tokens=8,
                total_tokens=21,
            ),
        )
    )

    class AsyncChunks:
        def __init__(self) -> None:
            self._iterator = iter(chunks)

        def __aiter__(self) -> AsyncChunks:
            return self

        async def __anext__(self) -> Any:
            try:
                return next(self._iterator)
            except StopIteration as error:
                raise StopAsyncIteration from error

        async def aclose(self) -> None:
            return None

    async def fake_acompletion(*args: Any, **kwargs: Any) -> Any:
        return AsyncChunks()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    instrument_litellm()
    stream = await llm.acompletion(model="gpt-4o-mini", messages=CHAT_MESSAGES, stream=True)
    delivered = [chunk async for chunk in stream]
    budget = stream._budget

    assert delivered == chunks
    assert budget.truncated is True
    assert budget.bytes_used <= budget.max_bytes
    assert len(stream._chunks) < len(chunks)
    a = attrs(only_span(memory))
    assert a["gen_ai.response.id"] == "bounded-async-stream"
    assert a["gen_ai.response.model"] == "gpt-4o-mini"
    assert tuple(cast(Any, a["gen_ai.response.finish_reasons"])) == ("stop",)
    assert a["gen_ai.usage.input_tokens"] == 13
    assert a["gen_ai.usage.output_tokens"] == 8
    assert a["gen_ai.usage.total_tokens"] == 21


async def test_async_positional_stream_argument_is_instrumented(memory: SimpleNamespace) -> None:
    instrument_litellm()
    stream = await llm.acompletion(
        "gpt-4o-mini",
        CHAT_MESSAGES,
        None,
        None,
        None,
        None,
        None,
        None,
        True,
        mock_response="Async positional stream",
    )

    chunks = [chunk async for chunk in stream]

    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        == "Async positional stream"
    )
    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"content": "Async positional stream", "role": "assistant"}
    ]
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)


async def test_embedding_and_aembedding_map_usage_without_output(memory: SimpleNamespace) -> None:
    instrument_litellm()
    embedding = llm.embedding(
        model="text-embedding-3-small",
        input="embed me",
        mock_response=[0.1, 0.2, 0.3, 0.4],
    )
    async_embedding = await llm.aembedding(
        model="text-embedding-3-small",
        input="embed async",
        mock_response=[0.5, 0.6, 0.7, 0.8],
    )

    assert embedding.data[0].embedding == [0.1, 0.2, 0.3, 0.4]
    assert async_embedding.data[0].embedding == [0.5, 0.6, 0.7, 0.8]
    spans = finished_spans(memory)
    assert len(spans) == 2
    for span in spans:
        a = attrs(span)
        assert span.name == "embeddings text-embedding-3-small"
        assert a["gen_ai.operation.name"] == "embeddings"
        assert a["gen_ai.request.model"] == "text-embedding-3-small"
        assert a["gen_ai.usage.input_tokens"] == 10
        assert "gen_ai.output.messages" not in a


async def test_rerank_and_arerank_capture_rankings_usage_and_provider(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        {"index": 1, "relevance_score": 0.93, "document": {"text": "second"}},
        {"index": 0, "relevance_score": 0.14, "document": {"text": "first"}},
    ]

    def response(response_id: str, provider: str, search_units: int) -> SimpleNamespace:
        return SimpleNamespace(
            id=response_id,
            results=results,
            meta={
                "billed_units": {"search_units": search_units, "total_tokens": 17},
                "tokens": {"input_tokens": 15, "output_tokens": 2},
            },
            _hidden_params={"custom_llm_provider": provider, "response_cost": 0.004},
        )

    def fake_rerank(
        model: str,
        query: str,
        documents: list[str | dict[str, Any]],
        **kwargs: Any,
    ) -> SimpleNamespace:
        return response("rerank-sync", "cohere", 1)

    async def fake_arerank(
        model: str,
        query: str,
        documents: list[str | dict[str, Any]],
        **kwargs: Any,
    ) -> SimpleNamespace:
        return response("rerank-async", "voyage", 2)

    monkeypatch.setattr(litellm, "rerank", fake_rerank)
    monkeypatch.setattr(litellm, "arerank", fake_arerank)
    documents: list[str | dict[str, Any]] = ["first", {"text": "second", "source": "kb"}]

    sync_response = telemetry_dev_litellm.rerank(
        "cohere/rerank-v3.5",
        "best result",
        documents,
        top_n=2,
        rank_fields=["text"],
        return_documents=True,
        max_tokens_per_doc=256,
        metadata={"tenant": "acme"},
    )
    instrument_litellm()
    async_response = await llm.arerank(
        "voyage/rerank-2.5",
        "best async result",
        documents,
        top_n=1,
        max_chunks_per_doc=3,
    )

    assert sync_response.id == "rerank-sync"
    assert async_response.id == "rerank-async"
    sync_span, async_span = finished_spans(memory)

    for span, expected_model, expected_provider, expected_id, expected_units in (
        (sync_span, "rerank-v3.5", "cohere", "rerank-sync", 1),
        (async_span, "rerank-2.5", "voyage", "rerank-async", 2),
    ):
        a = attrs(span)
        assert span.name == f"rerank {expected_model}"
        assert a["gen_ai.operation.name"] == "rerank"
        assert a["gen_ai.request.model"] == expected_model
        assert a["gen_ai.provider.name"] == expected_provider
        assert a["gen_ai.response.id"] == expected_id
        assert a["gen_ai.usage.input_tokens"] == 15
        assert a["gen_ai.usage.output_tokens"] == 2
        assert a["gen_ai.usage.total_tokens"] == 17
        assert a["gen_ai.usage.cost"] == 0.004
        assert a["td.metadata.result_count"] == 2
        assert a["td.metadata.search_units"] == expected_units
        assert json_attr(a["gen_ai.output.messages"]) == results

    assert attrs(sync_span)["td.metadata.tenant"] == "acme"
    assert json_attr(attrs(sync_span)["gen_ai.input.messages"]) == {
        "query": "best result",
        "documents": documents,
        "top_n": 2,
        "rank_fields": ["text"],
        "return_documents": True,
        "max_tokens_per_doc": 256,
    }
    assert json_attr(attrs(async_span)["gen_ai.input.messages"]) == {
        "query": "best async result",
        "documents": documents,
        "top_n": 1,
        "max_chunks_per_doc": 3,
    }


def test_rerank_bounds_large_results_and_preserves_total_count(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = [
        {"index": index, "relevance_score": 0.5, "document": {"text": "x" * 1_000}}
        for index in range(2_000)
    ]

    def fake_rerank(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(id="rerank-large", results=results, meta={})

    monkeypatch.setattr(litellm, "rerank", fake_rerank)
    instrument_litellm()

    llm.rerank(model="cohere/rerank-v3.5", query="best", documents=["document"])

    a = attrs(only_span(memory))
    assert a["td.metadata.result_count"] == 2_000
    assert a["telemetry.dev.capture.truncated"] is True
    assert "gen_ai.output.messages" not in a


def test_rerank_bounds_large_inputs_without_changing_provider_documents(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = [{"text": "x" * 1_000, "index": index} for index in range(2_000)]
    received: list[Any] = []

    def fake_rerank(**kwargs: Any) -> SimpleNamespace:
        received.append(kwargs["documents"])
        return SimpleNamespace(id="rerank-large-input", results=[], meta={})

    monkeypatch.setattr(litellm, "rerank", fake_rerank)
    instrument_litellm()

    llm.rerank(model="cohere/rerank-v3.5", query="best", documents=documents)

    assert received == [documents]
    a = attrs(only_span(memory))
    assert a["telemetry.dev.capture.truncated"] is True
    assert "gen_ai.input.messages" not in a


def test_rerank_capture_disabled_counts_results_without_iterating(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CountOnlyResults(Sequence[Any]):
        def __len__(self) -> int:
            return 250

        def __getitem__(self, index: int | slice) -> Any:
            raise AssertionError(f"read result {index}")

    env = make(capture_output=False)

    def fake_rerank(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(id="rerank-count", results=CountOnlyResults(), meta={})

    monkeypatch.setattr(litellm, "rerank", fake_rerank)
    instrument_litellm()

    llm.rerank(model="cohere/rerank-v3.5", query="best", documents=["document"])

    a = attrs(only_span(env))
    assert a["td.metadata.result_count"] == 250
    assert "gen_ai.output.messages" not in a


def responses_result(status: str = "completed") -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_123",
        model="gpt-4.1-mini",
        status=status,
        output=[
            {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello"}],
            }
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=5,
            total_tokens=17,
            input_tokens_details=SimpleNamespace(
                cached_tokens=2, text_tokens=7, image_tokens=3, audio_tokens=2
            ),
            output_tokens_details=SimpleNamespace(
                reasoning_tokens=1, text_tokens=4, image_tokens=1, audio_tokens=0
            ),
        ),
        error={"message": "provider rejected response"} if status == "failed" else None,
        _hidden_params={"custom_llm_provider": "openai", "response_cost": 0.002},
    )


async def test_responses_and_aresponses_map_response_shape_and_modalities(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_responses(input: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        return responses_result()

    async def fake_aresponses(input: Any, model: str, **kwargs: Any) -> SimpleNamespace:
        return responses_result()

    monkeypatch.setattr(litellm, "responses", fake_responses)
    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

    telemetry_dev_litellm.responses(
        "Describe this",
        "openai/gpt-4.1-mini",
        instructions="Be brief",
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=64,
        metadata={"tenant": "acme"},
    )
    instrument_litellm()
    await llm.aresponses(input=[{"role": "user", "content": "Async"}], model="gpt-4.1-mini")

    sync_span, async_span = finished_spans(memory)
    sync_attrs = attrs(sync_span)
    assert sync_span.name == "chat gpt-4.1-mini"
    assert sync_attrs["gen_ai.operation.name"] == "chat"
    assert sync_attrs["gen_ai.system_instructions"] == "Be brief"
    assert sync_attrs["gen_ai.request.temperature"] == 0.2
    assert sync_attrs["gen_ai.request.top_p"] == 0.8
    assert sync_attrs["gen_ai.request.max_tokens"] == 64
    assert sync_attrs["td.metadata.tenant"] == "acme"
    assert sync_attrs["gen_ai.response.id"] == "resp_123"
    assert sync_attrs["gen_ai.response.status"] == "completed"
    assert json_attr(sync_attrs["gen_ai.output.messages"])[0]["type"] == "message"
    assert sync_attrs["gen_ai.usage.text.input_tokens"] == 7
    assert sync_attrs["gen_ai.usage.image.input_tokens"] == 3
    assert sync_attrs["gen_ai.usage.audio.input_tokens"] == 2
    assert sync_attrs["gen_ai.usage.text.output_tokens"] == 4
    assert sync_attrs["gen_ai.usage.image.output_tokens"] == 1
    assert sync_attrs["gen_ai.usage.audio.output_tokens"] == 0
    assert attrs(async_span)["gen_ai.response.id"] == "resp_123"


class ResponsesStream:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = iter(events)
        self.closed = False

    def __iter__(self) -> ResponsesStream:
        return self

    def __next__(self) -> SimpleNamespace:
        return next(self._events)

    def close(self) -> None:
        self.closed = True


class AsyncResponsesStream:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self) -> AsyncResponsesStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


class ResponsesTransport:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class AsyncResponsesTransport:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class RealShapedResponsesStream:
    def __init__(self, events: list[SimpleNamespace], response: Any) -> None:
        self._events = iter(events)
        self.response = response

    def __iter__(self) -> RealShapedResponsesStream:
        return self

    def __next__(self) -> SimpleNamespace:
        return next(self._events)

    def __aiter__(self) -> RealShapedResponsesStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class FailingResponsesStream:
    def __init__(self, failure: BaseException, response: Any) -> None:
        self.failure = failure
        self.response = response

    def __iter__(self) -> FailingResponsesStream:
        return self

    def __next__(self) -> SimpleNamespace:
        raise self.failure


class CancellingResponsesStream:
    def __init__(self, failure: asyncio.CancelledError, response: Any) -> None:
        self.failure = failure
        self.response = response

    def __aiter__(self) -> CancellingResponsesStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        raise self.failure


def fake_responses_stream(events: list[SimpleNamespace]) -> Any:
    def fake_responses(*args: Any, **kwargs: Any) -> ResponsesStream:
        return ResponsesStream(events)

    return fake_responses


def test_responses_input_strips_binary_media_and_keeps_remote_references(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "responses", fake_responses_stream([]))
    instrument_litellm()

    list(
        llm.responses(
            input=[
                {
                    "type": "input_audio",
                    "input_audio": {"data": "SECRET_AUDIO", "format": "wav"},
                },
                {"type": "input_image", "image_url": "https://example.com/image.png"},
            ],
            model="gpt-4.1-mini",
            stream=True,
        )
    )

    captured = str(attrs(only_span(memory))["gen_ai.input.messages"])
    assert "SECRET_AUDIO" not in captured
    assert "https://example.com/image.png" in captured


def test_responses_input_capture_disabled_does_not_traverse_input(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnreadableInput(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            raise AssertionError(f"read input key {key}")

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("iterated input")

        def __len__(self) -> int:
            raise AssertionError("measured input")

    env = make(capture_input=False)
    monkeypatch.setattr(litellm, "responses", fake_responses_stream([]))
    instrument_litellm()

    list(llm.responses(input=UnreadableInput(), model="gpt-4.1-mini", stream=True))

    assert "gen_ai.input.messages" not in attrs(only_span(env))


async def test_responses_streams_use_events_and_close_early(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = SimpleNamespace(type="response.completed", response=responses_result())
    sync_inner = ResponsesStream(
        [
            SimpleNamespace(type="response.created", response=responses_result()),
            SimpleNamespace(type="response.output_text.delta", delta="Hel"),
            SimpleNamespace(type="response.output_text.delta", delta="lo"),
            completed,
        ]
    )
    async_inner = AsyncResponsesStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="partial"),
            completed,
        ]
    )

    def fake_responses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return sync_inner

    async def fake_aresponses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return async_inner

    monkeypatch.setattr(litellm, "responses", fake_responses)
    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)
    instrument_litellm()

    stream = llm.responses(input="hi", model="gpt-4.1-mini", stream=True)
    events = list(stream)
    assert [event.type for event in events][-1] == "response.completed"
    assert sync_inner.closed

    async_stream = await llm.aresponses(input="hi", model="gpt-4.1-mini", stream=True)
    event = await async_stream.__anext__()
    assert event.delta == "partial"
    await async_stream.aclose()
    await async_stream.aclose()
    assert async_inner.closed

    completed_span, partial_span = finished_spans(memory)
    assert json_attr(attrs(completed_span)["gen_ai.output.messages"])[0]["type"] == "message"
    assert isinstance(attrs(completed_span)["gen_ai.response.time_to_first_chunk"], float)
    assert json_attr(attrs(partial_span)["gen_ai.output.messages"]) == [
        {"type": "output_text", "text": "partial"}
    ]


def test_responses_stream_capture_disabled_does_not_read_terminal_output(
    make: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TerminalResponse:
        id = "resp_disabled"
        model = "gpt-4.1-mini"
        status = "completed"
        usage = None
        error = None
        _hidden_params = None

        @property
        def output(self) -> object:
            raise AssertionError("capture-disabled output was read")

    env = make(capture_output=False)
    events = [SimpleNamespace(type="response.completed", response=TerminalResponse())]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    assert "gen_ai.output.messages" not in attrs(only_span(env))


def test_responses_stream_retains_added_items_and_sanitized_distinct_partial_shapes(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item={"id": "reason_1", "type": "reasoning", "summary": [{"text": "why"}]},
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item={
                "id": "mcp_1",
                "type": "mcp_call",
                "name": "lookup",
                "server_label": "docs",
            },
        ),
        SimpleNamespace(
            type="response.mcp_call_arguments.delta",
            output_index=1,
            item_id="mcp_1",
            delta='{"query":"telemetry"}',
        ),
        SimpleNamespace(
            type="response.content_part.added",
            output_index=2,
            content_index=0,
            part={"type": "output_audio", "data": "SECRET_AUDIO", "transcript": "hello"},
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=3,
            item={"id": "shell_1", "type": "shell_call", "command": "pwd"},
        ),
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    stream = llm.responses(input="hi", model="gpt-4.1-mini", stream=True)
    next(stream)
    next(stream)
    next(stream)
    next(stream)
    next(stream)
    stream.close()

    output = json_attr(attrs(only_span(memory))["gen_ai.output.messages"])
    captured = json.dumps(output)
    assert "why" in captured
    assert '"arguments": "{\\"query\\":\\"telemetry\\"}"' in captured
    assert "lookup" in captured and "docs" in captured
    assert "hello" in captured and "SECRET_AUDIO" not in captured
    assert "shell_call" in captured and "pwd" in captured


async def test_responses_streams_close_inner_http_response(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AsyncClosingStream(RealShapedResponsesStream):
        def __init__(self, events: list[SimpleNamespace], response: Any) -> None:
            super().__init__(events, response)
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    sync_transport = ResponsesTransport()
    async_transport = AsyncResponsesTransport()
    async_inner = AsyncClosingStream([], async_transport)
    streams = iter(
        [
            RealShapedResponsesStream([], sync_transport),
            async_inner,
        ]
    )

    def fake_responses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return next(streams)

    async def fake_aresponses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return next(streams)

    monkeypatch.setattr(litellm, "responses", fake_responses)
    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)
    instrument_litellm()

    llm.responses(input="sync", model="gpt-4.1-mini", stream=True).close()
    async_stream = await llm.aresponses(input="async", model="gpt-4.1-mini", stream=True)
    await async_stream.aclose()

    assert sync_transport.closed
    assert async_inner.closed
    assert async_transport.closed
    assert len(finished_spans(memory)) == 2


async def test_responses_stream_cleanup_always_closes_nested_transport(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_transport = ResponsesTransport()
    async_transport = AsyncResponsesTransport()
    sync_cleanup_error = RuntimeError("sync iterator cleanup failed")
    async_cleanup_error = RuntimeError("async iterator cleanup failed")

    class SyncStream(RealShapedResponsesStream):
        def close(self) -> None:
            raise sync_cleanup_error

    class AsyncStream(RealShapedResponsesStream):
        async def aclose(self) -> None:
            raise async_cleanup_error

    def fake_responses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return SyncStream([], sync_transport)

    async def fake_aresponses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return AsyncStream([], async_transport)

    monkeypatch.setattr(litellm, "responses", fake_responses)
    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)
    instrument_litellm()

    sync_stream = llm.responses(input="sync", model="gpt-4.1-mini", stream=True)
    with pytest.raises(RuntimeError) as sync_error:
        sync_stream.close()
    assert sync_error.value is sync_cleanup_error
    assert sync_transport.closed

    async_stream = await llm.aresponses(input="async", model="gpt-4.1-mini", stream=True)
    with pytest.raises(RuntimeError) as async_error:
        await async_stream.aclose()
    assert async_error.value is async_cleanup_error
    assert async_transport.closed
    assert len(finished_spans(memory)) == 2


async def test_responses_stream_failures_close_inner_http_response(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingCloseTransport(ResponsesTransport):
        def close(self) -> None:
            super().close()
            raise RuntimeError("sync cleanup failed")

    class FailingAsyncCloseTransport(AsyncResponsesTransport):
        async def aclose(self) -> None:
            await super().aclose()
            raise RuntimeError("async cleanup failed")

    read_failure = RuntimeError("response read failed")
    cancellation = asyncio.CancelledError()
    sync_transport = FailingCloseTransport()
    async_transport = FailingAsyncCloseTransport()

    def fake_responses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return FailingResponsesStream(read_failure, sync_transport)

    async def fake_aresponses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return CancellingResponsesStream(cancellation, async_transport)

    monkeypatch.setattr(litellm, "responses", fake_responses)
    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)
    instrument_litellm()

    sync_stream = llm.responses(input="sync", model="gpt-4.1-mini", stream=True)
    with pytest.raises(RuntimeError) as sync_error:
        next(sync_stream)
    assert sync_error.value is read_failure
    assert sync_transport.closed

    async_stream = await llm.aresponses(input="async", model="gpt-4.1-mini", stream=True)
    with pytest.raises(asyncio.CancelledError) as async_error:
        await async_stream.__anext__()
    assert async_error.value is cancellation
    assert async_transport.closed

    sync_span, async_span = finished_spans(memory)
    assert sync_span.status.status_code is StatusCode.ERROR
    assert async_span.status.status_code is StatusCode.ERROR


def test_responses_stream_only_buffers_bounded_output_text(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk_times: list[float | None] = []
    original = telemetry_dev.SpanHandle.record_output_chunk

    def record_output_chunk(
        handle: telemetry_dev.SpanHandle, timestamp_ms: float | None = None
    ) -> telemetry_dev.SpanHandle:
        chunk_times.append(timestamp_ms)
        return original(handle, timestamp_ms)

    events = [
        SimpleNamespace(type="response.function_call_arguments.delta", delta='{"city":"SF"}'),
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="private reasoning"),
        SimpleNamespace(type="response.output_text.delta", delta="x" * 100_000),
    ]

    def fake_responses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return ResponsesStream(events)

    monkeypatch.setattr(telemetry_dev.SpanHandle, "record_output_chunk", record_output_chunk)
    monkeypatch.setattr(litellm, "responses", fake_responses)
    instrument_litellm()

    list(llm.responses(input="bounded", model="gpt-4.1-mini", stream=True))

    assert len(chunk_times) == 3
    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"type": "function_call", "arguments": '{"city":"SF"}'},
        {"type": "summary_text", "text": "private reasoning"},
    ]
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_accounts_for_small_deltas_incrementally(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted_text_lengths: list[int] = []
    original_accept = telemetry_dev.CaptureBudget.accept

    def accept(budget: telemetry_dev.CaptureBudget, value: object) -> bool:
        if isinstance(value, str):
            accepted_text_lengths.append(len(value))
        elif isinstance(value, Mapping):
            accepted_text_lengths.extend(
                len(item)
                for item in cast(Mapping[object, object], value).values()
                if isinstance(item, str)
            )
        return original_accept(budget, cast(object, value))

    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=0,
            delta="x",
        )
        for _ in range(1_000)
    ]
    monkeypatch.setattr(telemetry_dev.CaptureBudget, "accept", accept)
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="incremental", model="gpt-4.1-mini", stream=True))

    assert sum(accepted_text_lengths) <= 2_000
    assert max(accepted_text_lengths) < 100
    assert json_attr(attrs(only_span(memory))["gen_ai.output.messages"]) == [
        {"type": "output_text", "text": "x" * 1_000}
    ]


def test_responses_stream_accounts_for_annotations_without_remeasuring_text(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted_text_lengths: list[int] = []
    original_accept = telemetry_dev.CaptureBudget.accept

    def accept(budget: telemetry_dev.CaptureBudget, value: object) -> bool:
        if isinstance(value, Mapping):
            accepted_text_lengths.extend(
                len(item)
                for item in cast(Mapping[object, object], value).values()
                if isinstance(item, str)
            )
        return original_accept(budget, cast(object, value))

    text = "x" * 59_000
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=0,
            delta=text,
        ),
        *[
            SimpleNamespace(
                type="response.output_text.annotation.added",
                output_index=0,
                content_index=0,
                annotation_index=index,
                annotation={"type": "url_citation", "url": f"https://example.com/{index}"},
            )
            for index in range(20)
        ],
    ]
    monkeypatch.setattr(telemetry_dev.CaptureBudget, "accept", accept)
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    stream = llm.responses(input="citation", model="gpt-4.1-mini", stream=True)
    next(stream)
    accepted_text_lengths.clear()
    list(stream)

    assert max(accepted_text_lengths) < 100
    output = json_attr(attrs(only_span(memory))["gen_ai.output.messages"])
    assert output[0]["text"] == text
    assert len(output[0]["annotations"]) == 20


def test_responses_stream_rejected_annotation_preserves_text_and_marks_truncated(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "x" * 59_000
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=0,
            delta=text,
        ),
        SimpleNamespace(
            type="response.output_text.annotation.added",
            output_index=0,
            content_index=0,
            annotation_index=0,
            annotation={"type": "url_citation", "url": "https://example.com/" + "c" * 7_000},
        ),
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="citation", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [{"type": "output_text", "text": text}]
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_rejects_sparse_annotation_indexes(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=0,
            delta="answer",
        ),
        SimpleNamespace(
            type="response.output_text.annotation.added",
            output_index=0,
            content_index=0,
            annotation_index=1_000_000_000,
            annotation={"type": "url_citation", "url": "https://example.com"},
        ),
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="citation", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [{"type": "output_text", "text": "answer"}]
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_merges_seeded_textual_event_families(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item={"id": "reason", "type": "reasoning", "summary": []},
        ),
        SimpleNamespace(
            type="response.reasoning_summary_part.added",
            output_index=0,
            summary_index=0,
            part={"type": "summary_text", "text": ""},
        ),
        SimpleNamespace(
            type="response.reasoning_summary_text.delta",
            output_index=0,
            summary_index=0,
            delta="summary",
        ),
        SimpleNamespace(
            type="response.reasoning_text.delta",
            output_index=0,
            content_index=0,
            delta="reasoning",
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item={"id": "message", "type": "message", "content": []},
        ),
        SimpleNamespace(
            type="response.content_part.added",
            output_index=1,
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=1,
            content_index=0,
            delta="answer",
        ),
        SimpleNamespace(
            type="response.output_text.annotation.added",
            output_index=1,
            content_index=0,
            annotation_index=0,
            annotation={"type": "url_citation", "url": "https://example.com"},
        ),
        SimpleNamespace(
            type="response.audio.transcript.delta",
            output_index=1,
            content_index=1,
            delta="spoken",
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=2,
            item={"id": "code", "type": "code_interpreter_call"},
        ),
        SimpleNamespace(
            type="response.code_interpreter_call_code.delta",
            output_index=2,
            delta="print(1)",
        ),
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    stream = llm.responses(input="hi", model="gpt-4.1-mini", stream=True)
    for _ in events:
        next(stream)
    stream.close()

    output = json_attr(attrs(only_span(memory))["gen_ai.output.messages"])
    assert output[0]["summary"] == [{"type": "summary_text", "text": "summary"}]
    assert output[0]["content"] == [{"type": "reasoning_text", "text": "reasoning"}]
    assert output[1]["content"][0]["text"] == "answer"
    assert output[1]["content"][0]["annotations"][0]["url"] == "https://example.com"
    assert output[1]["content"][1]["transcript"] == "spoken"
    assert output[2]["code"] == "print(1)"


def test_bounded_responses_conversion_stops_without_model_dump(memory: SimpleNamespace) -> None:
    class Bomb:
        def __init__(self) -> None:
            self.output = [{"type": "output_text", "text": "x" * 100_000}]
            self.status = "completed"

        def model_dump(self, **kwargs: Any) -> Any:
            raise AssertionError("model_dump must not run")

    fields = vars(telemetry_dev_litellm)["_responses_response"](Bomb())
    assert fields["output"] is None
    assert fields["attributes"]["telemetry.dev.capture.truncated"] is True


def test_bounded_responses_conversion_repeats_shared_acyclic_values(
    memory: SimpleNamespace,
) -> None:
    shared = {"text": "same"}

    captured, budget = vars(telemetry_dev_litellm)["_bounded_responses_native"]([shared, shared])

    assert captured == [{"text": "same"}, {"text": "same"}]
    assert budget.truncated is False


def test_bounded_responses_conversion_normalizes_opaque_values(memory: SimpleNamespace) -> None:
    class Opaque:
        __slots__ = ()

    captured, budget = vars(telemetry_dev_litellm)["_bounded_responses_native"](
        {"opaque": Opaque()}
    )

    assert captured == {"opaque": None}
    assert json.dumps(captured) == '{"opaque": null}'
    assert budget.truncated is False


def test_responses_stream_terminal_item_preserves_truncation_overflow(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = "x" * 70_000
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=index,
            delta=oversized,
        )
        for index in range(1_025)
    ]
    events.extend(
        [
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item={"id": "msg_done", "type": "message", "content": []},
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                output_index=1,
                content_index=0,
                delta="fits",
            ),
        ]
    )
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"id": "msg_done", "type": "message", "content": []},
        {"type": "output_text", "text": "fits"},
    ]
    assert a["telemetry.dev.capture.truncated"] is True


async def test_responses_stream_terminal_items_do_not_erase_untracked_overflow(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_budget = telemetry_dev.CaptureBudget.from_client

    def limited_budget() -> telemetry_dev.CaptureBudget:
        budget = original_budget()
        budget.max_items = 8
        return budget

    def events() -> list[SimpleNamespace]:
        return [
            *[
                SimpleNamespace(
                    type="response.output_text.delta",
                    output_index=index,
                    content_index=0,
                    delta="x" * 70_000,
                )
                for index in range(9)
            ],
            *[
                SimpleNamespace(
                    type="response.output_item.done",
                    output_index=index,
                    item=True,
                )
                for index in range(8)
            ],
        ]

    sync_inner = ResponsesStream(events())
    async_inner = AsyncResponsesStream(events())

    def fake_responses(*args: Any, **kwargs: Any) -> ResponsesStream:
        return sync_inner

    async def fake_aresponses(*args: Any, **kwargs: Any) -> AsyncResponsesStream:
        return async_inner

    monkeypatch.setattr(
        telemetry_dev.CaptureBudget,
        "from_client",
        staticmethod(limited_budget),
    )
    monkeypatch.setattr(litellm, "responses", fake_responses)
    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)
    instrument_litellm()

    list(llm.responses(input="sync", model="gpt-4.1-mini", stream=True))
    async_stream = await llm.aresponses(input="async", model="gpt-4.1-mini", stream=True)
    _ = [event async for event in async_stream]

    sync_span, async_span = finished_spans(memory)
    for span in (sync_span, async_span):
        a = attrs(span)
        assert json_attr(a["gen_ai.output.messages"]) == [True] * 8
        assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_terminal_items_clear_multi_owner_truncation(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = "x" * 70_000
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=output_index,
            content_index=0,
            delta=oversized,
        )
        for output_index in (0, 1)
    ]
    events.extend(
        [
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item={"id": "msg_0", "type": "message", "content": []},
            ),
            SimpleNamespace(
                type="response.output_item.done",
                output_index=1,
                item={"id": "msg_1", "type": "message", "content": []},
            ),
            SimpleNamespace(
                type="response.output_text.delta",
                output_index=2,
                content_index=0,
                delta="fits",
            ),
        ]
    )
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"id": "msg_0", "type": "message", "content": []},
        {"id": "msg_1", "type": "message", "content": []},
        {"type": "output_text", "text": "fits"},
    ]
    assert "telemetry.dev.capture.truncated" not in a


def test_responses_stream_evaluates_later_items_after_rejected_delta(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=0,
            delta="x" * 70_000,
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=1,
            content_index=0,
            delta="fits",
        ),
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [{"type": "output_text", "text": "fits"}]
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_bounds_auxiliary_index_state(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=index,
            content_index=0,
            delta="x",
        )
        for index in range(2_000)
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    stream = llm.responses(input="bounded indexes", model="gpt-4.1-mini", stream=True)
    list(stream)

    retained = (
        len(stream._partial_content)
        + len(stream._partial_calls)
        + len(stream._completed_items)
        + len(stream._truncated_items)
    )
    assert retained <= stream._budget.max_items
    assert stream._retained_item_count == retained
    assert stream._truncated_key_count == sum(
        len(values) for values in stream._truncated_item_keys.values()
    )
    assert len(stream._item_indexes) <= stream._budget.max_items
    assert stream._truncation_overflow is True
    assert attrs(only_span(memory))["telemetry.dev.capture.truncated"] is True


def test_responses_stream_bounds_rejected_content_keys_for_one_owner(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=index,
            delta="x" * 70_000,
        )
        for index in range(2_000)
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    stream = llm.responses(input="bounded keys", model="gpt-4.1-mini", stream=True)
    list(stream)

    owner = ("index", 0)
    assert owner in stream._truncated_items
    assert len(stream._truncated_item_keys[owner]) <= stream._budget.max_items
    assert stream._truncated_key_count == len(stream._truncated_item_keys[owner])
    assert stream._truncation_overflow is True
    assert attrs(only_span(memory))["telemetry.dev.capture.truncated"] is True


def test_responses_stream_charges_one_character_fragments_against_item_budget(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=0,
            content_index=0,
            delta="x",
        )
        for _ in range(2_000)
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    stream = llm.responses(input="bounded fragments", model="gpt-4.1-mini", stream=True)
    assert list(stream) == events

    fragments = stream._partial_content[("index", 0)][("index", 0)]["_fragments"]
    assert len(fragments) <= stream._budget.max_items
    assert stream._budget.items_used <= stream._budget.max_items
    assert attrs(only_span(memory))["telemetry.dev.capture.truncated"] is True


async def test_responses_stream_surrogate_deltas_fail_closed_and_finish(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_event = SimpleNamespace(type="response.output_text.delta", delta="\ud800")
    async_event = SimpleNamespace(type="response.output_text.delta", delta="\udfff")
    sync_inner = ResponsesStream([sync_event])
    async_inner = AsyncResponsesStream([async_event])

    def fake_responses(*args: Any, **kwargs: Any) -> ResponsesStream:
        return sync_inner

    async def fake_aresponses(*args: Any, **kwargs: Any) -> AsyncResponsesStream:
        return async_inner

    monkeypatch.setattr(litellm, "responses", fake_responses)
    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)
    instrument_litellm()

    assert list(llm.responses(input="sync", model="gpt-4.1-mini", stream=True)) == [sync_event]
    async_stream = await llm.aresponses(input="async", model="gpt-4.1-mini", stream=True)
    assert [event async for event in async_stream] == [async_event]

    assert sync_inner.closed
    assert async_inner.closed
    sync_span, async_span = finished_spans(memory)
    for span in (sync_span, async_span):
        assert span.status.status_code is not StatusCode.ERROR
        assert attrs(span)["telemetry.dev.capture.truncated"] is True
        assert "gen_ai.output.messages" not in attrs(span)


def test_responses_stream_content_done_replaces_matching_text_deltas(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=4,
            content_index=7,
            delta="Hel",
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=4,
            content_index=7,
            delta="lo",
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=4,
            content_index=7,
            delta="x" * 100_000,
        ),
        SimpleNamespace(
            type="response.content_part.done",
            output_index=4,
            content_index=7,
            part={"type": "output_text", "text": "Hello"},
        ),
    ]

    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [{"type": "output_text", "text": "Hello"}]
    assert "telemetry.dev.capture.truncated" not in a


def test_responses_stream_content_done_resolves_each_rejected_part(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_budget = telemetry_dev.CaptureBudget.from_client

    def limited_budget() -> telemetry_dev.CaptureBudget:
        budget = original_budget()
        budget.max_bytes = 500
        return budget

    monkeypatch.setattr(
        telemetry_dev.CaptureBudget,
        "from_client",
        staticmethod(limited_budget),
    )
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=4,
            content_index=index,
            delta="x" * 1_000,
        )
        for index in range(2)
    ] + [
        SimpleNamespace(
            type="response.content_part.done",
            output_index=4,
            content_index=index,
            part={"type": "output_text", "text": text},
        )
        for index, text in enumerate(("A", "B"))
    ]
    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"type": "output_text", "text": "A"},
        {"type": "output_text", "text": "B"},
    ]
    assert "telemetry.dev.capture.truncated" not in a


def test_responses_stream_many_completed_items_materialize_output_once(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.output_item.done",
            output_index=index,
            item={"id": f"msg_{index}", "type": "message", "content": []},
        )
        for index in range(100)
    ]
    materializations = 0

    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()
    stream = llm.responses(input="many", model="gpt-4.1-mini", stream=True)
    stream_type = cast(type[Any], type(stream))
    original = stream_type._partial_output

    def counted(stream: Any) -> list[Any]:
        nonlocal materializations
        materializations += 1
        return original(stream)

    monkeypatch.setattr(stream_type, "_partial_output", counted)

    list(stream)

    assert materializations == 1
    assert len(json_attr(attrs(only_span(memory))["gen_ai.output.messages"])) == 100


def test_responses_stream_reconstructs_refusal_and_function_arguments(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            output_index=8,
            item_id="call_asymmetric",
            delta='{"city":',
        ),
        SimpleNamespace(
            type="response.refusal.delta",
            output_index=3,
            content_index=9,
            delta="I cannot",
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            output_index=8,
            item_id="call_asymmetric",
            delta='"SF"}',
        ),
    ]

    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    assert json_attr(attrs(only_span(memory))["gen_ai.output.messages"]) == [
        {"type": "refusal", "refusal": "I cannot"},
        {
            "type": "function_call",
            "id": "call_asymmetric",
            "arguments": '{"city":"SF"}',
        },
    ]


@pytest.mark.parametrize("replacement", ["item", "terminal"])
def test_responses_stream_replacement_resets_exhausted_partial_budget(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    original_budget = telemetry_dev.CaptureBudget.from_client

    def limited_budget() -> telemetry_dev.CaptureBudget:
        budget = original_budget()
        budget.max_bytes = 500
        return budget

    monkeypatch.setattr(
        telemetry_dev.CaptureBudget,
        "from_client",
        staticmethod(limited_budget),
    )
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=6,
            content_index=2,
            delta="partial",
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=6,
            content_index=2,
            delta="x" * 1_000,
        ),
    ]
    expected: list[dict[str, Any]]
    if replacement == "item":
        expected = [{"id": "msg_done", "type": "message", "content": []}]
        events.append(
            SimpleNamespace(
                type="response.output_item.done",
                output_index=6,
                item={"id": "msg_done", "type": "message", "content": []},
            )
        )
    else:
        expected = [{"type": "output_text", "text": "terminal"}]
        terminal = responses_result()
        terminal.output = expected
        events.append(SimpleNamespace(type="response.completed", response=terminal))

    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == expected
    assert "telemetry.dev.capture.truncated" not in a


def test_responses_stream_oversized_terminal_retains_bounded_partial(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_budget = telemetry_dev.CaptureBudget.from_client

    def limited_budget() -> telemetry_dev.CaptureBudget:
        budget = original_budget()
        budget.max_bytes = 500
        return budget

    monkeypatch.setattr(
        telemetry_dev.CaptureBudget,
        "from_client",
        staticmethod(limited_budget),
    )
    terminal = responses_result()
    terminal.output = [{"type": "output_text", "text": "x" * 1_000}]
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            output_index=5,
            content_index=1,
            delta="bounded partial",
        ),
        SimpleNamespace(type="response.completed", response=terminal),
    ]

    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"type": "output_text", "text": "bounded partial"}
    ]
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_stream_keeps_a_prefix_after_a_rejected_delta(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_budget = telemetry_dev.CaptureBudget.from_client

    def limited_budget() -> telemetry_dev.CaptureBudget:
        budget = original_budget()
        budget.max_bytes = 500
        return budget

    monkeypatch.setattr(
        telemetry_dev.CaptureBudget,
        "from_client",
        staticmethod(limited_budget),
    )
    terminal = responses_result()
    terminal.output = [{"type": "output_text", "text": "x" * 1_000}]
    events = [
        SimpleNamespace(
            type="response.output_text.delta", output_index=0, content_index=0, delta=delta
        )
        for delta in ("Hello ", "x" * 1_000, " end.")
    ]
    events.append(SimpleNamespace(type="response.completed", response=terminal))

    monkeypatch.setattr(litellm, "responses", fake_responses_stream(events))
    instrument_litellm()

    list(llm.responses(input="hi", model="gpt-4.1-mini", stream=True))

    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [{"type": "output_text", "text": "Hello "}]
    assert a["telemetry.dev.capture.truncated"] is True


def test_responses_failed_event_marks_span_error(
    memory: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = responses_result("failed")
    streams = iter(
        [
            ResponsesStream([SimpleNamespace(type="response.failed", response=failed)]),
            ResponsesStream(
                [
                    SimpleNamespace(
                        type="error",
                        error=SimpleNamespace(
                            message="stream disconnected", code="transport_error"
                        ),
                    )
                ]
            ),
        ]
    )

    def fake_responses(input: Any, model: str, stream: bool = False, **kwargs: Any) -> Any:
        return next(streams)

    monkeypatch.setattr(litellm, "responses", fake_responses)
    instrument_litellm()
    list(llm.responses(input="fail", model="gpt-4.1-mini", stream=True))
    list(llm.responses(input="error", model="gpt-4.1-mini", stream=True))

    failed_span, error_span = finished_spans(memory)
    assert failed_span.status.status_code is StatusCode.ERROR
    assert attrs(failed_span)["error.type"] == "RuntimeError"
    assert attrs(failed_span)["gen_ai.response.status"] == "failed"
    assert error_span.status.status_code is StatusCode.ERROR
    assert attrs(error_span)["error.type"] == "RuntimeError"
    exception = next(event for event in error_span.events if event.name == "exception")
    assert exception.attributes is not None
    assert exception.attributes["exception.message"] == "stream disconnected (transport_error)"


def test_router_completion_traced(memory: SimpleNamespace) -> None:
    instrument_litellm()
    router = llm.Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "gpt-4o-mini", "mock_response": "routed"},
            }
        ]
    )
    router.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES)

    spans = finished_spans(memory)
    assert len(spans) == 1
    assert attrs(spans[0])["td.metadata.model_group"] == "gpt-4o-mini"

    uninstrument_litellm()
    memory.span_exporter.clear()
    router = wrap_router(
        llm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "gpt-4o-mini", "mock_response": "wrapped"},
                }
            ]
        )
    )
    router.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES)

    spans = finished_spans(memory)
    assert len(spans) == 1
    assert spans[0].name == "chat gpt-4o-mini"

    memory.span_exporter.clear()
    instrument_litellm()
    router = wrap_router(
        llm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "gpt-4o-mini", "mock_response": "nested"},
                }
            ]
        )
    )
    router.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES)

    spans = finished_spans(memory)
    assert len(spans) == 2
    roots = [span for span in spans if span.parent is None]
    children = [span for span in spans if span.parent is not None]
    assert len(roots) == 1
    assert len(children) == 1
    parent = children[0].parent
    root_context = roots[0].context
    assert parent is not None
    assert root_context is not None
    assert parent.span_id == root_context.span_id


async def test_wrap_router_acompletion_positional_stream_is_instrumented(
    memory: SimpleNamespace,
) -> None:
    router = wrap_router(
        llm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "gpt-4o-mini"},
                }
            ]
        )
    )

    stream = await router.acompletion(
        "gpt-4o-mini", CHAT_MESSAGES, True, mock_response="Router async stream"
    )
    chunks = [chunk async for chunk in stream]

    assert (
        "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Router async stream"
    )
    a = attrs(only_span(memory))
    assert json_attr(a["gen_ai.output.messages"]) == [
        {"content": "Router async stream", "role": "assistant"}
    ]
    assert isinstance(a["gen_ai.response.time_to_first_chunk"], float)
    assert tuple(cast(Any, a["gen_ai.response.finish_reasons"])) == ("stop",)


def test_instrument_uninstrument_idempotent_and_restores(memory: SimpleNamespace) -> None:
    originals = (
        llm.completion,
        llm.acompletion,
        llm.embedding,
        llm.aembedding,
        llm.rerank,
        llm.arerank,
        llm.responses,
        llm.aresponses,
    )

    instrument_litellm()
    instrumented = (
        llm.completion,
        llm.acompletion,
        llm.embedding,
        llm.aembedding,
        llm.rerank,
        llm.arerank,
        llm.responses,
        llm.aresponses,
    )
    assert instrumented != originals
    instrument_litellm()
    assert (
        llm.completion,
        llm.acompletion,
        llm.embedding,
        llm.aembedding,
        llm.rerank,
        llm.arerank,
        llm.responses,
        llm.aresponses,
    ) == instrumented
    llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES, mock_response="one")
    assert len(finished_spans(memory)) == 1

    uninstrument_litellm()
    assert (
        llm.completion,
        llm.acompletion,
        llm.embedding,
        llm.aembedding,
        llm.rerank,
        llm.arerank,
        llm.responses,
        llm.aresponses,
    ) == originals
    llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES, mock_response="two")
    assert len(finished_spans(memory)) == 1
    uninstrument_litellm()


def test_uninstrument_does_not_clobber_later_patch(memory: SimpleNamespace) -> None:
    original = llm.completion

    def later_patch(*args: Any, **kwargs: Any) -> str:
        return "later"

    try:
        instrument_litellm()
        llm.completion = later_patch
        uninstrument_litellm()

        assert llm.completion is later_patch

        llm.completion = original
        instrument_litellm()
        llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES, mock_response="wrapped")
        assert len(finished_spans(memory)) == 1

        uninstrument_litellm()
        llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES, mock_response="unwrapped")
        assert len(finished_spans(memory)) == 1
    finally:
        llm.completion = original
        uninstrument_litellm()


def test_wrappers_fail_open_without_telemetry_init() -> None:
    instrument_litellm()
    response = llm.completion(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        mock_response="works without init",
    )

    assert response.choices[0].message.content == "works without init"


def test_provider_attribution_non_openai(memory: SimpleNamespace) -> None:
    instrument_litellm()
    llm.completion(
        model="anthropic/claude-3-5-haiku-20241022",
        messages=CHAT_MESSAGES,
        mock_response="anthropic ok",
    )
    llm.completion(
        model="azure/my-deploy",
        messages=CHAT_MESSAGES,
        mock_response="azure ok",
    )
    llm.completion(
        model="my-azure-deploy",
        messages=CHAT_MESSAGES,
        azure=True,
        mock_response="azure flag ok",
    )
    llm.completion(
        model="gpt-4o-mini",
        messages=CHAT_MESSAGES,
        deployment_id="deployment-from-kwarg",
        mock_response="deployment ok",
    )

    anthropic_span, azure_span, azure_flag_span, deployment_span = finished_spans(memory)
    anthropic_attrs = attrs(anthropic_span)
    azure_attrs = attrs(azure_span)
    azure_flag_attrs = attrs(azure_flag_span)
    deployment_attrs = attrs(deployment_span)
    assert anthropic_attrs["gen_ai.provider.name"] == "anthropic"
    assert anthropic_attrs["gen_ai.request.model"] == "claude-3-5-haiku-20241022"
    assert azure_attrs["gen_ai.provider.name"] == "azure.ai.openai"
    assert azure_attrs["gen_ai.request.model"] == "my-deploy"
    assert azure_flag_attrs["gen_ai.provider.name"] == "azure.ai.openai"
    assert azure_flag_attrs["gen_ai.request.model"] == "my-azure-deploy"
    assert deployment_attrs["gen_ai.provider.name"] == "azure.ai.openai"
    assert deployment_attrs["gen_ai.request.model"] == "deployment-from-kwarg"


def test_cost_maps_to_cost_usd_when_available(memory: SimpleNamespace) -> None:
    instrument_litellm()
    llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES, mock_response="priced")

    assert number_attr(attrs(only_span(memory))["gen_ai.usage.cost"]) > 0
