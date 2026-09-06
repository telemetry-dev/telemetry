from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import litellm
import pytest
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
            id="bounded-stream",
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(content=None),
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


async def test_async_completion_and_stream(memory: SimpleNamespace) -> None:
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
    originals = (llm.completion, llm.acompletion, llm.embedding, llm.aembedding)

    instrument_litellm()
    instrumented = (llm.completion, llm.acompletion, llm.embedding, llm.aembedding)
    assert instrumented != originals
    instrument_litellm()
    assert (
        llm.completion,
        llm.acompletion,
        llm.embedding,
        llm.aembedding,
    ) == instrumented
    llm.completion(model="gpt-4o-mini", messages=CHAT_MESSAGES, mock_response="one")
    assert len(finished_spans(memory)) == 1

    uninstrument_litellm()
    assert (
        llm.completion,
        llm.acompletion,
        llm.embedding,
        llm.aembedding,
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
