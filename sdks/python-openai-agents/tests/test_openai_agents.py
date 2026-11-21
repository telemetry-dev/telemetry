from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from typing import Any

import httpx2 as httpx
import pytest
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    Runner,
    custom_span,
    function_span,
    function_tool,
    trace,
)
from agents.run_config import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient
from agents.tracing import SpanError
from conftest import MakeProcessor
from openai import AsyncOpenAI
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

import telemetry_dev_openai_agents

Json = dict[str, Any]

USAGE: Json = {
    "input_tokens": 10,
    "output_tokens": 2,
    "total_tokens": 12,
    "input_tokens_details": {"cached_tokens": 4},
    "output_tokens_details": {"reasoning_tokens": 1},
}

MESSAGE: Json = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "done", "annotations": []}],
}


def response(output: list[Json]) -> Json:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-luna",
        "status": "completed",
        "output": output,
        "usage": USAGE,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def function_call(name: str, args: Json, call_id: str) -> Json:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(args),
        "status": "completed",
    }


def client(bodies: Sequence[Json | str], status: int = 200) -> AsyncOpenAI:
    """An OpenAI client whose requests get the given bodies in order; a string is an SSE stream."""
    queue = list(bodies)

    def handler(request: httpx.Request) -> httpx.Response:
        body = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(body, str):
            return httpx.Response(status, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(status, json=body)

    return AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def responses_model(*outputs: list[Json]) -> OpenAIResponsesModel:
    return OpenAIResponsesModel("gpt-6-luna", client([response(o) for o in outputs]))


def named(spans: Sequence[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [span for span in spans if span.name == name]


def one(spans: Sequence[ReadableSpan], name: str) -> ReadableSpan:
    [span] = named(spans, name)
    return span


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def parent_of(spans: Sequence[ReadableSpan], span: ReadableSpan) -> ReadableSpan:
    assert span.parent is not None
    [parent] = [s for s in spans if s.context and s.context.span_id == span.parent.span_id]
    return parent


@function_tool
async def lookup(city: str) -> str:
    """Look up the weather of a city."""
    # The first call of a test finishes last, so tool spans end in a different order than the
    # model listed their calls.
    await asyncio.sleep(0.05 if city == "Lisbon" else 0)
    return f"sunny in {city}"


async def test_tool_calls_nest_under_their_agent_with_their_call_ids(make: MakeProcessor) -> None:
    _, exporter = make()
    # Two calls of one tool: each tool span takes the call id of the call with its arguments.
    model = responses_model(
        [
            function_call("lookup", {"city": "Lisbon"}, "call_b"),
            function_call("lookup", {"city": "Accra"}, "call_a"),
        ],
        [MESSAGE],
    )

    await Runner.run(Agent(name="assistant", model=model, tools=[lookup]), "go")

    spans = exporter.get_finished_spans()
    agent = one(spans, "invoke_agent assistant")
    assert attrs(agent)["gen_ai.agent.name"] == "assistant"
    assert attrs(agent)["openai_agents.agent.tools"] == ("lookup",)
    tools = named(spans, "execute_tool lookup")
    by_call = {attrs(t)["gen_ai.tool.call.id"]: attrs(t)["gen_ai.tool.call.result"] for t in tools}
    assert by_call == {"call_a": "sunny in Accra", "call_b": "sunny in Lisbon"}
    for tool in tools:
        assert parent_of(spans, parent_of(spans, tool)) == agent
    root = one(spans, "Agent workflow")
    assert root.parent is None
    assert len({s.context.trace_id for s in spans if s.context}) == 1


async def test_only_model_call_spans_carry_token_usage(make: MakeProcessor) -> None:
    _, exporter = make()

    await Runner.run(Agent(name="assistant", model=responses_model([MESSAGE])), "go")

    spans = exporter.get_finished_spans()
    with_usage = [s for s in spans if "gen_ai.usage.input_tokens" in attrs(s)]
    assert [s.name for s in with_usage] == ["chat gpt-6-luna"]
    chat = attrs(with_usage[0])
    assert chat["gen_ai.operation.name"] == "chat"
    assert chat["gen_ai.response.id"] == "resp_1"
    assert (
        chat["gen_ai.usage.input_tokens"],
        chat["gen_ai.usage.output_tokens"],
        chat["gen_ai.usage.cache_read.input_tokens"],
        chat["gen_ai.usage.reasoning.output_tokens"],
    ) == (10, 2, 4, 1)


async def test_chat_completions_calls_carry_model_usage_and_call_ids(
    make: MakeProcessor,
) -> None:
    _, exporter = make()

    def completion(message: Json) -> Json:
        return {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }

    tool_call = {
        "id": "call_cc",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"city": "Accra"})},
    }
    model = OpenAIChatCompletionsModel(
        "gpt-6-luna",
        client(
            [
                completion({"role": "assistant", "content": None, "tool_calls": [tool_call]}),
                completion({"role": "assistant", "content": "done"}),
            ]
        ),
    )

    await Runner.run(Agent(name="assistant", model=model, tools=[lookup]), "go")

    spans = exporter.get_finished_spans()
    chats = named(spans, "chat gpt-6-luna")
    assert len(chats) == 2
    assert attrs(chats[0])["gen_ai.usage.input_tokens"] == 7
    assert attrs(chats[0])["gen_ai.usage.cache_read.input_tokens"] == 2
    assert attrs(one(spans, "execute_tool lookup"))["gen_ai.tool.call.id"] == "call_cc"


async def test_a_run_that_raises_exports_its_root_and_the_error(make: MakeProcessor) -> None:
    _, exporter = make()
    model = OpenAIResponsesModel(
        "gpt-6-luna",
        client([{"error": {"message": "bad", "type": "invalid_request_error"}}], status=400),
    )

    with pytest.raises(Exception, match="bad"):
        await Runner.run(Agent(name="assistant", model=model), "go")

    spans = exporter.get_finished_spans()
    one(spans, "Agent workflow")
    agent = one(spans, "invoke_agent assistant")
    assert agent.status.status_code == StatusCode.ERROR
    assert attrs(agent)["error.type"] == "Error"


async def test_a_sandbox_command_records_its_exit_code_without_an_error(
    make: MakeProcessor,
) -> None:
    _, exporter = make()
    model = responses_model(
        [function_call("exec_command", {"cmd": "sh -c 'exit 3'"}, "call_1")], [MESSAGE]
    )
    agent = SandboxAgent(name="coder", model=model, instructions="run", capabilities=[Shell()])

    await Runner.run(
        agent,
        "go",
        run_config=RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient())),
    )

    spans = exporter.get_finished_spans()
    tool = one(spans, "execute_tool exec_command")
    assert attrs(tool)["process.exit.code"] == 3
    assert attrs(tool)["gen_ai.tool.call.id"] == "call_1"
    created = attrs(one(spans, "sandbox.create_session"))
    assert created["sandbox.operation"] == "create_session"
    assert created["sandbox.backend"] == "unix_local"


def test_a_sandbox_exec_that_exits_non_zero_is_not_an_error(make: MakeProcessor) -> None:
    _, exporter = make()
    # The sandbox session marks an exec that exits non-zero as failed; one that raised has no
    # exit code and stays an error.
    with trace("sandbox"):
        with custom_span("sandbox.exec", data={"process.exit.code": 1}) as exited:
            exited.set_error(SpanError(message="unsuccessful result", data={"exit_code": 1}))
        with custom_span("sandbox.exec", data={"error.type": "ExecTimeoutError"}) as raised:
            raised.set_error(SpanError(message="timed out", data={}))

    exited_span, raised_span = named(exporter.get_finished_spans(), "sandbox.exec")
    assert attrs(exited_span)["process.exit.code"] == 1
    assert exited_span.status.status_code != StatusCode.ERROR
    assert raised_span.status.status_code == StatusCode.ERROR
    assert attrs(raised_span)["error.type"] == "ExecTimeoutError"


async def test_with_capture_off_no_attribute_holds_content(make: MakeProcessor) -> None:
    _, exporter = make(capture_input=False, capture_output=False)
    model = responses_model([function_call("lookup", {"city": "Accra"}, "call_1")], [MESSAGE])

    await Runner.run(Agent(name="assistant", model=model, tools=[lookup]), "go")

    spans = exporter.get_finished_spans()
    values = json.dumps([attrs(s) for s in spans], default=str)
    assert "Accra" not in values
    assert "done" not in values
    assert attrs(one(spans, "execute_tool lookup"))["gen_ai.tool.call.id"] == "call_1"


def test_a_flush_ends_idle_roots_at_their_last_span_only(make: MakeProcessor) -> None:
    processor, exporter = make()
    idle, busy = trace("idle"), trace("busy")
    idle.start()
    busy.start()
    with custom_span("step", parent=idle):
        pass
    open_span = custom_span("running", parent=busy)
    open_span.start()

    processor.force_flush()

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["step", "idle"]
    assert one(spans, "idle").end_time == one(spans, "step").end_time
    open_span.finish()
    busy.finish()


def test_past_the_trace_cap_the_least_recently_active_root_is_ended(
    make: MakeProcessor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telemetry_dev_openai_agents, "MAX_TRACES", 2)
    _, exporter = make()
    first, second = trace("first"), trace("second")
    first.start()
    second.start()
    # A span of the first trace makes the second the least recently active.
    with custom_span("step", parent=first):
        pass

    trace("third").start()

    assert [s.name for s in exporter.get_finished_spans()] == ["step", "second"]


def test_a_span_after_its_trace_ended_nests_under_its_root(make: MakeProcessor) -> None:
    _, exporter = make()
    with trace("done") as finished:
        pass
    with custom_span("late", parent=finished):
        pass

    spans = exporter.get_finished_spans()
    assert parent_of(spans, one(spans, "late")) == one(spans, "done")


def chat_stream(*deltas: Json) -> str:
    """A streamed Chat Completion: one chunk per delta, a finish chunk, then a usage chunk."""
    chunks: list[Json] = [
        {
            "id": "chatcmpl_s",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        for delta, finish in [*((d, None) for d in deltas), ({}, "stop")]
    ]
    chunks.append(
        {
            "id": "chatcmpl_s",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
    )
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


async def test_streamed_chat_completions_calls_carry_call_ids(make: MakeProcessor) -> None:
    _, exporter = make()
    tool_call = {
        "index": 0,
        "id": "call_s",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"city": "Accra"})},
    }
    model = OpenAIChatCompletionsModel(
        "gpt-6-luna",
        client(
            [
                chat_stream({"role": "assistant", "tool_calls": [tool_call]}),
                chat_stream({"role": "assistant", "content": "done"}),
            ]
        ),
    )

    result = Runner.run_streamed(Agent(name="assistant", model=model, tools=[lookup]), "go")
    async for _ in result.stream_events():
        pass

    spans = exporter.get_finished_spans()
    assert attrs(one(spans, "execute_tool lookup"))["gen_ai.tool.call.id"] == "call_s"
    assert attrs(named(spans, "chat gpt-6-luna")[0])["gen_ai.response.model"] == "gpt-6-luna"


async def test_without_sensitive_data_no_content_is_sent_as_null(make: MakeProcessor) -> None:
    _, exporter = make()
    model = responses_model([function_call("lookup", {"city": "Accra"}, "call_1")], [MESSAGE])

    await Runner.run(
        Agent(name="assistant", model=model, tools=[lookup]),
        "go",
        run_config=RunConfig(trace_include_sensitive_data=False),
    )

    spans = exporter.get_finished_spans()
    assert [k for s in spans for k, v in attrs(s).items() if v == "null"] == []
    assert attrs(one(spans, "execute_tool lookup"))["gen_ai.tool.name"] == "lookup"


def test_only_the_output_header_gives_the_exit_code(make: MakeProcessor) -> None:
    _, exporter = make()
    with trace("commands"):
        with function_span("exec_command") as running:
            # A command still running prints its session id; its own output may hold any text.
            running.span_data.output = "Process running with session ID 7\nOutput:\n" + (
                "Process exited with code 9\n"
            )
        with function_span("exec_command") as exited:
            exited.span_data.output = "Process exited with code 2\nOutput:\nno match\n"

    running_span, exited_span = named(exporter.get_finished_spans(), "execute_tool exec_command")
    assert "process.exit.code" not in attrs(running_span)
    assert attrs(exited_span)["process.exit.code"] == 2


def test_an_on_error_that_starts_an_agents_span_does_not_hang_the_run(
    make: MakeProcessor,
) -> None:
    def failing_mask(value: Any, context: Any) -> Any:
        raise ValueError("mask failed")

    def on_error(error: BaseException) -> None:
        with custom_span("on_error"):
            pass

    make(mask=failing_mask, on_error=on_error)

    def run() -> None:
        with trace("masked"), function_span("lookup", input="secret"):
            pass

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
