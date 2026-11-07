# @telemetry-dev/langchain

LangChain.js and LangGraph.js instrumentation for telemetry.dev. A LangChain callback handler records each run as an OpenTelemetry GenAI span through `@telemetry-dev/sdk`.

Docs: https://telemetry.dev/docs/integrations/langchain-langgraph

## Install

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/langchain @langchain/core
```

Requires `@langchain/core` 1.x and `@telemetry-dev/sdk`, which needs Node.js 20.19.0 or later.

Set `TELEMETRY_DEV_API_KEY` for telemetry export.

## Trace a graph or a chain

Call `init()` once, then give the handler to the run. LangChain hands the handler down to each child run: graph nodes, chains, model calls, tools, and retrievers.

```ts
import { init } from "@telemetry-dev/sdk";
import { TelemetryDevCallbackHandler } from "@telemetry-dev/langchain";

init({ serviceName: "support-agent" });

const agent = graph.compile().withConfig({
  callbacks: [new TelemetryDevCallbackHandler()],
});

await agent.invoke(
  { messages: [{ role: "user", content: "Where is order 48213?" }] },
  { configurable: { thread_id: "thread-1" }, metadata: { userId: "user-1" } },
);
```

`withConfig()` attaches the handler to each run of `agent`. To trace only one run, pass the handler in the config of that run: `agent.invoke(input, { callbacks: [handler] })`.

In a short script, call `await shutdown()` from `@telemetry-dev/sdk` before the process exits. Otherwise the last spans can stay in the export queue.

The handler records nothing until `init()` runs with an API key.

## Spans

| LangChain run               | Span name                   | `gen_ai.operation.name` |
| --------------------------- | --------------------------- | ----------------------- |
| LangGraph graph             | `invoke_agent <graph name>` | `invoke_agent`          |
| Graph node, chain, runnable | the run name                | `function`              |
| Chat model call             | `chat <model>`              | `chat`                  |
| Text completion model call  | `text_completion <model>`   | `text_completion`       |
| Tool                        | `execute_tool <tool name>`  | `execute_tool`          |
| Retriever                   | the retriever name          | `function`              |

The handler does not record runs that LangChain tags `langsmith:hidden`, such as the `__start__` step of LangGraph. Their child runs go under the nearest recorded run.

Each span has `langchain.run.id`, the id of its LangChain run. telemetry.dev uses this attribute to show the span as a LangChain span.

A graph span records the input of the graph. If the state of the graph has a `messages` list, the output of the graph span is the last message, which is the answer of the agent. Otherwise the output is the full state. The node, model, and tool spans record each step.

Model spans record the request model, the provider (from `ls_provider`), temperature, maximum tokens, top-p, stop sequences, the input and output messages, the response model, response id, finish reason, and token usage from `usage_metadata`. Input token counts include cache reads and cache writes. telemetry.dev calculates the cost from the token usage.

Model spans also record the type and name of each tool bound to the model in `gen_ai.tool.definitions`. This attribute does not include the descriptions or schemas of the tools, and the capture options do not change it.

Streamed model calls also record the time to the first chunk and the time between chunks.

## Sessions, users, and metadata

The handler reads these keys from the metadata of a run:

- `thread_id`, `sessionId`, `session_id`, or `conversation_id` sets `gen_ai.conversation.id`. LangGraph copies `configurable.thread_id` into the metadata of each run.
- `userId` or `user_id` sets `user.id`.

The two attributes go on each span of the run. The other metadata keys of the app go on the top span as `td.metadata.<key>`. The handler ignores the keys that LangChain and LangGraph add, such as `ls_*`, `langgraph_*`, and `checkpoint_ns`.

`propagateAttributes()` from `@telemetry-dev/sdk` also operates. Explicit metadata keys on the run take priority.

## Nesting

When a run starts in an active span of the app, for example in `startActiveSpan()` or `observe()` from `@telemetry-dev/sdk`, the top span of the run goes under that span.

The handler sets `awaitHandlers`. Thus LangChain calls the handler before it continues the run, and each span starts in the context of the caller. The handler does only in-memory work, and the SDK exports the spans in the background.

## Capture

The `captureInput`, `captureOutput`, `mask`, and `maxAttributeLength` options of `init()` apply to the content that the handler records. With capture off, spans keep the names, model, usage, and errors. The tool names, the run names, and the metadata of the app are still sent.
