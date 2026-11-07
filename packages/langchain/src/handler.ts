import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import type { DocumentInterface } from "@langchain/core/documents";
import type { Serialized } from "@langchain/core/load/serializable";
import { type BaseMessage, isBaseMessage } from "@langchain/core/messages";
import type { LLMResult } from "@langchain/core/outputs";
import type { ChainValues } from "@langchain/core/utils/types";
import {
  type SpanFields,
  type SpanHandle,
  startSpan,
  type StartSpanOptions,
} from "@telemetry-dev/sdk";

import { completionFields, generationRequest, toolInput, toolOutput } from "./generation.ts";
import { identityAttributes, userMetadata } from "./metadata.ts";
import { plain, plainDocument, plainMessage } from "./plain.ts";

type Metadata = Record<string, unknown>;

// LangChain marks its own plumbing runs (LangGraph's __start__ and channel writes) with this tag.
const HIDDEN_TAG = "langsmith:hidden";

/**
 * Records LangChain and LangGraph runs as telemetry.dev spans: chains and graph nodes, chat and
 * completion model calls, tools, and retrievers. Pass it in `callbacks` of a run config; LangChain
 * hands it down to every child run. Spans export through the client of `init()` from
 * `@telemetry-dev/sdk`, and without one the handler records nothing.
 */
export class TelemetryDevCallbackHandler extends BaseCallbackHandler {
  name = "telemetry_dev";

  // Visible runs by run id, and for each hidden run the id of its nearest visible ancestor.
  private readonly spans = new Map<string, SpanHandle>();
  private readonly hidden = new Map<string, string | undefined>();
  private readonly firstChunks = new Map<string, { startedAt: number; seen: boolean }>();
  private readonly graphs = new Set<string>();

  constructor() {
    // LangChain runs a handler that it does not await in a shared background queue. There the
    // active context is no longer the caller's, and a span would not nest in the caller's span.
    super({ _awaitHandler: true });
  }

  // The declared parameter order of handleChainStart differs from the order LangChain calls it
  // with: (chain, inputs, runId, parentRunId, tags, metadata, runType, runName).
  override handleChainStart(
    chain: Serialized,
    inputs: ChainValues,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Metadata,
    _runType?: string,
    runName?: string,
  ): void {
    if (tags?.includes(HIDDEN_TAG)) {
      this.hidden.set(runId, this.visible(parentRunId));

      return;
    }

    const name = runName ?? lastId(chain) ?? "chain";
    const isGraph = chain?.id?.[0] === "langgraph";

    if (isGraph) this.graphs.add(runId);

    this.start(runId, parentRunId, isGraph ? `invoke_agent ${name}` : name, metadata, {
      type: isGraph ? "agent" : "span",
      agentName: isGraph ? name : undefined,
      input: plain(inputs),
    });
  }

  override handleChainEnd(outputs: ChainValues, runId: string): void {
    this.end(runId, { output: this.graphs.has(runId) ? agentOutput(outputs) : plain(outputs) });
  }

  override handleChainError(error: unknown, runId: string): void {
    this.end(runId, { error });
  }

  override handleChatModelStart(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Metadata,
    _tags?: string[],
    metadata?: Metadata,
  ): void {
    const request = generationRequest(extraParams, metadata);

    this.startGeneration(runId, parentRunId, metadata, {
      ...request,
      name: `chat ${request.model ?? lastId(llm) ?? "unknown"}`,
      input: (messages[0] ?? []).map(plainMessage),
    });
  }

  override handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Metadata,
    _tags?: string[],
    metadata?: Metadata,
  ): void {
    const request = generationRequest(extraParams, metadata);

    this.startGeneration(runId, parentRunId, metadata, {
      ...request,
      name: `text_completion ${request.model ?? lastId(llm) ?? "unknown"}`,
      input: prompts[0],
      attributes: { ...request.attributes, "gen_ai.operation.name": "text_completion" },
    });
  }

  override handleLLMNewToken(token: string, _idx: unknown, runId: string): void {
    const span = this.spans.get(runId);
    const chunks = this.firstChunks.get(runId);

    if (!span || !chunks || token === "") return;

    span.recordOutputChunk();

    if (chunks.seen) return;
    chunks.seen = true;
    span.update({ timeToFirstChunkMs: performance.now() - chunks.startedAt });
  }

  override handleLLMEnd(output: LLMResult, runId: string): void {
    this.end(runId, completionFields(output));
  }

  override handleLLMError(error: unknown, runId: string): void {
    this.end(runId, { error });
  }

  override handleToolStart(
    tool: Serialized,
    input: string,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    metadata?: Metadata,
    runName?: string,
    toolCallId?: string,
  ): void {
    const name = runName ?? lastId(tool) ?? "tool";

    this.start(runId, parentRunId, `execute_tool ${name}`, metadata, {
      type: "tool",
      toolName: name,
      toolCallId,
      input: toolInput(input),
    });
  }

  override handleToolEnd(output: unknown, runId: string): void {
    this.end(runId, { output: toolOutput(output) });
  }

  override handleToolError(error: unknown, runId: string): void {
    this.end(runId, { error });
  }

  override handleRetrieverStart(
    retriever: Serialized,
    query: string,
    runId: string,
    parentRunId?: string,
    _tags?: string[],
    metadata?: Metadata,
    name?: string,
  ): void {
    this.start(runId, parentRunId, name ?? lastId(retriever) ?? "retriever", metadata, {
      input: query,
    });
  }

  override handleRetrieverEnd(documents: DocumentInterface[], runId: string): void {
    this.end(runId, { output: documents.map(plainDocument) });
  }

  override handleRetrieverError(error: unknown, runId: string): void {
    this.end(runId, { error });
  }

  private startGeneration(
    runId: string,
    parentRunId: string | undefined,
    metadata: Metadata | undefined,
    fields: SpanFields & { name: string },
  ): void {
    const { name, ...rest } = fields;
    this.start(runId, parentRunId, name, metadata, { ...rest, type: "generation" });
    this.firstChunks.set(runId, { startedAt: performance.now(), seen: false });
  }

  private start(
    runId: string,
    parentRunId: string | undefined,
    name: string,
    metadata: Metadata | undefined,
    options: StartSpanOptions,
  ): void {
    const parentId = this.visible(parentRunId);
    const parent = parentId === undefined ? undefined : this.spans.get(parentId);

    // LangChain hands metadata down to each child run, so the user's keys go on the top span only.
    const span = startSpan(name, {
      ...options,
      parent,
      metadata: parent ? undefined : userMetadata(metadata),
      // telemetry.dev marks a span with a `langchain.*` attribute as a LangChain span.
      attributes: {
        ...identityAttributes(metadata),
        ...options.attributes,
        "langchain.run.id": runId,
      },
    });

    this.spans.set(runId, span);
  }

  private end(runId: string, fields: SpanFields): void {
    this.hidden.delete(runId);
    this.graphs.delete(runId);
    this.firstChunks.delete(runId);
    const span = this.spans.get(runId);

    if (!span) return;
    this.spans.delete(runId);
    span.end(fields);
  }

  private visible(runId: string | undefined): string | undefined {
    return runId !== undefined && this.hidden.has(runId) ? this.hidden.get(runId) : runId;
  }
}

/**
 * The answer of a graph whose state holds messages: its last message. The whole state holds each
 * step of the run again, and the model, tool, and node spans already record the steps.
 */
function agentOutput(outputs: ChainValues): unknown {
  const messages: unknown = outputs?.["messages"];
  const last: unknown = Array.isArray(messages) ? messages.at(-1) : undefined;

  return isBaseMessage(last) ? [plainMessage(last)] : plain(outputs);
}

function lastId(serialized: Serialized | undefined): string | undefined {
  return serialized?.id?.at(-1);
}
