import type { Attributes, AttributeValue } from "@opentelemetry/api";

import { diag, jsonAttr, omitUndefined } from "@telemetry-dev/otel";

import { type CaptureConfig, prepareCaptureValue, truncate } from "./capture.ts";

export type SpanType = "span" | "generation" | "tool" | "agent" | "embedding";

// Operation names are taken verbatim by the ingest's deriveOperation. Plain spans always get an
// explicit "function": without one, the propagated gen_ai.conversation.id would trip the ingest's
// hasGenAi heuristic and misclassify them as "llm".
export const SPAN_TYPE_OPERATIONS = {
  span: "function",
  generation: "chat",
  tool: "execute_tool",
  agent: "invoke_agent",
  embedding: "embeddings",
} satisfies Record<SpanType, string>;

export function inputKeyFor(type: SpanType): string {
  return type === "tool" ? "gen_ai.tool.call.arguments" : "gen_ai.input.messages";
}

export function outputKeyFor(type: SpanType): string {
  return type === "tool" ? "gen_ai.tool.call.result" : "gen_ai.output.messages";
}

export interface TokenUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadInputTokens?: number;
  cacheCreationInputTokens?: number;
  reasoningOutputTokens?: number;
}

export interface GenerationFields {
  model?: string;
  provider?: string;
  systemInstructions?: Parameters<import("./config.ts").MaskFn>[0];
  responseModel?: string;
  responseId?: string;
  usage?: TokenUsage;
  /** Optional client-side cost override; the server computes cost from usage by default. */
  costUsd?: number;
  finishReason?: string;
  outputType?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  stopSequences?: string[];
  seed?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  timeToFirstChunkMs?: number;
}

export interface ToolFields {
  toolName?: string;
  toolCallId?: string;
  toolDescription?: string;
}

export interface AgentFields {
  agentName?: string;
  agentId?: string;
}

export interface SpanFields extends GenerationFields, ToolFields, AgentFields {
  /** Rename the span (update only). */
  name?: string;
  input?: Parameters<import("./config.ts").MaskFn>[0];
  output?: Parameters<import("./config.ts").MaskFn>[0];
  /** Marks the span failed: status ERROR + error.type + an "exception" event. */
  error?: unknown;
  /** td.metadata.<key> on this span only; use propagateAttributes for scope-wide metadata. */
  metadata?: Record<string, Parameters<import("./config.ts").MaskFn>[0]>;
  /** Raw attribute escape hatch, merged last. */
  attributes?: Record<string, AttributeValue>;
}

export interface FieldCaptureConfig extends CaptureConfig {
  captureInput: boolean;
  captureOutput: boolean;
}

const RESERVED_METADATA_KEYS = new Set(["userId", "sessionId", "user_id", "session_id"]);

export function fieldsToAttributes(
  fields: SpanFields,
  type: SpanType,
  cfg: FieldCaptureConfig,
): Attributes {
  const attrs = omitUndefined({
    "gen_ai.request.model": fields.model,
    "gen_ai.provider.name": fields.provider,
    "gen_ai.response.model": fields.responseModel,
    "gen_ai.response.id": fields.responseId,
    "gen_ai.usage.input_tokens": fields.usage?.inputTokens,
    "gen_ai.usage.output_tokens": fields.usage?.outputTokens,
    "gen_ai.usage.total_tokens": fields.usage?.totalTokens,
    "gen_ai.usage.cache_read.input_tokens": fields.usage?.cacheReadInputTokens,
    "gen_ai.usage.cache_creation.input_tokens": fields.usage?.cacheCreationInputTokens,
    "gen_ai.usage.reasoning.output_tokens": fields.usage?.reasoningOutputTokens,
    "gen_ai.usage.cost": fields.costUsd,
    "gen_ai.response.finish_reasons": fields.finishReason ? [fields.finishReason] : undefined,
    "gen_ai.output.type": fields.outputType,
    "gen_ai.request.temperature": fields.temperature,
    "gen_ai.request.top_p": fields.topP,
    "gen_ai.request.top_k": fields.topK,
    "gen_ai.request.max_tokens": fields.maxTokens,
    "gen_ai.request.stop_sequences": fields.stopSequences,
    "gen_ai.request.seed": fields.seed,
    "gen_ai.request.frequency_penalty": fields.frequencyPenalty,
    "gen_ai.request.presence_penalty": fields.presencePenalty,
    // The key carries no "ms" unit suffix, so the ingest reads the value as seconds.
    "gen_ai.response.time_to_first_chunk":
      fields.timeToFirstChunkMs !== undefined ? fields.timeToFirstChunkMs / 1000 : undefined,
    "gen_ai.tool.name": fields.toolName,
    "gen_ai.tool.call.id": fields.toolCallId,
    "gen_ai.tool.description": fields.toolDescription,
    "gen_ai.agent.name": fields.agentName,
    "gen_ai.agent.id": fields.agentId,
  });

  if (fields.systemInstructions !== undefined && cfg.captureInput) {
    const value = prepareCaptureValue("gen_ai.system_instructions", fields.systemInstructions, cfg);
    if (value !== undefined) attrs["gen_ai.system_instructions"] = value;
  }
  if (fields.input !== undefined && cfg.captureInput) {
    const key = inputKeyFor(type);
    const value = prepareCaptureValue(key, fields.input, cfg);
    if (value !== undefined) attrs[key] = value;
  }
  if (fields.output !== undefined && cfg.captureOutput) {
    const key = outputKeyFor(type);
    const value = prepareCaptureValue(key, fields.output, cfg);
    if (value !== undefined) attrs[key] = value;
  }
  if (fields.metadata) {
    for (const [key, value] of Object.entries(fields.metadata)) {
      if (RESERVED_METADATA_KEYS.has(key)) {
        diag.debug(`metadata key "${key}" is reserved; pass it via propagateAttributes instead`);
        continue;
      }
      const attr = typeof value === "string" ? value : jsonAttr(value);
      if (attr !== undefined) {
        attrs[`td.metadata.${key}`] = truncate(attr, cfg.maxAttributeLength);
      }
    }
  }
  if (fields.attributes) {
    Object.assign(attrs, omitUndefined(fields.attributes));
  }
  return attrs;
}
