import type { SpanFields } from "@telemetry-dev/sdk";

import {
  arrayValue,
  awsMetadataFields,
  isRecord,
  mergeFields,
  modeledStreamError,
  numberValue,
  omitUndefined,
  parseJson,
  PROVIDER,
  stringValue,
  type JsonRecord,
  type JsonValue,
  type StreamState,
} from "./internal.ts";

export type MessagePart = object;

export interface NormalizedMessage {
  role: string | undefined;
  parts: MessagePart[];
  finish_reason?: string;
}

export function converseRequestFields(input: JsonRecord): SpanFields {
  const inference = isRecord(input.inferenceConfig) ? input.inferenceConfig : {};
  const guardrail = isRecord(input.guardrailConfig) ? input.guardrailConfig : undefined;

  return omitUndefined({
    provider: PROVIDER,
    model: stringValue(input.modelId),
    input: normalizeMessages(input.messages),
    systemInstructions: normalizeSystem(input.system),
    temperature: numberValue(inference.temperature),
    topP: numberValue(inference.topP),
    maxTokens: numberValue(inference.maxTokens),
    stopSequences: arrayValue<string>(inference.stopSequences),
    metadata: guardrail
      ? omitUndefined({
          guardrail_id: stringValue(guardrail.guardrailIdentifier),
          guardrail_version: stringValue(guardrail.guardrailVersion),
        })
      : undefined,
  });
}

export function converseResponseFields<T>(output: T): SpanFields {
  const result: JsonRecord = isRecord(output) ? output : {};
  const usage = usageFromConverse(result.usage);
  const metrics = isRecord(result.metrics) ? result.metrics : undefined;
  const trace = isRecord(result.trace) ? result.trace : undefined;
  const promptRouter = isRecord(trace?.promptRouter) ? trace?.promptRouter : undefined;
  const guardrail = isRecord(trace?.guardrail) ? trace?.guardrail : undefined;
  const stopReason = stringValue(result.stopReason);
  const message = isRecord(result.output) ? result.output.message : undefined;

  return mergeFields(awsMetadataFields(result.$metadata), {
    output: normalizeOutputMessage(message, stopReason),
    usage,
    finishReason: stopReason,
    responseModel: stringValue(promptRouter?.invokedModelId),
    metadata: omitUndefined({
      server_latency_ms: numberValue(metrics?.latencyMs),
      guardrail_action: stringValue(result.guardrailAction) ?? stringValue(guardrail?.action),
      guardrail_action_reason: stringValue(guardrail?.actionReason),
    }),
  });
}

export function usageFromConverse<T>(value: T): SpanFields["usage"] | undefined {
  if (!isRecord(value)) return undefined;

  return omitUndefined({
    inputTokens: numberValue(value.inputTokens),
    outputTokens: numberValue(value.outputTokens),
    totalTokens: numberValue(value.totalTokens),
    cacheReadInputTokens: numberValue(value.cacheReadInputTokens),
    cacheCreationInputTokens: numberValue(value.cacheWriteInputTokens),
  });
}

export function normalizeMessages<T>(value: T): NormalizedMessage[] | undefined {
  const messages = arrayValue<JsonRecord>(value);

  if (!messages) return undefined;

  return messages.map((message) => ({
    role: stringValue(message.role),
    parts: normalizeContentList(message.content),
  }));
}

export function normalizeOutputMessage<T>(
  value: T,
  finishReason?: string,
): NormalizedMessage[] | undefined {
  if (!isRecord(value)) return undefined;

  const message = omitUndefined({
    role: stringValue(value.role),
    parts: normalizeContentList(value.content),
    finish_reason: finishReason,
  });

  return [message];
}

export function normalizeSystem<T>(value: T): MessagePart[] | undefined {
  const blocks = arrayValue<JsonRecord>(value);

  if (!blocks) return undefined;
  const parts = blocks.flatMap((block) => normalizeContentBlock(block));

  return parts.length > 0 ? parts : undefined;
}

function normalizeContentList<T>(value: T): MessagePart[] {
  const blocks = arrayValue<JsonRecord>(value);

  if (!blocks) return [];

  return blocks.flatMap((block) => normalizeContentBlock(block));
}

function normalizeContentBlock(block: JsonRecord): MessagePart[] {
  const text = stringValue(block.text);

  if (text !== undefined)
    return [
      omitUndefined({
        type: "text",
        content: text,
        citations: arrayValue<unknown>(block.citations),
      }),
    ];

  const toolUse = isRecord(block.toolUse) ? block.toolUse : undefined;

  if (toolUse) {
    return [
      omitUndefined({
        type: "tool_call",
        id: stringValue(toolUse.toolUseId),
        name: stringValue(toolUse.name),
        arguments: toolUse.input,
      }),
    ];
  }

  const toolResult = isRecord(block.toolResult) ? block.toolResult : undefined;

  if (toolResult) {
    return [
      omitUndefined({
        type: "tool_call_response",
        id: stringValue(toolResult.toolUseId),
        response: simplifyToolResult(toolResult.content),
      }),
    ];
  }

  const reasoning = isRecord(block.reasoningContent) ? block.reasoningContent : undefined;

  if (reasoning) {
    const reasoningText = isRecord(reasoning.reasoningText) ? reasoning.reasoningText : undefined;

    return [
      {
        type: "reasoning",
        content:
          stringValue(reasoningText?.text) ?? (reasoning.redactedContent ? "[redacted]" : ""),
      },
    ];
  }

  for (const modality of ["image", "document", "video", "audio"] as const) {
    const media = isRecord(block[modality]) ? block[modality] : undefined;

    if (!media) continue;
    const format = stringValue(media.format);
    const source = isRecord(media.source) ? media.source : undefined;

    if (source?.bytes !== undefined) {
      return [
        omitUndefined({
          type: "blob",
          modality,
          mime_type: format ? `${modality}/${format}` : undefined,
        }),
      ];
    }

    const s3 = isRecord(source?.s3Location) ? source?.s3Location : undefined;
    const uri = stringValue(source?.uri) ?? stringValue(s3?.uri) ?? s3Uri(s3);

    if (uri) return [omitUndefined({ type: "uri", uri, modality, mime_type: format })];
  }

  const citationsContent = isRecord(block.citationsContent) ? block.citationsContent : undefined;

  if (citationsContent) {
    const contentBlocks = arrayValue<JsonRecord>(citationsContent.content) ?? [];
    const citations = arrayValue<JsonValue>(citationsContent.citations);

    return contentBlocks.flatMap((contentBlock) => {
      if (!citations) return normalizeContentBlock(contentBlock);

      return normalizeContentBlock({ ...contentBlock, citations });
    });
  }

  const guardContent = isRecord(block.guardContent) ? block.guardContent : undefined;

  if (guardContent) {
    const guardText = isRecord(guardContent.text) ? guardContent.text : undefined;
    const content = stringValue(guardText?.text) ?? stringValue(guardContent.text);

    if (content) return [{ type: "text", content }];
  }

  try {
    return [{ type: "text", content: JSON.stringify(block) }];
  } catch {
    return [{ type: "text", content: "[unsupported content block]" }];
  }
}

function simplifyToolResult<T>(content: T): JsonValue {
  const blocks = arrayValue<JsonRecord>(content);

  if (!blocks) return isJsonValue(content) ? content : undefined;
  const text = stringValue(blocks[0]?.text);

  if (blocks.length === 1 && text !== undefined) return text;

  return blocks.flatMap((block) => normalizeContentBlock(block)) as JsonValue[];
}

function s3Uri(value: JsonRecord | undefined): string | undefined {
  const bucket = stringValue(value?.bucket);
  const key = stringValue(value?.key);

  return bucket && key ? `s3://${bucket}/${key}` : undefined;
}

interface StreamBlock {
  kind: "text" | "tool" | "reasoning" | "content";
  text?: string;
  toolUseId?: string;
  name?: string;
  input?: string;
  content?: JsonRecord;
  citations?: unknown[];
}

export class ConverseStreamState implements StreamState {
  private role = "assistant";
  private readonly blocks = new Map<number, StreamBlock>();
  private stopReason: string | undefined;
  private usage: SpanFields["usage"];
  private metadata: JsonRecord = {};
  private responseModel: string | undefined;
  private error: Error | undefined;
  private errorFields: SpanFields | undefined;

  feed<T>(event: T): void {
    if (!isRecord(event)) return;
    const streamError = modeledStreamError(event);

    if (streamError) {
      this.error = streamError.error;
      this.errorFields = streamError.fields;

      return;
    }

    const messageStart = isRecord(event.messageStart) ? event.messageStart : undefined;

    if (messageStart) this.role = stringValue(messageStart.role) ?? this.role;

    const blockStart = isRecord(event.contentBlockStart) ? event.contentBlockStart : undefined;

    if (blockStart) {
      const index = numberValue(blockStart.contentBlockIndex) ?? this.blocks.size;
      const start = isRecord(blockStart.start) ? blockStart.start : undefined;
      const toolUse = isRecord(start?.toolUse) ? start?.toolUse : undefined;
      const toolResult = isRecord(start?.toolResult) ? start?.toolResult : undefined;
      const image = isRecord(start?.image) ? start?.image : undefined;
      this.blocks.set(
        index,
        toolUse
          ? {
              kind: "tool",
              toolUseId: stringValue(toolUse.toolUseId),
              name: stringValue(toolUse.name),
              input: "",
            }
          : toolResult
            ? {
                kind: "content",
                content: {
                  toolResult: omitUndefined({
                    toolUseId: stringValue(toolResult.toolUseId),
                    status: stringValue(toolResult.status),
                    content: [],
                  }),
                },
              }
            : image
              ? { kind: "content", content: { image } }
              : { kind: "text", text: "" },
      );
    }

    const deltaEvent = isRecord(event.contentBlockDelta) ? event.contentBlockDelta : undefined;

    if (deltaEvent) {
      const index = numberValue(deltaEvent.contentBlockIndex) ?? 0;
      const delta = isRecord(deltaEvent.delta) ? deltaEvent.delta : {};
      const block = this.blocks.get(index) ?? { kind: "text", text: "" };
      const text = stringValue(delta.text);

      if (text !== undefined) {
        block.kind = "text";
        block.text = `${block.text ?? ""}${text}`;
      }

      const toolUse = isRecord(delta.toolUse) ? delta.toolUse : undefined;
      const input = stringValue(toolUse?.input);

      if (toolUse && input !== undefined) {
        block.kind = "tool";
        block.input = `${block.input ?? ""}${input}`;
      }

      const reasoning = isRecord(delta.reasoningContent) ? delta.reasoningContent : undefined;

      if (reasoning) {
        block.kind = "reasoning";
        block.text = `${block.text ?? ""}${stringValue(reasoning.text) ?? (reasoning.redactedContent ? "[redacted]" : "")}`;
      }

      const image = isRecord(delta.image) ? delta.image : undefined;

      if (image) {
        const content = isRecord(block.content) ? block.content : {};
        const current: JsonRecord = isRecord(content.image) ? content.image : {};
        block.kind = "content";
        block.content = { image: { ...current, ...image } };
      }

      const toolResult = arrayValue<JsonRecord>(delta.toolResult);

      if (toolResult) {
        const content = isRecord(block.content) ? block.content : {};
        const current: JsonRecord = isRecord(content.toolResult) ? content.toolResult : {};
        const currentContent = arrayValue<JsonValue>(current.content) ?? [];
        block.kind = "content";
        block.content = {
          toolResult: {
            ...current,
            content: [...currentContent, ...toolResult],
          },
        };
      }

      if (isRecord(delta.citation)) {
        block.citations = [...(block.citations ?? []), delta.citation];
      }

      this.blocks.set(index, block);
    }

    const stop = isRecord(event.messageStop) ? event.messageStop : undefined;

    if (stop) this.stopReason = stringValue(stop.stopReason) ?? this.stopReason;

    const metadata = isRecord(event.metadata) ? event.metadata : undefined;

    if (metadata) {
      this.usage = usageFromConverse(metadata.usage) ?? this.usage;
      const metrics = isRecord(metadata.metrics) ? metadata.metrics : undefined;
      const trace = isRecord(metadata.trace) ? metadata.trace : undefined;
      const promptRouter = isRecord(trace?.promptRouter) ? trace?.promptRouter : undefined;
      const guardrail = isRecord(trace?.guardrail) ? trace?.guardrail : undefined;
      this.responseModel = stringValue(promptRouter?.invokedModelId) ?? this.responseModel;
      this.metadata = {
        ...this.metadata,
        ...omitUndefined({
          server_latency_ms: numberValue(metrics?.latencyMs),
          guardrail_action: stringValue(guardrail?.action),
          guardrail_action_reason: stringValue(guardrail?.actionReason),
        }),
      };
    }
  }

  finish(partial: boolean): SpanFields {
    const parts = [...this.blocks.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, block]) => blockToPart(block));

    return mergeFields(
      omitUndefined({
        output:
          parts.length > 0
            ? [
                omitUndefined({
                  role: this.role,
                  parts,
                  finish_reason: partial ? undefined : this.stopReason,
                }),
              ]
            : undefined,
        finishReason: partial ? undefined : this.stopReason,
        usage: this.usage,
        responseModel: this.responseModel,
        metadata: Object.keys(this.metadata).length > 0 ? this.metadata : undefined,
        error: this.error,
      }),
      this.errorFields,
    );
  }
}

function isJsonValue<T>(value: T): value is T & JsonValue {
  return (
    value == null ||
    String(value) === value ||
    Number(value) === value ||
    Boolean(value) === value ||
    Array.isArray(value) ||
    isRecord(value)
  );
}

function blockToPart(block: StreamBlock) {
  if (block.kind === "tool") {
    return omitUndefined({
      type: "tool_call",
      id: block.toolUseId,
      name: block.name,
      arguments: parseJson(block.input) ?? block.input ?? "",
    });
  }

  if (block.kind === "reasoning") return { type: "reasoning", content: block.text ?? "" };

  if (block.kind === "content" && block.content)
    return normalizeContentBlock(block.content)[0] ?? {};

  return omitUndefined({
    type: "text",
    content: block.text ?? "",
    citations: block.citations && block.citations.length > 0 ? block.citations : undefined,
  });
}
