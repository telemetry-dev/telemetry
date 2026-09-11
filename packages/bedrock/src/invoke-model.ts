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

export function isEmbeddingModel<T>(modelId: T): boolean {
  const model = stringValue(modelId)?.toLowerCase() ?? "";
  return (
    model.includes("titan-embed") || model.includes("cohere.embed") || model.includes("-embedding")
  );
}

export function invokeModelSpanName(input: JsonRecord): string {
  const model = stringValue(input.modelId) ?? "unknown";
  return `${isEmbeddingModel(model) ? "embeddings" : "chat"} ${model}`;
}

export function invokeModelRequestFields(input: JsonRecord): SpanFields {
  const model = stringValue(input.modelId);
  const body = parseBody(input.body, input.contentType);
  return omitUndefined({
    provider: PROVIDER,
    model,
    input: body,
    outputType: isEmbeddingModel(model) ? "embedding" : undefined,
    ...nativeSamplingFields(model, body),
  });
}

export function invokeModelResponseFields(input: JsonRecord, output: JsonValue): SpanFields {
  const model = stringValue(input.modelId);
  const body = isRecord(output) ? parseBody(output.body, output.contentType) : undefined;
  const metadata = isRecord(output) ? output.$metadata : undefined;
  return mergeFields(awsMetadataFields(metadata), nativeResponseFields(model, body, metadata));
}

export function parseBody<TBody, TContentType>(body: TBody, contentType: TContentType): JsonValue {
  const type = stringValue(contentType) ?? "application/json";
  if (!type.includes("json")) return undefined;
  return parseJson(body);
}

function nativeSamplingFields<T>(model: string | undefined, body: T): SpanFields {
  if (!isRecord(body)) return {};
  const lower = model?.toLowerCase() ?? "";
  if (lower.includes("amazon.titan")) {
    const cfg = isRecord(body.textGenerationConfig) ? body.textGenerationConfig : {};
    return omitUndefined({
      temperature: numberValue(cfg.temperature),
      topP: numberValue(cfg.topP),
      maxTokens: numberValue(cfg.maxTokenCount),
      stopSequences: arrayValue<string>(cfg.stopSequences),
    });
  }
  if (lower.includes("amazon.nova")) {
    const cfg = isRecord(body.inferenceConfig) ? body.inferenceConfig : {};
    return omitUndefined({
      temperature: numberValue(cfg.temperature),
      topP: numberValue(cfg.topP) ?? numberValue(cfg.top_p),
      topK: numberValue(cfg.topK),
      maxTokens: numberValue(cfg.maxTokens) ?? numberValue(cfg.max_new_tokens),
      stopSequences: arrayValue<string>(cfg.stopSequences),
    });
  }
  if (lower.includes("anthropic.claude")) {
    return omitUndefined({
      temperature: numberValue(body.temperature),
      topP: numberValue(body.top_p),
      topK: numberValue(body.top_k),
      maxTokens: numberValue(body.max_tokens),
      stopSequences: arrayValue<string>(body.stop_sequences),
    });
  }
  if (lower.includes("meta.llama")) {
    return omitUndefined({
      temperature: numberValue(body.temperature),
      topP: numberValue(body.top_p),
      maxTokens: numberValue(body.max_gen_len),
    });
  }
  return omitUndefined({
    temperature: numberValue(body.temperature),
    topP: numberValue(body.top_p) ?? numberValue(body.topP),
    maxTokens: numberValue(body.max_tokens) ?? numberValue(body.maxTokens),
  });
}

function nativeResponseFields(
  model: string | undefined,
  body: JsonValue,
  metadata: JsonValue,
): SpanFields {
  if (!isRecord(body)) return headerUsageFields(metadata);
  const lower = model?.toLowerCase() ?? "";
  if (isEmbeddingModel(model)) {
    return withHeaderUsage(
      {
        output: body,
        outputType: "embedding",
        usage: omitUndefined({ inputTokens: numberValue(body.inputTextTokenCount) }),
      },
      metadata,
    );
  }
  if (lower.includes("amazon.titan")) {
    const first = Array.isArray(body.results) && isRecord(body.results[0]) ? body.results[0] : {};
    return omitUndefined({
      output: body,
      finishReason: stringValue(first.completionReason),
      usage: omitUndefined({
        inputTokens: numberValue(body.inputTextTokenCount),
        outputTokens: numberValue(first.tokenCount),
      }),
    });
  }
  if (lower.includes("amazon.nova")) {
    const usage = isRecord(body.usage) ? body.usage : {};
    return omitUndefined({
      output: body,
      finishReason: stringValue(body.stopReason),
      usage: omitUndefined({
        inputTokens: numberValue(usage.inputTokens),
        outputTokens: numberValue(usage.outputTokens),
        totalTokens: numberValue(usage.totalTokens),
      }),
    });
  }
  if (lower.includes("anthropic.claude")) {
    const usage = isRecord(body.usage) ? body.usage : {};
    return withHeaderUsage(
      {
        output: body,
        finishReason: stringValue(body.stop_reason),
        usage: omitUndefined({
          inputTokens: numberValue(usage.input_tokens),
          outputTokens: numberValue(usage.output_tokens),
          cacheReadInputTokens: numberValue(usage.cache_read_input_tokens),
          cacheCreationInputTokens: numberValue(usage.cache_creation_input_tokens),
        }),
      },
      metadata,
    );
  }
  if (lower.includes("meta.llama")) {
    return omitUndefined({
      output: body,
      finishReason: stringValue(body.stop_reason),
      usage: omitUndefined({
        inputTokens: numberValue(body.prompt_token_count),
        outputTokens: numberValue(body.generation_token_count),
      }),
    });
  }
  if (lower.includes("mistral")) {
    const first = Array.isArray(body.outputs) && isRecord(body.outputs[0]) ? body.outputs[0] : {};
    return withHeaderUsage(
      { output: body, finishReason: stringValue(first.stop_reason) },
      metadata,
    );
  }
  if (lower.includes("cohere.command-r")) {
    return withHeaderUsage(
      { output: body, finishReason: stringValue(body.finish_reason) },
      metadata,
    );
  }
  if (lower.includes("cohere.command")) {
    const first =
      Array.isArray(body.generations) && isRecord(body.generations[0]) ? body.generations[0] : {};
    return withHeaderUsage(
      { output: body, finishReason: stringValue(first.finish_reason) },
      metadata,
    );
  }
  return withHeaderUsage({ output: body }, metadata);
}

function withHeaderUsage<T>(fields: SpanFields, metadata: T): SpanFields {
  if (fields.usage && Object.keys(fields.usage).length > 0) return fields;
  return mergeFields(fields, headerUsageFields(metadata));
}

function headerUsageFields<T>(metadata: T): SpanFields {
  const headers = isRecord(metadata)
    ? isRecord(metadata.httpHeaders)
      ? metadata.httpHeaders
      : isRecord(metadata.HTTPHeaders)
        ? metadata.HTTPHeaders
        : undefined
    : undefined;
  const usage = omitUndefined({
    inputTokens: headerNumber(headers?.["x-amzn-bedrock-input-token-count"]),
    outputTokens: headerNumber(headers?.["x-amzn-bedrock-output-token-count"]),
  });
  return { usage: Object.keys(usage).length > 0 ? usage : undefined };
}

function headerNumber<T>(value: T): number | undefined {
  const text = stringValue(value);
  if (text !== undefined) {
    const parsed = Number.parseInt(text, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return numberValue(value);
}

export class InvokeModelStreamState implements StreamState {
  private readonly chunks: unknown[] = [];
  private text = "";
  private usage: SpanFields["usage"];
  private finishReason: string | undefined;
  private error: Error | undefined;
  private errorFields: SpanFields | undefined;
  feed<T>(event: T): boolean {
    if (!isRecord(event)) return false;
    const streamError = modeledStreamError(event);
    if (streamError) {
      this.error = streamError.error;
      this.errorFields = streamError.fields;

      return false;
    }
    const chunk = isRecord(event.chunk) ? event.chunk : undefined;
    const parsed = parseJson(chunk?.bytes);

    if (parsed === undefined) {
      const raw = chunk?.bytes;

      return (typeof raw === "string" || raw instanceof Uint8Array) && raw.length > 0;
    }

    this.chunks.push(parsed);
    this.text += textFromProviderChunk(parsed);
    if (isRecord(parsed)) {
      const delta = isRecord(parsed.delta) ? parsed.delta : undefined;
      const message = isRecord(parsed.message) ? parsed.message : undefined;
      const messageUsage = isRecord(message?.usage) ? message?.usage : undefined;
      const usage = isRecord(parsed.usage) ? parsed.usage : undefined;
      const metrics = isRecord(parsed["amazon-bedrock-invocationMetrics"])
        ? parsed["amazon-bedrock-invocationMetrics"]
        : undefined;
      const outputs =
        Array.isArray(parsed.outputs) && isRecord(parsed.outputs[0])
          ? parsed.outputs[0]
          : undefined;
      const generation =
        Array.isArray(parsed.generations) && isRecord(parsed.generations[0])
          ? parsed.generations[0]
          : undefined;
      this.finishReason =
        stringValue(delta?.stop_reason) ??
        stringValue(parsed.stop_reason) ??
        stringValue(outputs?.stop_reason) ??
        stringValue(generation?.finish_reason) ??
        stringValue(parsed.completionReason) ??
        this.finishReason;
      this.usage = omitUndefined({
        ...this.usage,
        inputTokens:
          numberValue(messageUsage?.input_tokens) ??
          numberValue(metrics?.inputTokenCount) ??
          numberValue(parsed.inputTextTokenCount) ??
          this.usage?.inputTokens,
        outputTokens:
          numberValue(usage?.output_tokens) ??
          numberValue(metrics?.outputTokenCount) ??
          numberValue(parsed.totalOutputTextTokenCount) ??
          this.usage?.outputTokens,
        cacheReadInputTokens:
          numberValue(messageUsage?.cache_read_input_tokens) ?? this.usage?.cacheReadInputTokens,
        cacheCreationInputTokens:
          numberValue(messageUsage?.cache_creation_input_tokens) ??
          this.usage?.cacheCreationInputTokens,
      });
    }

    return providerChunkHasOutput(parsed);
  }

  finish(): SpanFields {
    return mergeFields(
      omitUndefined({
        output: this.text
          ? [
              {
                role: "assistant",
                parts: [{ type: "text", content: this.text }],
                finish_reason: this.finishReason,
              },
            ]
          : this.chunks,
        finishReason: this.finishReason,
        usage: this.usage,
        error: this.error,
      }),
      this.errorFields,
    );
  }
}

function textFromProviderChunk<T>(value: T): string {
  if (!isRecord(value)) return "";
  const delta = isRecord(value.delta) ? value.delta : undefined;
  const contentBlockDelta = isRecord(value.contentBlockDelta)
    ? value.contentBlockDelta
    : isRecord(value.content_block_delta)
      ? value.content_block_delta
      : undefined;
  const nestedDelta = isRecord(contentBlockDelta?.delta) ? contentBlockDelta?.delta : undefined;
  const generation =
    Array.isArray(value.generations) && isRecord(value.generations[0])
      ? value.generations[0]
      : undefined;
  return (
    stringValue(value.outputText) ??
    stringValue(value.generation) ??
    stringValue(value.completion) ??
    stringValue(value.text) ??
    stringValue(delta?.text) ??
    stringValue(nestedDelta?.text) ??
    (Array.isArray(value.outputs) && isRecord(value.outputs[0])
      ? stringValue(value.outputs[0].text)
      : undefined) ??
    stringValue(generation?.text) ??
    ""
  );
}

function providerChunkHasOutput<T>(value: T): boolean {
  if (!isRecord(value)) return false;
  const delta = isRecord(value.delta) ? value.delta : undefined;

  return (
    textFromProviderChunk(value).length > 0 ||
    (typeof delta?.thinking === "string" && delta.thinking.length > 0) ||
    (typeof delta?.partial_json === "string" && delta.partial_json.length > 0)
  );
}
