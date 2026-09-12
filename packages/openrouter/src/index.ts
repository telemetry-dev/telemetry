import {
  startSpan,
  type SpanFields,
  type SpanHandle,
  type StartSpanOptions,
} from "@telemetry-dev/sdk";
import { activeContext } from "@telemetry-dev/otel";
import { Chat } from "@openrouter/sdk/sdk/chat.js";
import { Embeddings } from "@openrouter/sdk/sdk/embeddings.js";
import { Responses } from "@openrouter/sdk/sdk/responses.js";

const WRAPPED = Symbol("telemetry.dev.openrouter.wrapped");
const ORIGINAL = Symbol("telemetry.dev.openrouter.original");
const wrappedClients = new WeakSet<object>();
const CAPTURE_TRUNCATED_ATTRIBUTE = "telemetry.dev.capture.truncated";
const MAX_STREAM_CAPTURE_BYTES = 64 * 1024;
const MAX_STREAM_CAPTURE_ITEMS = 1024;
const MAX_STREAM_CAPTURE_DEPTH = 32;
const STREAM_CAPTURE_ITEM_BYTES = 16;

function jsonEscapedLength(codePoint: number): number {
  if (codePoint === 0x22 || codePoint === 0x5c) return 2;

  if (codePoint >= 0xd800 && codePoint <= 0xdfff) return 6;

  if (codePoint >= 0x20) return 1;

  return codePoint === 0x08 ||
    codePoint === 0x09 ||
    codePoint === 0x0a ||
    codePoint === 0x0c ||
    codePoint === 0x0d
    ? 2
    : 6;
}

class StreamCaptureBudget {
  private bytesUsed = 0;
  private itemsUsed = 0;
  truncated = false;

  accept<T>(value: T): boolean {
    if (this.truncated) return false;

    try {
      const measured = this.measure(
        value,
        MAX_STREAM_CAPTURE_BYTES - this.bytesUsed,
        MAX_STREAM_CAPTURE_ITEMS - this.itemsUsed,
        0,
        new Set(),
      );

      if (!measured) {
        this.truncated = true;

        return false;
      }

      const [byteCount, itemCount] = measured;
      this.bytesUsed += byteCount;
      this.itemsUsed += itemCount;

      return true;
    } catch {
      this.truncated = true;

      return false;
    }
  }

  canAccept<T>(value: T): boolean {
    const candidate = new StreamCaptureBudget();
    candidate.bytesUsed = this.bytesUsed;
    candidate.itemsUsed = this.itemsUsed;
    candidate.truncated = this.truncated;

    return candidate.accept(value);
  }

  replace<T>(value: T): boolean {
    const replacement = new StreamCaptureBudget();

    if (!replacement.accept(value)) {
      this.truncated = true;

      return false;
    }

    this.bytesUsed = replacement.bytesUsed;
    this.itemsUsed = replacement.itemsUsed;
    this.truncated = false;

    return true;
  }

  private measure<T>(
    value: T,
    remainingBytes: number,
    remainingItems: number,
    depth: number,
    seen: Set<object>,
  ): [number, number] | undefined {
    if (
      remainingBytes < STREAM_CAPTURE_ITEM_BYTES ||
      remainingItems <= 0 ||
      depth > MAX_STREAM_CAPTURE_DEPTH
    ) {
      return undefined;
    }

    let byteCount = STREAM_CAPTURE_ITEM_BYTES;
    let itemCount = 1;
    const text = readString(value);

    if (text !== undefined) {
      for (const character of text) {
        const codePoint = character.codePointAt(0) ?? 0;

        const utf8Bytes =
          codePoint <= 0x7f ? 1 : codePoint <= 0x7ff ? 2 : codePoint <= 0xffff ? 3 : 4;

        byteCount += Math.max(utf8Bytes, jsonEscapedLength(codePoint));

        if (byteCount > remainingBytes) return undefined;
      }

      return [byteCount, itemCount];
    }

    const record = asRecord(value);

    if (!record) return [byteCount, itemCount];

    if (seen.has(record)) return [byteCount, itemCount];
    seen.add(record);
    const values = asArray(value);

    if (values) {
      for (const child of values) {
        const measured = this.measure(
          child,
          remainingBytes - byteCount,
          remainingItems - itemCount,
          depth + 1,
          seen,
        );

        if (!measured) return undefined;
        byteCount += measured[0];
        itemCount += measured[1];
      }

      return [byteCount, itemCount];
    }

    for (const key in record) {
      if (!Object.prototype.hasOwnProperty.call(record, key)) continue;

      for (const part of [key, record[key]]) {
        const measured = this.measure(
          part,
          remainingBytes - byteCount,
          remainingItems - itemCount,
          depth + 1,
          seen,
        );

        if (!measured) return undefined;
        byteCount += measured[0];
        itemCount += measured[1];
      }
    }

    return [byteCount, itemCount];
  }
}

type JsonValue = string | number | boolean | null | undefined | JsonValue[] | JsonRecord;

interface JsonRecord {
  [key: string]: JsonValue;
}

type MethodValue = JsonValue | ReadableStream<JsonValue>;

type Method = (...args: never[]) => MethodValue | Promise<MethodValue>;

type WrappedFunction = Method & {
  [WRAPPED]?: true;
  [ORIGINAL]?: Method;
};

interface CallableRecord {
  [key: string]: JsonValue | WrappedFunction;
}

type Operation = "chat" | "responses" | "embeddings";

interface OpenRouterClient {
  chat: object;
  responses: object;
  embeddings: object;
}

interface RequestMapping {
  name: string;
  fields: StartSpanOptions;
}

interface TraceContext {
  end: (fields?: SpanFields) => void;
  span: () => SpanHandle;
  startedAt: number;
  operation: Operation;
  streaming: boolean;
  mapResponse: <T>(response: T) => SpanFields;
}

interface ChatChoiceState {
  role?: string;
  content: string;
  reasoning: string;
  reasoningDetails: JsonValue[];
  refusal: string;
  toolCalls: Map<number, JsonRecord>;
  finishReason?: string;
}

interface ResponsesOutputItemState {
  content: Map<number, JsonRecord>;
  item: JsonRecord;
  summary: Map<number, JsonRecord>;
}

function asRecord<T>(value: T): JsonRecord | undefined {
  if (value === null || value === undefined || value instanceof Function) return undefined;

  return Object(value) === value ? (value as JsonRecord) : undefined;
}

function asArray<T>(value: T): JsonValue[] | undefined {
  return Array.isArray(value) ? (value as JsonValue[]) : undefined;
}

function readString<T>(value: T): string | undefined {
  return String(value) === value ? String(value) : undefined;
}

function readNumber<T>(value: T): number | undefined {
  return Number(value) === value ? Number(value) : undefined;
}

function snakeCase(key: string): string {
  return key.replace(/[A-Z]/g, (character) => `_${character.toLowerCase()}`);
}

function responseNative<T>(value: T, seen = new Map<object, JsonValue>()): JsonValue {
  const record = asRecord(value);

  if (!record) return value as JsonValue;
  const existing = seen.get(record);

  if (existing !== undefined) return existing;
  const values = asArray(value);

  if (values) {
    const result: JsonValue[] = [];
    seen.set(record, result);

    for (const item of values) result.push(responseNative(item, seen));

    return result;
  }

  const raw = asRecord(record.raw);

  if (
    raw &&
    (record.isUnknown === true || record.is_unknown === true || record.type === "UNKNOWN")
  ) {
    return responseNative(raw, seen);
  }

  const result: JsonRecord = {};
  seen.set(record, result);

  for (const key in record) {
    if (!Object.prototype.hasOwnProperty.call(record, key)) continue;
    result[snakeCase(key)] = responseNative(record[key], seen);
  }

  return result;
}

function eventRecord<T>(value: T): JsonRecord {
  const record = asRecord(value) ?? {};
  const raw = asRecord(record.raw);

  return raw &&
    (record.isUnknown === true || record.is_unknown === true || record.type === "UNKNOWN")
    ? raw
    : record;
}

function eventNumber(event: JsonRecord, camel: string, snake: string): number | undefined {
  return readNumber(event[camel]) ?? readNumber(event[snake]);
}

function eventString(event: JsonRecord, camel: string, snake: string): string | undefined {
  return readString(event[camel]) ?? readString(event[snake]);
}

function compactUsage(usage: SpanFields["usage"]): SpanFields["usage"] {
  if (!usage) return undefined;

  return Object.values(usage).some((value) => value !== undefined) ? usage : undefined;
}

function stopSequences<T>(value: T): string[] | undefined {
  const single = readString(value);

  if (single !== undefined) return [single];
  const values = asArray(value);

  if (!values) return undefined;

  const strings = values.flatMap((item) => {
    const string = readString(item);

    return string === undefined ? [] : [string];
  });

  return strings.length > 0 ? strings : undefined;
}

function usageCost<T>(usage: T): number | undefined {
  const u = asRecord(usage);
  const details = asRecord(u?.costDetails) ?? asRecord(u?.cost_details);

  return (
    readNumber(u?.cost) ??
    readNumber(details?.upstreamInferenceCost) ??
    readNumber(details?.upstream_inference_cost)
  );
}

function chatRequest(request: JsonRecord): RequestMapping {
  const body = asRecord(request.chatRequest) ?? {};
  const model = readString(body.model);

  return {
    name: `chat ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      provider: "openrouter",
      model,
      input: body.messages,
      temperature: readNumber(body.temperature),
      topP: readNumber(body.topP),
      topK: readNumber(body.topK),
      maxTokens: readNumber(body.maxCompletionTokens) ?? readNumber(body.maxTokens),
      stopSequences: stopSequences(body.stop),
      seed: readNumber(body.seed),
      frequencyPenalty: readNumber(body.frequencyPenalty),
      presencePenalty: readNumber(body.presencePenalty),
    },
  };
}

function chatUsage<T>(usage: T): SpanFields["usage"] {
  const u = asRecord(usage);
  const promptDetails = asRecord(u?.promptTokensDetails);
  const completionDetails = asRecord(u?.completionTokensDetails);

  return compactUsage({
    inputTokens: readNumber(u?.promptTokens),
    outputTokens: readNumber(u?.completionTokens),
    totalTokens: readNumber(u?.totalTokens),
    cacheReadInputTokens: readNumber(promptDetails?.cachedTokens),
    cacheCreationInputTokens: readNumber(promptDetails?.cacheWriteTokens),
    reasoningOutputTokens: readNumber(completionDetails?.reasoningTokens),
  });
}

function finishReasonFields(finishReasons: string[]): SpanFields {
  const fields: SpanFields = { finishReason: finishReasons[0] };

  if (finishReasons.length > 1) {
    fields.attributes = { "gen_ai.response.finish_reasons": finishReasons };
  }

  return fields;
}

function chatResponse<T>(response: T): SpanFields {
  const r = asRecord(response) ?? {};
  const choices = asArray(r.choices) ?? [];

  const finishReasons = choices
    .map((choice) => readString(asRecord(choice)?.finishReason))
    .filter((reason): reason is string => reason !== undefined);

  return {
    responseModel: readString(r.model),
    responseId: readString(r.id),
    output: choices
      .map((choice) => asRecord(choice)?.message)
      .filter((message) => message !== undefined),
    usage: chatUsage(r.usage),
    costUsd: usageCost(r.usage),
    ...finishReasonFields(finishReasons),
  };
}

function responsesRequest(request: JsonRecord): RequestMapping {
  const body = asRecord(request.responsesRequest) ?? {};
  const model = readString(body.model);

  return {
    name: `chat ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      provider: "openrouter",
      model,
      input: body.input,
      systemInstructions: body.instructions,
      temperature: readNumber(body.temperature),
      topP: readNumber(body.topP),
      topK: readNumber(body.topK),
      maxTokens: readNumber(body.maxOutputTokens),
      frequencyPenalty: readNumber(body.frequencyPenalty),
      presencePenalty: readNumber(body.presencePenalty),
    },
  };
}

function responsesUsage<T>(usage: T): SpanFields["usage"] {
  const u = asRecord(usage);
  const inputDetails = asRecord(u?.inputTokensDetails) ?? asRecord(u?.input_tokens_details);
  const outputDetails = asRecord(u?.outputTokensDetails) ?? asRecord(u?.output_tokens_details);

  return compactUsage({
    inputTokens: readNumber(u?.inputTokens) ?? readNumber(u?.input_tokens),
    outputTokens: readNumber(u?.outputTokens) ?? readNumber(u?.output_tokens),
    totalTokens: readNumber(u?.totalTokens) ?? readNumber(u?.total_tokens),
    cacheReadInputTokens:
      readNumber(inputDetails?.cachedTokens) ?? readNumber(inputDetails?.cached_tokens),
    cacheCreationInputTokens:
      readNumber(inputDetails?.cacheWriteTokens) ?? readNumber(inputDetails?.cache_write_tokens),
    reasoningOutputTokens:
      readNumber(outputDetails?.reasoningTokens) ?? readNumber(outputDetails?.reasoning_tokens),
  });
}

function responsesResponse<T>(response: T, includeOutput = true): SpanFields {
  const r = asRecord(response) ?? {};
  const status = readString(r.status);
  const incompleteDetails = asRecord(r.incompleteDetails) ?? asRecord(r.incomplete_details);

  const fields: SpanFields = {
    responseModel: readString(r.model),
    responseId: readString(r.id),
    usage: responsesUsage(r.usage),
    costUsd: usageCost(r.usage),
    finishReason:
      status === "completed" ? "stop" : (readString(incompleteDetails?.reason) ?? status),
  };

  if (includeOutput) fields.output = responseNative(r.output);

  if (status === "failed") fields.error = responseFailedError(response);

  return fields;
}

function responseFailedError<T>(response: T): Error {
  const error = asRecord(asRecord(response)?.error);
  const code = readNumber(error?.code) ?? readString(error?.code);
  const message = readString(error?.message);

  return new Error(["response.failed", code, message].filter(Boolean).join(": "));
}

function responseStreamError<T>(event: T): Error {
  const e = asRecord(event);
  const code = readNumber(e?.code) ?? readString(e?.code);
  const message = readString(e?.message);

  return new Error(["response.error", code, message].filter(Boolean).join(": "));
}

function embeddingsRequest(request: JsonRecord): RequestMapping {
  const body = asRecord(request.requestBody) ?? {};
  const model = readString(body.model);

  return {
    name: `embeddings ${model ?? "unknown"}`,
    fields: {
      type: "embedding",
      provider: "openrouter",
      model,
      input: body.input,
    },
  };
}

function embeddingsResponse<T>(response: T): SpanFields {
  const r = asRecord(response) ?? {};
  const usage = asRecord(r.usage);

  return {
    responseModel: readString(r.model),
    responseId: readString(r.id),
    usage: compactUsage({
      inputTokens: readNumber(usage?.promptTokens),
      totalTokens: readNumber(usage?.totalTokens),
    }),
    costUsd: usageCost(usage),
  };
}

function isWrapped<T>(fn: T): fn is T & WrappedFunction {
  return fn instanceof Function && (fn as T & WrappedFunction)[WRAPPED] === true;
}

function markWrapped<T extends WrappedFunction>(fn: T, original: Method): T {
  Object.defineProperty(fn, WRAPPED, { value: true });
  Object.defineProperty(fn, ORIGINAL, { value: original });

  return fn;
}

function endOnce(span: SpanHandle): (fields?: SpanFields) => void {
  let ended = false;

  return (fields?: SpanFields) => {
    if (ended) return;
    ended = true;
    span.end(fields);
  };
}

function getChoiceState(
  states: Map<number, ChatChoiceState>,
  index: number,
  budget: StreamCaptureBudget,
): ChatChoiceState | undefined {
  let state = states.get(index);

  if (!state) {
    if (states.size >= MAX_STREAM_CAPTURE_ITEMS) {
      budget.truncated = true;

      return undefined;
    }

    state = {
      content: "",
      reasoning: "",
      reasoningDetails: [],
      refusal: "",
      toolCalls: new Map(),
    };
    states.set(index, state);
  }

  return state;
}

function mergeToolCall(state: ChatChoiceState, delta: JsonRecord): void {
  const index = readNumber(delta.index) ?? state.toolCalls.size;
  const current = { ...state.toolCalls.get(index) };

  if (delta.id !== undefined) current.id = delta.id;

  if (delta.type !== undefined) current.type = delta.type;
  const incomingFunction = asRecord(delta.function);

  if (incomingFunction) {
    const currentFunction = { ...asRecord(current.function) };

    if (incomingFunction.name !== undefined) currentFunction.name = incomingFunction.name;

    if (incomingFunction.arguments !== undefined) {
      currentFunction.arguments = `${readString(currentFunction.arguments) ?? ""}${readString(incomingFunction.arguments) ?? ""}`;
    }

    current.function = currentFunction;
  }

  state.toolCalls.set(index, current);
}

function chatOutput(states: Map<number, ChatChoiceState>): JsonRecord[] {
  return [...states.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, state]) => {
      const message = { role: state.role ?? "assistant" };

      if (state.content.length > 0) Object.assign(message, { content: state.content });
      else if (state.toolCalls.size > 0) Object.assign(message, { content: null });

      if (state.reasoning.length > 0) Object.assign(message, { reasoning: state.reasoning });

      if (state.reasoningDetails.length > 0) {
        Object.assign(message, { reasoningDetails: state.reasoningDetails });
      }

      if (state.refusal.length > 0) Object.assign(message, { refusal: state.refusal });

      if (state.toolCalls.size > 0) {
        Object.assign(message, {
          tool_calls: [...state.toolCalls.entries()]
            .sort(([left], [right]) => left - right)
            .map(([, toolCall]) => toolCall),
        });
      }

      return message;
    });
}

function chatPartialFields(
  states: Map<number, ChatChoiceState>,
  usage: SpanFields["usage"],
  costUsd: number | undefined,
  budget: StreamCaptureBudget,
): SpanFields {
  const finishReasons = [...states.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, state]) => state.finishReason)
    .filter((reason): reason is string => reason !== undefined);

  const fields: SpanFields = {
    output: states.size > 0 ? chatOutput(states) : undefined,
    usage,
    costUsd,
    ...finishReasonFields(finishReasons),
  };

  if (budget.truncated) {
    fields.attributes = {
      ...fields.attributes,
      [CAPTURE_TRUNCATED_ATTRIBUTE]: true,
    };
  }

  return fields;
}

function recordChatChunk<T>(
  chunk: T,
  states: Map<number, ChatChoiceState>,
  budget: StreamCaptureBudget,
) {
  const c = asRecord(chunk) ?? {};

  for (const choice of asArray(c.choices) ?? []) {
    const choiceRecord = asRecord(choice) ?? {};
    const state = getChoiceState(states, readNumber(choiceRecord.index) ?? 0, budget);

    if (!state) continue;
    const delta = asRecord(choiceRecord.delta) ?? {};
    const role = readString(delta.role);

    if (role !== undefined && budget.accept(role)) state.role = role;
    const content = readString(delta.content);

    if (content !== undefined && budget.accept(content)) {
      state.content += content;
    }

    const reasoning = readString(delta.reasoning);

    if (reasoning !== undefined && budget.accept(reasoning)) {
      state.reasoning += reasoning;
    }

    for (const detail of asArray(delta.reasoningDetails) ?? []) {
      if (budget.accept(detail)) state.reasoningDetails.push(detail);
    }

    const refusal = readString(delta.refusal);

    if (refusal !== undefined && budget.accept(refusal)) {
      state.refusal += refusal;
    }

    for (const toolCall of asArray(delta.toolCalls) ?? []) {
      const toolCallRecord = asRecord(toolCall);

      if (toolCallRecord && budget.accept(toolCallRecord)) mergeToolCall(state, toolCallRecord);
    }

    const finishReason = readString(choiceRecord.finishReason);

    if (finishReason) state.finishReason = finishReason;
  }

  const error = asRecord(c.error);
  const code = readNumber(error?.code) ?? readString(error?.code);
  const message = readString(error?.message);

  const update = {
    responseId: readString(c.id),
    responseModel: readString(c.model),
    usagePresent: c.usage !== undefined,
    usage: chatUsage(c.usage),
    costUsd: usageCost(c.usage),
  };

  if (!error) return update;

  return {
    ...update,
    error: new Error(["chat stream error", code, message].filter(Boolean).join(": ")),
  };
}

interface StreamObserver {
  record(value: JsonValue, receivedAt: number): SpanFields | undefined;
  finish(cause?: unknown): SpanFields;
}

function observeStream(
  source: ReadableStream<JsonValue>,
  observer: StreamObserver,
  end: (fields?: SpanFields) => void,
): ReadableStream<JsonValue> {
  let reader: ReadableStreamDefaultReader<JsonValue> | undefined;
  let released = false;

  const getReader = () => {
    reader ??= source.getReader();

    return reader;
  };

  const release = () => {
    if (released || !reader) return;
    released = true;
    reader.releaseLock();
  };

  const observed = new ReadableStream<JsonValue>(
    {
      async pull(controller) {
        try {
          const result = await getReader().read();
          const receivedAt = performance.now();

          if (result.done) {
            release();
            end(observer.finish());
            controller.close();

            return;
          }

          const terminal = observer.record(result.value, receivedAt);

          if (terminal) end(terminal);
          controller.enqueue(result.value);
        } catch (error) {
          release();
          end(observer.finish(error));
          controller.error(error);
        }
      },
      async cancel(reason) {
        try {
          await getReader().cancel(reason);
        } catch (error) {
          end(observer.finish(error));
          throw error;
        } finally {
          release();
          end(observer.finish());
        }
      },
    },
    { highWaterMark: 0 },
  );

  Object.setPrototypeOf(observed, Object.getPrototypeOf(source));

  return observed;
}

function createObservedChatStream(
  source: ReadableStream<JsonValue>,
  span: () => SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
): ReadableStream<JsonValue> {
  const states = new Map<number, ChatChoiceState>();
  const budget = new StreamCaptureBudget();
  let usage: SpanFields["usage"];
  let costUsd: number | undefined;
  let responseId: string | undefined;
  let responseModel: string | undefined;
  let streamError: Error | undefined;
  let sawFirst = false;

  return observeStream(
    source,
    {
      record(chunk, receivedAt) {
        const update = recordChatChunk(chunk, states, budget);

        if (
          (asArray(asRecord(chunk)?.choices) ?? []).some((choice) => {
            const delta = asRecord(asRecord(choice)?.delta);
            const audio = asRecord(delta?.audio);

            const reasoningDetails = [
              ...(asArray(delta?.reasoningDetails) ?? []),
              ...(asArray(delta?.reasoning_details) ?? []),
            ];

            return (
              delta &&
              ((typeof delta.content === "string" && delta.content.length > 0) ||
                (typeof delta.reasoning === "string" && delta.reasoning.length > 0) ||
                (typeof delta.refusal === "string" && delta.refusal.length > 0) ||
                (typeof audio?.data === "string" && audio.data.length > 0) ||
                reasoningDetails.some((detail) => {
                  const record = asRecord(detail);

                  return (
                    (typeof record?.text === "string" && record.text.length > 0) ||
                    (typeof record?.summary === "string" && record.summary.length > 0)
                  );
                }) ||
                (asArray(delta.toolCalls) ?? []).some((toolCall) => {
                  const fn = asRecord(asRecord(toolCall)?.function);

                  return typeof fn?.arguments === "string" && fn.arguments.length > 0;
                }))
            );
          })
        )
          span().recordOutputChunk?.(receivedAt);

        if (!sawFirst) {
          sawFirst = true;
          span().update({
            timeToFirstChunkMs: Date.now() - startedAt,
            responseId: update.responseId,
            responseModel: update.responseModel,
          });
        }

        if (update.responseId !== undefined) responseId = update.responseId;

        if (update.responseModel !== undefined) responseModel = update.responseModel;

        if (update.usagePresent) {
          usage = update.usage;
          costUsd = update.costUsd;
        }

        if ("error" in update && update.error) {
          streamError = update.error;

          return {
            ...chatPartialFields(states, usage, costUsd, budget),
            responseId,
            responseModel,
            error: update.error,
          };
        }

        return undefined;
      },
      finish(cause) {
        const fields: SpanFields = {
          ...chatPartialFields(states, usage, costUsd, budget),
          responseId,
          responseModel,
        };

        if (cause !== undefined || streamError !== undefined) {
          fields.error = (cause ?? streamError) as Error;
        }

        return fields;
      },
    },
    end,
  );
}

function responsesStreamOutput(
  states: Map<number, ResponsesOutputItemState>,
  syntheticEvents: JsonRecord[],
): JsonRecord[] {
  return [
    ...[...states.entries()].sort(([left], [right]) => left - right).map(([, state]) => state.item),
    ...syntheticEvents,
  ];
}

function responsesStreamOutputWithItem(
  states: Map<number, ResponsesOutputItemState>,
  syntheticEvents: JsonRecord[],
  outputIndex: number,
  item: JsonRecord,
): JsonRecord[] {
  const items = [...states.entries()].map(([index, state]) => [index, state.item] as const);
  const position = items.findIndex(([index]) => index === outputIndex);

  if (position === -1) items.push([outputIndex, item]);
  else items[position] = [outputIndex, item];

  return [
    ...items.sort(([left], [right]) => left - right).map(([, value]) => value),
    ...syntheticEvents,
  ];
}

function hydrateResponsesItem(
  states: Map<number, ResponsesOutputItemState>,
  outputIndex: number,
  item: JsonRecord,
): ResponsesOutputItemState {
  const content = new Map<number, JsonRecord>();

  for (const [index, value] of (asArray(item.content) ?? []).entries()) {
    const part = asRecord(value);

    if (part) content.set(index, part);
  }

  const summary = new Map<number, JsonRecord>();

  for (const [index, value] of (asArray(item.summary) ?? []).entries()) {
    const part = asRecord(value);

    if (part) summary.set(index, part);
  }

  const state = { content, item, summary };
  states.set(outputIndex, state);

  return state;
}

function getResponsesItem(
  states: Map<number, ResponsesOutputItemState>,
  outputIndex: number,
  event: JsonRecord,
  type: string,
): ResponsesOutputItemState {
  const existing = states.get(outputIndex);

  if (existing) return existing;
  const reasoning = type.startsWith("response.reasoning_");

  const item = {
    id: eventString(event, "itemId", "item_id") ?? `output_${outputIndex}`,
    type: reasoning ? "reasoning" : "message",
    status: "in_progress",
    ...(reasoning ? { summary: [] } : { role: "assistant" }),
    content: [],
  };

  return hydrateResponsesItem(states, outputIndex, item);
}

function responsesItemCandidate(
  state: ResponsesOutputItemState,
  contentIndex?: number,
  content?: JsonRecord,
  summaryIndex?: number,
  summary?: JsonRecord,
) {
  const item = { ...state.item };

  if (contentIndex !== undefined && content !== undefined) {
    const values = new Map(state.content);
    values.set(contentIndex, content);
    item.content = [...values.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, value]) => value);
  }

  if (summaryIndex !== undefined && summary !== undefined) {
    const values = new Map(state.summary);
    values.set(summaryIndex, summary);
    item.summary = [...values.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, value]) => value);
  }

  return item;
}

function syncResponsesItem(state: ResponsesOutputItemState): void {
  if (state.content.size > 0) {
    state.item.content = [...state.content.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, content]) => content);
  }

  if (state.summary.size > 0) {
    state.item.summary = [...state.summary.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, summary]) => summary);
  }
}

function replaceResponsesOutput<T>(
  budget: StreamCaptureBudget,
  output: T,
  providerCapture: { truncated: boolean },
): boolean {
  if (!budget.replace(output)) return false;
  budget.truncated = providerCapture.truncated;

  return true;
}

function responsesProviderEventPayload(event: JsonRecord, type: string) {
  if (type !== "response.debug") return responseNative(event);
  const debug = asRecord(event.debug);
  const timings = debug?.timings;
  const sequenceNumber = eventNumber(event, "sequenceNumber", "sequence_number");
  const value = timings === undefined ? {} : { timings: responseNative(timings) };

  return sequenceNumber === undefined
    ? { type, debug: value }
    : { type, sequence_number: sequenceNumber, debug: value };
}

function recordResponsesEvent(
  event: JsonRecord,
  states: Map<number, ResponsesOutputItemState>,
  syntheticEvents: JsonRecord[],
  budget: StreamCaptureBudget,
  providerCapture: { truncated: boolean },
): boolean {
  const type = readString(event.type);

  if (!type) return false;
  const outputIndex = eventNumber(event, "outputIndex", "output_index") ?? 0;

  if (type === "response.output_item.added" || type === "response.output_item.done") {
    const replacement = type.endsWith(".done");
    const preflight = replacement ? new StreamCaptureBudget() : budget;

    if (!preflight.canAccept(event.item)) {
      budget.truncated = true;

      return false;
    }

    const item = asRecord(responseNative(event.item));

    if (!item) return false;

    if (replacement) {
      if (
        !replaceResponsesOutput(
          budget,
          responsesStreamOutputWithItem(states, syntheticEvents, outputIndex, item),
          providerCapture,
        )
      ) {
        return false;
      }
    } else if (!budget.accept(item)) return false;
    hydrateResponsesItem(states, outputIndex, item);

    return true;
  }

  if (type === "response.content_part.added" || type === "response.content_part.done") {
    const replacement = type.endsWith(".done");
    const preflight = replacement ? new StreamCaptureBudget() : budget;

    if (!preflight.canAccept(event.part)) {
      budget.truncated = true;

      return false;
    }

    const part = asRecord(responseNative(event.part));

    if (!part) return false;
    const existing = states.get(outputIndex);
    const state = getResponsesItem(states, outputIndex, event, type);
    const contentIndex = eventNumber(event, "contentIndex", "content_index") ?? 0;

    if (replacement) {
      const item = responsesItemCandidate(state, contentIndex, part);

      if (
        !replaceResponsesOutput(
          budget,
          responsesStreamOutputWithItem(states, syntheticEvents, outputIndex, item),
          providerCapture,
        )
      ) {
        if (!existing) states.delete(outputIndex);

        return false;
      }
    } else if (!budget.accept(part)) return false;
    state.content.set(contentIndex, part);
    syncResponsesItem(state);

    return true;
  }

  if (type === "response.output_text.annotation.added") {
    const contentIndex = eventNumber(event, "contentIndex", "content_index") ?? 0;
    const existingPart = states.get(outputIndex)?.content.get(contentIndex);
    const annotations = asArray(existingPart?.annotations) ?? [];
    const annotationIndex = eventNumber(event, "annotationIndex", "annotation_index");

    if (
      annotationIndex !== undefined &&
      (!Number.isSafeInteger(annotationIndex) ||
        annotationIndex < 0 ||
        annotationIndex > annotations.length)
    ) {
      budget.truncated = true;

      return false;
    }

    if (!budget.canAccept(event.annotation)) {
      budget.truncated = true;

      return false;
    }

    const annotation = responseNative(event.annotation);

    if (annotation === undefined || !budget.accept(annotation)) return false;
    const state = getResponsesItem(states, outputIndex, event, type);
    let part = state.content.get(contentIndex);

    if (!part) {
      part = { type: "output_text", text: "", annotations: [] };
      state.content.set(contentIndex, part);
    }

    if (annotationIndex === undefined) annotations.push(annotation);
    else annotations[annotationIndex] = annotation;
    part.annotations = annotations;
    syncResponsesItem(state);

    return true;
  }

  if (
    type === "response.function_call_arguments.delta" ||
    type === "response.function_call_arguments.done" ||
    type === "response.custom_tool_call_input.delta" ||
    type === "response.custom_tool_call_input.done"
  ) {
    const functionCall = type.startsWith("response.function_call_arguments");
    const existing = states.get(outputIndex);

    const state =
      existing ??
      hydrateResponsesItem(states, outputIndex, {
        id: eventString(event, "itemId", "item_id") ?? `output_${outputIndex}`,
        type: functionCall ? "function_call" : "custom_tool_call",
        status: "in_progress",
        ...(functionCall ? { arguments: "" } : { input: "" }),
      });

    const field = functionCall ? "arguments" : "input";
    const done = type.endsWith(".done");
    const value = done ? readString(event[field]) : readString(event.delta);

    if (value === undefined) return false;

    if (done) {
      const item = {
        ...state.item,
        [field]: value,
      };

      if (event.name !== undefined) item.name = event.name;
      item.status = "completed";

      if (
        !replaceResponsesOutput(
          budget,
          responsesStreamOutputWithItem(states, syntheticEvents, outputIndex, item),
          providerCapture,
        )
      ) {
        if (!existing) states.delete(outputIndex);

        return false;
      }

      state.item = item;
    } else {
      if (!budget.accept(value)) return false;
      state.item[field] = `${readString(state.item[field]) ?? ""}${value}`;

      if (event.name !== undefined) state.item.name = event.name;
    }

    return true;
  }

  if (
    type === "response.reasoning_summary_part.added" ||
    type === "response.reasoning_summary_part.done"
  ) {
    const replacement = type.endsWith(".done");
    const preflight = replacement ? new StreamCaptureBudget() : budget;

    if (!preflight.canAccept(event.part)) {
      budget.truncated = true;

      return false;
    }

    const part = asRecord(responseNative(event.part));

    if (!part) return false;
    const existing = states.get(outputIndex);
    const state = getResponsesItem(states, outputIndex, event, type);
    const summaryIndex = eventNumber(event, "summaryIndex", "summary_index") ?? 0;

    if (replacement) {
      const item = responsesItemCandidate(state, undefined, undefined, summaryIndex, part);

      if (
        !replaceResponsesOutput(
          budget,
          responsesStreamOutputWithItem(states, syntheticEvents, outputIndex, item),
          providerCapture,
        )
      ) {
        if (!existing) states.delete(outputIndex);

        return false;
      }
    } else if (!budget.accept(part)) return false;
    state.summary.set(summaryIndex, part);
    syncResponsesItem(state);

    return true;
  }

  if (
    type === "response.image_generation_call.partial_image" ||
    type === "response.image_generation_call.in_progress" ||
    type === "response.image_generation_call.generating" ||
    type === "response.image_generation_call.completed" ||
    type === "response.apply_patch_call_operation_diff.delta" ||
    type === "response.apply_patch_call_operation_diff.done" ||
    type === "response.fusion_call.in_progress" ||
    type === "response.fusion_call.completed" ||
    type === "response.fusion_call.analysis.in_progress" ||
    type === "response.fusion_call.analysis.completed" ||
    type === "response.fusion_call.panel.added" ||
    type === "response.fusion_call.panel.delta" ||
    type === "response.fusion_call.panel.reasoning.delta" ||
    type === "response.fusion_call.panel.completed" ||
    type === "response.fusion_call.panel.failed" ||
    type === "response.web_search_call.in_progress" ||
    type === "response.web_search_call.searching" ||
    type === "response.web_search_call.completed" ||
    type === "response.debug"
  ) {
    const synthetic = {
      type: "telemetry.dev.response_stream_event",
      event_type: type,
      payload: responsesProviderEventPayload(event, type),
    };

    if (!budget.accept(synthetic)) {
      budget.truncated = true;
      providerCapture.truncated = true;

      return false;
    }

    syntheticEvents.push(synthetic);

    return true;
  }

  if (
    ![
      "response.output_text.delta",
      "response.output_text.done",
      "response.reasoning_text.delta",
      "response.reasoning_text.done",
      "response.reasoning_summary_text.delta",
      "response.reasoning_summary_text.done",
      "response.refusal.delta",
      "response.refusal.done",
    ].includes(type)
  ) {
    return false;
  }

  const done = type.endsWith(".done");
  const refusal = type.startsWith("response.refusal.");
  const value = done ? readString(event[refusal ? "refusal" : "text"]) : readString(event.delta);

  if (value === undefined || (!done && !budget.accept(value))) return false;
  const summaryText = type.startsWith("response.reasoning_summary_text.");
  const existing = states.get(outputIndex);
  const state = getResponsesItem(states, outputIndex, event, type);

  if (summaryText) {
    const summaryIndex = eventNumber(event, "summaryIndex", "summary_index") ?? 0;
    let summary = state.summary.get(summaryIndex);

    if (!summary) {
      summary = { type: "summary_text", text: "" };

      if (!done) {
        state.summary.set(summaryIndex, summary);
        syncResponsesItem(state);
      }
    }

    if (done) {
      const part = { ...summary, text: value };
      const item = responsesItemCandidate(state, undefined, undefined, summaryIndex, part);

      if (
        !replaceResponsesOutput(
          budget,
          responsesStreamOutputWithItem(states, syntheticEvents, outputIndex, item),
          providerCapture,
        )
      ) {
        if (!existing) states.delete(outputIndex);

        return false;
      }

      state.summary.set(summaryIndex, part);
      syncResponsesItem(state);
    } else {
      summary.text = `${readString(summary.text) ?? ""}${value}`;
    }

    return true;
  }

  const contentIndex = eventNumber(event, "contentIndex", "content_index") ?? 0;
  let part = state.content.get(contentIndex);

  if (!part) {
    if (refusal) {
      part = { type: "refusal", refusal: "" };
    } else {
      part = {
        type: type.startsWith("response.reasoning_text.") ? "reasoning_text" : "output_text",
        text: "",
      };

      if (type.startsWith("response.output_text.")) part.annotations = [];
    }

    if (!done) {
      state.content.set(contentIndex, part);
      syncResponsesItem(state);
    }
  }

  if (done) {
    const replacement = {
      ...part,
      [refusal ? "refusal" : "text"]: value,
    };

    const item = responsesItemCandidate(state, contentIndex, replacement);

    if (
      !replaceResponsesOutput(
        budget,
        responsesStreamOutputWithItem(states, syntheticEvents, outputIndex, item),
        providerCapture,
      )
    ) {
      if (!existing) states.delete(outputIndex);

      return false;
    }

    state.content.set(contentIndex, replacement);
    syncResponsesItem(state);
  } else if (refusal) {
    part.refusal = `${readString(part.refusal) ?? ""}${value}`;
  } else {
    part.text = `${readString(part.text) ?? ""}${value}`;
  }

  return true;
}

function createObservedResponsesStream(
  source: ReadableStream<JsonValue>,
  span: () => SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
): ReadableStream<JsonValue> {
  let partial: SpanFields = {};
  const outputStates = new Map<number, ResponsesOutputItemState>();
  const syntheticEvents: JsonRecord[] = [];
  const providerCapture = { truncated: false };
  let budget = new StreamCaptureBudget();
  let sawFirst = false;

  return observeStream(
    source,
    {
      record(event, receivedAt) {
        const e = eventRecord(event);

        if (responseEventHasOutput(e)) span().recordOutputChunk?.(receivedAt);
        const response = asRecord(e.response);

        if (!sawFirst) {
          sawFirst = true;
          span().update({
            timeToFirstChunkMs: Date.now() - startedAt,
            responseId: readString(response?.id),
            responseModel: readString(response?.model),
          });
        }

        if (response) {
          const retainedOutput = partial.output;
          let output = retainedOutput;

          if (response.output !== undefined) {
            const responseBudget = new StreamCaptureBudget();

            if (!responseBudget.canAccept(response.output)) {
              budget.truncated = true;
            } else {
              const responseOutput = responseNative(response.output);
              const responseItems = asArray(responseOutput);

              if (responseItems === undefined) {
                budget.truncated = true;
              } else {
                const responseArtifact = [...responseItems, ...syntheticEvents];

                if (!responseBudget.accept(responseArtifact)) {
                  budget.truncated = true;
                } else {
                  outputStates.clear();

                  for (const [index, value] of responseItems.entries()) {
                    const item = asRecord(value);

                    if (item) hydrateResponsesItem(outputStates, index, item);
                  }

                  output = responseArtifact;
                  responseBudget.truncated = providerCapture.truncated;
                  budget = responseBudget;
                }
              }
            }
          }

          partial = responsesResponse(response, false);

          if (output !== undefined) partial.output = output;
        } else if (
          recordResponsesEvent(e, outputStates, syntheticEvents, budget, providerCapture)
        ) {
          partial = {
            ...partial,
            output: responsesStreamOutput(outputStates, syntheticEvents),
          };
        }

        if (budget.truncated) {
          partial = {
            ...partial,
            attributes: {
              ...partial.attributes,
              [CAPTURE_TRUNCATED_ATTRIBUTE]: true,
            },
          };
        } else if (partial.attributes?.[CAPTURE_TRUNCATED_ATTRIBUTE] !== undefined) {
          const attributes = { ...partial.attributes };
          delete attributes[CAPTURE_TRUNCATED_ATTRIBUTE];
          partial = {
            ...partial,
            ...(Object.keys(attributes).length > 0 ? { attributes } : { attributes: undefined }),
          };
        }

        if (e.type === "response.failed") {
          partial = { ...partial, error: responseFailedError(response) };

          return partial;
        }

        if (e.type === "error") {
          partial = { ...partial, error: responseStreamError(e) };

          return partial;
        }

        if (e.type === "response.completed" || e.type === "response.incomplete") return partial;

        return undefined;
      },
      finish(cause) {
        const fields: SpanFields = { ...partial };

        if (budget.truncated) {
          fields.attributes = {
            ...partial.attributes,
            [CAPTURE_TRUNCATED_ATTRIBUTE]: true,
          };
        }

        if (cause !== undefined) fields.error = cause as Error;

        return fields;
      },
    },
    end,
  );
}

function responseEventHasOutput(event: JsonRecord): boolean {
  return (
    typeof event.type === "string" &&
    [
      "response.output_text.delta",
      "response.refusal.delta",
      "response.reasoning_text.delta",
      "response.reasoning_summary_text.delta",
      "response.function_call_arguments.delta",
      "response.custom_tool_call_input.delta",
      "response.code_interpreter_call_code.delta",
      "response.mcp_call_arguments.delta",
      "response.apply_patch_call_operation_diff.delta",
      "response.fusion_call.panel.delta",
      "response.fusion_call.panel.reasoning.delta",
      "response.output_audio.delta",
      "response.audio.delta",
      "response.audio.transcript.delta",
    ].includes(event.type) &&
    typeof event.delta === "string" &&
    event.delta.length > 0
  );
}

function asReadableStream<T>(value: T): ReadableStream<JsonValue> | undefined {
  const stream = asRecord(value);

  return stream?.getReader instanceof Function ? (value as ReadableStream<JsonValue>) : undefined;
}

function finalizeResult<T>(value: T, context: TraceContext) {
  if (context.streaming) {
    const stream = asReadableStream(value);

    if (stream) {
      return context.operation === "chat"
        ? createObservedChatStream(stream, context.span, context.startedAt, context.end)
        : createObservedResponsesStream(stream, context.span, context.startedAt, context.end);
    }
  }

  context.end(context.mapResponse(value));

  return value;
}

function wrapMethod<Fn extends Method>(
  original: Fn,
  operation: Operation,
  mapRequest: (request: JsonRecord) => RequestMapping,
  mapResponse: <T>(response: T) => SpanFields,
): Fn {
  if (isWrapped(original)) return original;

  const wrapped = function <This>(this: This, ...args: JsonValue[]) {
    const request = asRecord(args[0]) ?? {};
    const bodyKey = operation === "chat" ? "chatRequest" : "responsesRequest";
    const streaming = operation !== "embeddings" && asRecord(request[bodyKey])?.stream === true;
    const mapped = mapRequest(request);
    const startedAt = Date.now();
    const parent = activeContext();
    let span: SpanHandle | undefined;
    let finish: ((fields?: SpanFields) => void) | undefined;

    const getSpan = () => {
      if (span) return span;
      span = startSpan(mapped.name, { ...mapped.fields, parent, startTime: startedAt });
      finish = endOnce(span);

      return span;
    };

    const end = (fields?: SpanFields) => {
      getSpan();
      finish?.(fields);
    };

    if (!streaming) getSpan();

    const context: TraceContext = {
      end,
      span: getSpan,
      startedAt,
      operation,
      streaming,
      mapResponse,
    };

    try {
      const result = original.apply(this, args as never[]);

      return Promise.resolve(result).then(
        (value) => finalizeResult(value, context),
        (cause) => {
          context.end({ error: cause as Error });
          throw cause;
        },
      );
    } catch (cause) {
      context.end({ error: cause as Error });
      throw cause;
    }
  };

  const marked = markWrapped(wrapped as WrappedFunction, original);

  return marked as Fn;
}

function patchInstanceMethod<T extends CallableRecord>(
  target: T,
  key: keyof T,
  operation: Operation,
  request: (value: JsonRecord) => RequestMapping,
  response: <R>(value: R) => SpanFields,
): void {
  const current = target[key];

  if (isWrapped(current) && Object.prototype.hasOwnProperty.call(target, key)) return;
  const original = isWrapped(current) ? current[ORIGINAL] : current;

  if (!(original instanceof Function)) return;
  const bound: Method = original.bind(target);
  const wrapped = wrapMethod(bound, operation, request, response);
  Object.assign(target, { [key]: wrapped });
}

function patchPrototype(
  klass: { prototype: object },
  key: string,
  operation: Operation,
  request: (value: JsonRecord) => RequestMapping,
  response: <R>(value: R) => SpanFields,
): () => void {
  const prototype = klass.prototype as CallableRecord;
  const original = prototype[key];

  if (!(original instanceof Function)) return () => {};

  const wrapped = wrapMethod(original as Method, operation, request, response);
  prototype[key] = wrapped;

  return () => {
    if (prototype[key] === wrapped) prototype[key] = original;
  };
}

let installed = false;
let restorePatches: Array<() => void> = [];

export function wrapOpenRouter<T extends OpenRouterClient>(client: T): T {
  if (wrappedClients.has(client)) return client;
  const rawChat: unknown = client.chat;
  const chat = rawChat as CallableRecord & { send?: WrappedFunction };
  const rawResponses: unknown = client.responses;
  const responses = rawResponses as CallableRecord & { send?: WrappedFunction };
  const rawEmbeddings: unknown = client.embeddings;
  const embeddings = rawEmbeddings as CallableRecord & { generate?: WrappedFunction };
  patchInstanceMethod(chat, "send", "chat", chatRequest, chatResponse);
  patchInstanceMethod(responses, "send", "responses", responsesRequest, responsesResponse);
  patchInstanceMethod(embeddings, "generate", "embeddings", embeddingsRequest, embeddingsResponse);
  wrappedClients.add(client);

  return client;
}

export function instrumentOpenRouter(): void {
  if (installed) return;
  restorePatches = [
    patchPrototype(Chat, "send", "chat", chatRequest, chatResponse),
    patchPrototype(Responses, "send", "responses", responsesRequest, responsesResponse),
    patchPrototype(Embeddings, "generate", "embeddings", embeddingsRequest, embeddingsResponse),
  ];
  installed = true;
}

export function uninstrumentOpenRouter(): void {
  if (!installed) return;

  for (const restore of restorePatches.reverse()) restore();
  restorePatches = [];
  installed = false;
}
