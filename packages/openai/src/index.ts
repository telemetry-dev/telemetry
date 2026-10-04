import {
  boundedCapture,
  boundedCaptureDetails,
  captureEnabled,
  startSpan,
  type SpanFields,
  type SpanHandle,
  type StartSpanOptions,
} from "@telemetry-dev/sdk";
import OpenAI from "openai";
import { Stream } from "openai/core/streaming";
import { Speech } from "openai/resources/audio/speech";
import { Transcriptions } from "openai/resources/audio/transcriptions";
import { Translations } from "openai/resources/audio/translations";
import { Batches } from "openai/resources/batches";
import { Completions } from "openai/resources/chat/completions/completions";
import { Embeddings } from "openai/resources/embeddings";
import { Images } from "openai/resources/images";
import { Responses } from "openai/resources/responses/responses";

const WRAPPED = Symbol("telemetry.dev.openai.wrapped");
const ORIGINAL = Symbol("telemetry.dev.openai.original");
const wrappedClients = new WeakSet<object>();
const wrappedRealtimeConnections = new WeakSet<object>();
let realtimeEventSequence = 0;
const MAX_CAPTURE_LENGTH = 65536;
const REALTIME_CORRELATION_KEY = "__telemetry_dev_response_id";
const REALTIME_TRACE_TIMEOUT_MS = 5 * 60_000;
const REALTIME_MAX_IN_FLIGHT = 100;
const REALTIME_CAPTURE_MAX_BYTES = 48 * 1024;
const REALTIME_CAPTURE_MAX_ITEMS = 1_000;
const CHAT_STREAM_CAPTURE_MAX_BYTES = 48 * 1024;
const CHAT_STREAM_CAPTURE_MAX_ITEMS = 1_000;

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue | undefined };

type JsonRecord = { [key: string]: JsonValue | undefined };

type WrappedFunction = ((...args: never[]) => JsonValue | Promise<JsonValue>) & {
  [WRAPPED]?: true;
  [ORIGINAL]?: (...args: never[]) => JsonValue | Promise<JsonValue>;
};

type ProviderResolver = <T>(resource: T) => string;

type Operation = "chat" | "responses" | "embeddings" | "images" | "transcriptions";

interface OpenAIClient {
  chat: { completions: object };
  responses: object;
  embeddings: object;
  images?: object;
  audio?: { speech?: object; transcriptions?: object; translations?: object };
  videos?: object;
  batches?: object;
}

interface RequestMapping {
  name: string;
  fields: StartSpanOptions;
}

interface WrappedResponse<T> {
  data: T;
  response: Response;
  request_id: string | null;
}

function asRecord<T>(value: T): JsonRecord | undefined {
  return value !== null && Object(value) === value ? (value as JsonRecord) : undefined;
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

function readOwn(
  record: JsonRecord,
  key: string,
  budget?: ChatCaptureBudget,
  failure?: { value: boolean },
  reportError?: (cause: unknown) => void,
): unknown {
  try {
    if (!Object.hasOwn(record, key)) return undefined;
    const descriptor = Object.getOwnPropertyDescriptor(record, key);

    return descriptor?.get ? descriptor.get.call(record) : descriptor?.value;
  } catch (error) {
    if (budget) budget.truncated = true;

    if (failure) failure.value = true;
    reportError?.(error);

    return undefined;
  }
}

function sanitizeResponsesCapture(value: unknown, maxBytes = CHAT_STREAM_CAPTURE_MAX_BYTES) {
  return boundedCapture(value, {
    maxBytes,
    skip(key, item, parent, path) {
      const type = readString(asRecord(parent)?.type) ?? path.at(-1);

      return (
        key === "b64_json" ||
        key === "file_data" ||
        (key === "image_url" && typeof item === "string" && /^data:/i.test(item)) ||
        (key === "url" &&
          type === "image_url" &&
          typeof item === "string" &&
          /^data:/i.test(item)) ||
        ((key === "data" || key === "audio") &&
          ["input_audio", "output_audio", "audio"].includes(type ?? "")) ||
        (key === "result" && type === "image_generation_call") ||
        (key === "partial_image_b64" && type === "image_generation_call")
      );
    },
  });
}

function asError<T>(cause: T): Error {
  return cause instanceof Error ? cause : new Error(String(cause));
}

function compactUsage(usage: SpanFields["usage"]): SpanFields["usage"] {
  if (!usage) return undefined;

  return Object.values(usage).some((value) => value !== undefined) ? usage : undefined;
}

function stopSequences<T>(value: T): string[] | undefined {
  if (String(value) === value) return [String(value)];

  if (!Array.isArray(value)) return undefined;
  const strings = value.filter((item): item is string => String(item) === item);

  return strings.length > 0 ? strings : undefined;
}

function chatRequest(body: JsonRecord): RequestMapping {
  const model = readString(body.model);
  const capture = captureEnabled("input") ? sanitizeResponsesCapture(body.messages) : undefined;

  return {
    name: `chat ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      model,
      input: capture?.value,
      temperature: readNumber(body.temperature),
      topP: readNumber(body.top_p),
      maxTokens: readNumber(body.max_completion_tokens) ?? readNumber(body.max_tokens),
      stopSequences: stopSequences(body.stop),
      seed: readNumber(body.seed),
      frequencyPenalty: readNumber(body.frequency_penalty),
      presencePenalty: readNumber(body.presence_penalty),
      attributes: capture?.truncated ? { "telemetry.dev.capture.truncated": true } : undefined,
    },
  };
}

function chatUsage<T>(usage: T): SpanFields["usage"] {
  const u = asRecord(usage);
  const promptDetails = asRecord(u?.prompt_tokens_details);
  const completionDetails = asRecord(u?.completion_tokens_details);

  return compactUsage({
    inputTokens: readNumber(u?.prompt_tokens),
    outputTokens: readNumber(u?.completion_tokens),
    totalTokens: readNumber(u?.total_tokens),
    cacheReadInputTokens: readNumber(promptDetails?.cached_tokens),
    textInputTokens: readNumber(promptDetails?.text_tokens),
    imageInputTokens: readNumber(promptDetails?.image_tokens),
    audioInputTokens: readNumber(promptDetails?.audio_tokens),
    textCacheReadInputTokens: readNumber(
      asRecord(promptDetails?.cached_tokens_details)?.text_tokens,
    ),
    imageCacheReadInputTokens: readNumber(
      asRecord(promptDetails?.cached_tokens_details)?.image_tokens,
    ),
    audioCacheReadInputTokens: readNumber(
      asRecord(promptDetails?.cached_tokens_details)?.audio_tokens,
    ),
    textOutputTokens: readNumber(completionDetails?.text_tokens),
    imageOutputTokens: readNumber(completionDetails?.image_tokens),
    audioOutputTokens: readNumber(completionDetails?.audio_tokens),
    reasoningOutputTokens: readNumber(completionDetails?.reasoning_tokens),
  });
}

function chatResponse<T>(response: T): SpanFields {
  const r = asRecord(response) ?? {};
  const choices = asArray(r.choices) ?? [];

  const capture = captureEnabled("output")
    ? sanitizeResponsesCapture(
        choices
          .map((choice) => asRecord(choice)?.message)
          .filter((message) => message !== undefined),
      )
    : undefined;

  const finishReasons = choices
    .map((choice) => readString(asRecord(choice)?.finish_reason))
    .filter((reason): reason is string => reason !== undefined);

  const fields: SpanFields = {
    responseModel: readString(r.model),
    responseId: readString(r.id),
    finishReason: finishReasons[0],
    output: capture?.value,
    usage: chatUsage(r.usage),
  };

  if (finishReasons.length > 1 || capture?.truncated) {
    fields.attributes = {
      ...(finishReasons.length > 1
        ? { "gen_ai.response.finish_reasons": finishReasons }
        : undefined),
      ...(capture?.truncated ? { "telemetry.dev.capture.truncated": true } : undefined),
    };
  }

  return fields;
}

function responsesRequest(body: JsonRecord): RequestMapping {
  const model = readString(body.model);
  const capture = captureEnabled("input") ? sanitizeResponsesCapture(body.input) : undefined;

  return {
    name: `chat ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      model,
      input: capture?.value,
      systemInstructions: body.instructions,
      temperature: readNumber(body.temperature),
      topP: readNumber(body.top_p),
      maxTokens: readNumber(body.max_output_tokens),
      attributes: capture?.truncated ? { "telemetry.dev.capture.truncated": true } : undefined,
    },
  };
}

function responsesUsage<T>(usage: T): SpanFields["usage"] {
  const u = asRecord(usage);
  const inputDetails = asRecord(u?.input_tokens_details);
  const outputDetails = asRecord(u?.output_tokens_details);

  return compactUsage({
    inputTokens: readNumber(u?.input_tokens),
    outputTokens: readNumber(u?.output_tokens),
    totalTokens: readNumber(u?.total_tokens),
    cacheReadInputTokens: readNumber(inputDetails?.cached_tokens),
    textInputTokens: readNumber(inputDetails?.text_tokens),
    imageInputTokens: readNumber(inputDetails?.image_tokens),
    audioInputTokens: readNumber(inputDetails?.audio_tokens),
    textCacheReadInputTokens: readNumber(
      asRecord(inputDetails?.cached_tokens_details)?.text_tokens,
    ),
    imageCacheReadInputTokens: readNumber(
      asRecord(inputDetails?.cached_tokens_details)?.image_tokens,
    ),
    audioCacheReadInputTokens: readNumber(
      asRecord(inputDetails?.cached_tokens_details)?.audio_tokens,
    ),
    textOutputTokens: readNumber(outputDetails?.text_tokens),
    imageOutputTokens: readNumber(outputDetails?.image_tokens),
    audioOutputTokens: readNumber(outputDetails?.audio_tokens),
    reasoningOutputTokens: readNumber(outputDetails?.reasoning_tokens),
  });
}

function responsesResponse<T>(
  response: T,
  captureOutput = captureEnabled("output"),
  maxBytes = CHAT_STREAM_CAPTURE_MAX_BYTES,
): SpanFields {
  const r = asRecord(response) ?? {};
  const status = readString(r.status);
  const incompleteDetails = asRecord(r.incomplete_details);
  const capture = captureOutput ? sanitizeResponsesCapture(r.output, maxBytes) : undefined;

  const fields: SpanFields = {
    responseModel: readString(r.model),
    responseId: readString(r.id),
    output: capture?.value,
    usage: responsesUsage(r.usage),
    finishReason:
      status === "completed" ? "stop" : (readString(incompleteDetails?.reason) ?? status),
  };

  if (capture?.truncated) fields.attributes = { "telemetry.dev.capture.truncated": true };

  if (status === "failed") fields.error = responseFailedError(response);

  return fields;
}

function responseFailedError<T>(response: T): Error {
  const error = asRecord(asRecord(response)?.error);
  const code = readString(error?.code);
  const message = readString(error?.message);

  return new Error(["response.failed", code, message].filter(Boolean).join(": "));
}

function responseStreamError<T>(event: T): Error {
  const e = asRecord(event);
  const code = readString(e?.code);
  const message = readString(e?.message);

  return new Error(["response.error", code, message].filter(Boolean).join(": "));
}

function responsesEventFailure<T>(event: T): Error | undefined {
  let type: unknown;

  try {
    type = asRecord(event)?.type;
  } catch {
    return undefined;
  }

  if (type !== "response.failed" && type !== "error") return undefined;

  try {
    return type === "response.failed"
      ? responseFailedError(asRecord(event)?.response)
      : responseStreamError(event);
  } catch {
    return new Error(type === "response.failed" ? "response.failed" : "response.error");
  }
}

function embeddingsRequest(body: JsonRecord): RequestMapping {
  const model = readString(body.model);

  return {
    name: `embeddings ${model ?? "unknown"}`,
    fields: {
      type: "embedding",
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
    usage: compactUsage({
      inputTokens: readNumber(usage?.prompt_tokens),
      totalTokens: readNumber(usage?.total_tokens),
    }),
  };
}

function modalityUsage<T>(usage: T, outputModality?: "image" | "text"): SpanFields["usage"] {
  const u = asRecord(usage);
  const input = asRecord(u?.input_tokens_details) ?? asRecord(u?.input_token_details);
  const output = asRecord(u?.output_tokens_details) ?? asRecord(u?.output_token_details);
  const cached = asRecord(input?.cached_tokens_details);

  return compactUsage({
    inputTokens: readNumber(u?.input_tokens),
    outputTokens: readNumber(u?.output_tokens),
    totalTokens: readNumber(u?.total_tokens),
    cacheReadInputTokens: readNumber(input?.cached_tokens),
    textInputTokens: readNumber(input?.text_tokens),
    imageInputTokens: readNumber(input?.image_tokens),
    audioInputTokens: readNumber(input?.audio_tokens),
    textCacheReadInputTokens: readNumber(cached?.text_tokens),
    imageCacheReadInputTokens: readNumber(cached?.image_tokens),
    audioCacheReadInputTokens: readNumber(cached?.audio_tokens),
    textOutputTokens:
      readNumber(output?.text_tokens) ??
      (outputModality === "text" ? readNumber(u?.output_tokens) : undefined),
    imageOutputTokens:
      readNumber(output?.image_tokens) ??
      (outputModality === "image" ? readNumber(u?.output_tokens) : undefined),
    audioOutputTokens: readNumber(output?.audio_tokens),
    reasoningOutputTokens: readNumber(output?.reasoning_tokens),
  });
}

function generationRequest(
  body: JsonRecord,
  label: string,
  outputType: string,
  input: JsonValue | undefined,
): RequestMapping {
  const model = readString(body.model);

  return {
    name: `${label} ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      model,
      input,
      outputType,
      attributes: { "gen_ai.operation.name": "generate_content" },
    },
  };
}

function imageRequest(body: JsonRecord): RequestMapping {
  return generationRequest(body, "image", "image", body.prompt);
}

function imageResponse<T>(response: T): SpanFields {
  const r = asRecord(response) ?? {};
  const data = asArray(r.data);

  return {
    usage: modalityUsage(r.usage, "image"),
    output: data
      ? {
          count: data.length,
          revised_prompts: data
            .map((image) => readString(asRecord(image)?.revised_prompt))
            .filter((prompt): prompt is string => prompt !== undefined),
        }
      : undefined,
  };
}

function audioRequest(body: JsonRecord, label: string, outputType: string): RequestMapping {
  return generationRequest(
    body,
    label,
    outputType,
    outputType === "speech" ? body.input : undefined,
  );
}

function transcriptionResponse<T>(response: T): SpanFields {
  if (typeof response === "string") return { output: response };
  const r = asRecord(response) ?? {};

  return { output: r.text, usage: modalityUsage(r.usage, "text") };
}

function videoCreateRequest(body: JsonRecord): RequestMapping {
  return generationRequest(body, "video", "video", body.prompt);
}

function videoResponse<T>(response: T): SpanFields {
  const r = asRecord(response) ?? {};
  const status = readString(r.status);

  return {
    responseId: readString(r.id),
    responseModel: readString(r.model),
    finishReason: status,
    error:
      status === "failed"
        ? asError(readString(asRecord(r.error)?.message) ?? "video failed")
        : undefined,
    output: {
      status,
      progress: readNumber(r.progress),
      seconds: readString(r.seconds),
      size: readString(r.size),
    },
  };
}

function batchRequest(body: JsonRecord, action: string): RequestMapping {
  return {
    name: `openai.batch.${action}`,
    fields: {
      type: "span",
      input: {
        endpoint: body.endpoint,
        completion_window: body.completion_window,
        input_file_id: body.input_file_id,
        batch_id: body.batch_id,
      },
      attributes: { "gen_ai.operation.name": `openai.batch.${action}` },
    },
  };
}

function batchResponse<T>(response: T): SpanFields {
  const r = asRecord(response) ?? {};
  const id = readString(r.id);
  const status = readString(r.status);
  const attributes: Record<string, string> = {};

  if (id !== undefined) attributes["openai.batch.id"] = id;

  if (status !== undefined) attributes["openai.batch.status"] = status;

  return {
    responseId: id,
    attributes,
    output: {
      status: r.status,
      endpoint: r.endpoint,
      request_counts: r.request_counts,
    },
  };
}

function baseURLHost<T>(baseURL: T): string | undefined {
  const url = readString(baseURL);

  if (url === undefined) return undefined;

  try {
    return new URL(url).hostname;
  } catch {
    return undefined;
  }
}

function providerForClient<T>(client: T): string {
  const clientRecord = asRecord(client);

  if (readString(clientRecord?.apiVersion) !== undefined) return "azure.ai.openai";
  const host = baseURLHost(clientRecord?.baseURL)?.replace(/\.$/, "");

  if (host === "openrouter.ai" || host?.endsWith(".openrouter.ai")) return "openrouter";

  const compatibleProviders: Array<[string, string]> = [
    ["groq.com", "groq"],
    ["x.ai", "x_ai"],
    ["deepseek.com", "deepseek"],
    ["together.xyz", "together_ai"],
    ["fireworks.ai", "fireworks_ai"],
  ];

  for (const [domain, provider] of compatibleProviders) {
    if (host === domain || host?.endsWith(`.${domain}`)) return provider;
  }

  return "openai";
}

function providerForResource<T>(resource: T): string {
  const client = asRecord(resource)?._client;

  return providerForClient(client);
}

function isWrapped<T>(fn: T): fn is T & WrappedFunction {
  return fn instanceof Function && (fn as T & WrappedFunction)[WRAPPED] === true;
}

function markWrapped<T extends WrappedFunction>(
  fn: T,
  original: (...args: never[]) => JsonValue | Promise<JsonValue>,
): T {
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

type TracedResult<T> = Promise<T> & {
  parse(): Promise<T>;
  asResponse(): Promise<Response>;
  withResponse(): Promise<WrappedResponse<T>>;
  _thenUnwrap<U>(transform: (data: T, ...args: never[]) => U): TracedResult<U>;
};

interface TracePromiseContext {
  end: (fields?: SpanFields) => void;
  span: SpanHandle;
  startedAt: number;
  streaming: boolean;
  operation: Operation;
  injectedUsage: boolean;
  mapResponse: <R>(response: R) => SpanFields;
}

type InnerAPIPromise<T = JsonValue> = Promise<T> & {
  parse?: () => Promise<T>;
  asResponse?: () => Promise<Response>;
  withResponse?: () => Promise<WrappedResponse<T>>;
  _thenUnwrap?: <U>(transform: (data: T, ...args: never[]) => U) => InnerAPIPromise<U>;
};

function mapRawResponse(response: Response): SpanFields {
  const requestId = response.headers.get("x-request-id");

  const fields: SpanFields = {
    attributes: { "http.response.status_code": response.status },
  };

  if (requestId) fields.responseId = requestId;

  return fields;
}

function finalizeParsedValue<T>(value: T, ctx: TracePromiseContext): T | Stream<JsonValue> {
  if (ctx.streaming) {
    try {
      return wrapStream(value, ctx.operation, ctx.span, ctx.startedAt, ctx.injectedUsage);
    } catch (cause) {
      reportSpanError(ctx.span, cause);
      ctx.end({ attributes: { "telemetry.dev.capture.truncated": true } });

      return value;
    }
  }

  try {
    ctx.end(ctx.mapResponse(value));
  } catch (cause) {
    reportSpanError(ctx.span, cause);
    ctx.end({ attributes: { "telemetry.dev.capture.truncated": true } });
  }

  return value;
}

function rejectMissing(method: string): Promise<never> {
  return Promise.reject(new TypeError(`wrapped result has no ${method}`));
}

function makeTracedPromise<T>(
  inner: InnerAPIPromise<T>,
  ctx: TracePromiseContext,
): TracedResult<T> {
  const source = inner;

  const handleError = <TError>(cause: TError): never => {
    ctx.end({ error: asError(cause) });
    throw cause;
  };

  const onParsed = (value: T) => {
    return finalizeParsedValue(value, ctx) as T;
  };

  const parsed = () =>
    (source.parse ? source.parse() : source.then((value) => value)).then(onParsed, handleError);

  return new Proxy(source, {
    get(target, property) {
      if (property === "parse") return parsed;

      if (property === "then") {
        return (
          onfulfilled: Parameters<Promise<T>["then"]>[0],
          onrejected: Parameters<Promise<T>["then"]>[1],
        ) => parsed().then(onfulfilled, onrejected);
      }

      if (property === "catch") {
        return (onrejected: Parameters<Promise<T>["catch"]>[0]) => parsed().catch(onrejected);
      }

      if (property === "finally") {
        return (onfinally: Parameters<Promise<T>["finally"]>[0]) => parsed().finally(onfinally);
      }

      if (property === "asResponse") {
        return () => {
          if (!target.asResponse) return rejectMissing("asResponse");

          return target.asResponse().then((response) => {
            ctx.end(mapRawResponse(response));

            return response;
          }, handleError);
        };
      }

      if (property === "withResponse") {
        return () => {
          if (!target.withResponse) return rejectMissing("withResponse");

          return target.withResponse().then(
            ({ data, response, request_id }) => ({
              data: finalizeParsedValue(data, ctx) as T,
              response,
              request_id,
            }),
            handleError,
          );
        };
      }

      if (property === "_thenUnwrap") {
        return <U>(transform: (data: T, ...args: never[]) => U): TracedResult<U> => {
          if (!target._thenUnwrap) {
            const missing = makeTracedPromise<U>(
              Promise.reject(new TypeError("wrapped result has no _thenUnwrap")),
              ctx,
            );

            return missing;
          }

          return makeTracedPromise(target._thenUnwrap(transform), ctx);
        };
      }

      const value = target[property as keyof InnerAPIPromise<T>];

      return value instanceof Function ? value.bind(target) : value;
    },
  }) as TracedResult<T>;
}

interface ChatChoiceState {
  role?: string;
  roleResolved: boolean;
  content: string;
  refusal: string;
  functionCall?: JsonRecord;
  unresolvedFunctionScalars: Set<"name">;
  toolCalls: Map<number, JsonRecord>;
  unresolvedToolScalars: Map<number, Set<"id" | "type" | "function.name" | "custom.name">>;
  terminal: boolean;
}

interface ChatMessageFields {
  role?: string;
  content?: string | null;
  refusal?: string;
  function_call?: JsonRecord;
  tool_calls?: JsonRecord[];
}

interface ChatCaptureBudget {
  remainingBytes: number;
  remainingItems: number;
  truncated: boolean;
}

interface ChatCaptureReservation {
  bytes: number;
  items: number;
}

interface ChatToolCallDelta {
  index?: number;
  id?: string;
  type?: string;
  functionName?: string;
  functionArguments?: string;
  customName?: string;
  customInput?: string;
  readFailed: boolean;
}

function reserveChatStructure(budget: ChatCaptureBudget, bytes: number, items = 0): boolean {
  if (budget.truncated || budget.remainingBytes < bytes || budget.remainingItems < items) {
    budget.truncated = true;

    return false;
  }

  budget.remainingBytes -= bytes;
  budget.remainingItems -= items;

  return true;
}

function captureChatString(
  value: string,
  budget: ChatCaptureBudget,
  structureBytes: number,
  includeQuotes: boolean,
  includeItem: boolean,
  recoverable = false,
): string | undefined {
  const quoteRefund = includeQuotes ? 0 : 2;
  const availableBytes = budget.remainingBytes - structureBytes + quoteRefund;

  if (budget.truncated || availableBytes < 0 || (includeItem && budget.remainingItems < 1)) {
    if (!recoverable) budget.truncated = true;

    return undefined;
  }

  const capture = boundedCaptureDetails(value, {
    maxBytes: availableBytes,
    maxItems: 1,
  });

  const captured = readString(capture.value);

  if (capture.truncated || captured === undefined) {
    if (!recoverable) budget.truncated = true;

    return undefined;
  }

  budget.remainingBytes -= capture.bytes - quoteRefund + structureBytes;

  if (includeItem) budget.remainingItems -= 1;

  return captured;
}

function releaseCapturedChatString(
  value: string,
  budget: ChatCaptureBudget,
  structureBytes: number,
  includeQuotes: boolean,
  includeItem: boolean,
): void {
  const capture = boundedCaptureDetails(value);
  budget.remainingBytes += capture.bytes - (includeQuotes ? 0 : 2) + structureBytes;

  if (includeItem) budget.remainingItems += 1;
}

function releaseCapturedChatField(
  key: string,
  value: string,
  budget: ChatCaptureBudget,
  hasSibling: boolean,
): void {
  const retained = boundedCaptureDetails({ [key]: value });
  const empty = boundedCaptureDetails({});
  budget.remainingBytes += retained.bytes - empty.bytes + (hasSibling ? 1 : 0);
  budget.remainingItems += retained.items - empty.items;
}

function replaceChatString(
  current: string,
  value: string,
  budget: ChatCaptureBudget,
): string | undefined {
  const held = boundedCaptureDetails(current);

  const capture = boundedCaptureDetails(value, {
    maxBytes: budget.remainingBytes + held.bytes,
    maxItems: budget.remainingItems + held.items,
  });

  const captured = readString(capture.value);

  if (capture.truncated || captured === undefined) return undefined;

  budget.remainingBytes += held.bytes - capture.bytes;
  budget.remainingItems += held.items - capture.items;

  return captured;
}

function replaceChatNullWithString(value: string, budget: ChatCaptureBudget): string | undefined {
  const held = boundedCaptureDetails(null);

  const capture = boundedCaptureDetails(value, {
    maxBytes: budget.remainingBytes + held.bytes,
    maxItems: budget.remainingItems + held.items,
  });

  const captured = readString(capture.value);

  if (capture.truncated || captured === undefined) {
    budget.truncated = true;

    return undefined;
  }

  budget.remainingBytes += held.bytes - capture.bytes;
  budget.remainingItems += held.items - capture.items;

  return captured;
}

function replaceFinishReason(
  index: number,
  finishReason: string,
  states: Map<number, string>,
  reservations: Map<number, ChatCaptureReservation>,
  rejected: Set<number>,
  budget: ChatCaptureBudget,
): void {
  const held = reservations.get(index) ?? { bytes: 0, items: 0 };

  const capture = boundedCaptureDetails(
    { index, finish_reason: finishReason },
    {
      maxBytes: budget.remainingBytes + held.bytes,
      maxItems: budget.remainingItems + held.items,
    },
  );

  const captured = asRecord(capture.value);
  const capturedIndex = readNumber(captured?.index);
  const capturedReason = readString(captured?.finish_reason);

  if (capture.truncated || capturedIndex === undefined || capturedReason === undefined) {
    budget.remainingBytes += held.bytes;
    budget.remainingItems += held.items;
    reservations.delete(index);
    states.delete(index);

    if (rejected.has(index) || rejected.size < CHAT_STREAM_CAPTURE_MAX_ITEMS) rejected.add(index);
    else budget.truncated = true;

    return;
  }

  budget.remainingBytes += held.bytes - capture.bytes;
  budget.remainingItems += held.items - capture.items;
  reservations.set(capturedIndex, { bytes: capture.bytes, items: capture.items });
  states.set(capturedIndex, capturedReason);
  rejected.delete(capturedIndex);
}

function readChatToolCallDelta(
  delta: JsonRecord,
  budget: ChatCaptureBudget | undefined,
  reportError: (cause: unknown) => void,
): ChatToolCallDelta {
  const failure = { value: false };
  const incomingFunction = asRecord(readOwn(delta, "function", budget, failure, reportError));
  const incomingCustom = asRecord(readOwn(delta, "custom", budget, failure, reportError));

  return {
    index: readNumber(readOwn(delta, "index", budget, failure, reportError)),
    id: readString(readOwn(delta, "id", budget, failure, reportError)),
    type: readString(readOwn(delta, "type", budget, failure, reportError)),
    functionName: incomingFunction
      ? readString(readOwn(incomingFunction, "name", budget, failure, reportError))
      : undefined,
    functionArguments: incomingFunction
      ? readString(readOwn(incomingFunction, "arguments", budget, failure, reportError))
      : undefined,
    customName: incomingCustom
      ? readString(readOwn(incomingCustom, "name", budget, failure, reportError))
      : undefined,
    customInput: incomingCustom
      ? readString(readOwn(incomingCustom, "input", budget, failure, reportError))
      : undefined,
    readFailed: failure.value,
  };
}

function captureToolCallPayload(
  current: JsonRecord,
  payloadKey: "function" | "custom",
  name: string | undefined,
  valueKey: "arguments" | "input",
  valueText: string | undefined,
  unresolved: Set<"id" | "type" | "function.name" | "custom.name">,
  budget: ChatCaptureBudget,
): boolean {
  if (name === undefined && valueText === undefined) return false;

  let changed = false;
  const unresolvedName = `${payloadKey}.name` as "function.name" | "custom.name";
  const hadPayload = asRecord(current[payloadKey]) !== undefined;
  const payload = { ...asRecord(current[payloadKey]) };

  if (!hadPayload) {
    if (!reserveChatStructure(budget, JSON.stringify(payloadKey).length + 4, 1)) return false;
    current[payloadKey] = payload;
    changed = true;
  }

  if (name !== undefined) {
    const currentName = readString(payload.name);

    const capturedName =
      currentName === undefined
        ? captureChatString(name, budget, JSON.stringify("name").length + 2, true, true, true)
        : replaceChatString(currentName, name, budget);

    if (capturedName !== undefined) {
      payload.name = capturedName;
      unresolved.delete(unresolvedName);
      changed = true;
    } else {
      if (currentName !== undefined) {
        releaseCapturedChatString(
          currentName,
          budget,
          JSON.stringify("name").length + 2,
          true,
          true,
        );
        delete payload.name;
        changed = true;
      }

      unresolved.add(unresolvedName);
    }
  }

  if (!budget.truncated && valueText !== undefined) {
    const existingValue = readString(payload[valueKey]);

    const capturedValue = captureChatString(
      valueText,
      budget,
      existingValue === undefined ? JSON.stringify(valueKey).length + 2 : 0,
      existingValue === undefined,
      existingValue === undefined,
    );

    if (capturedValue !== undefined) {
      payload[valueKey] = `${existingValue ?? ""}${capturedValue}`;
      changed = true;
    }
  }

  current[payloadKey] = payload;

  return changed;
}

function captureToolCallDelta(
  state: ChatChoiceState,
  delta: ChatToolCallDelta,
  budget: ChatCaptureBudget,
): void {
  const index = delta.index ?? state.toolCalls.size;
  const hadToolCall = state.toolCalls.has(index);
  const current = { ...state.toolCalls.get(index) };
  const unresolved = new Set(state.unresolvedToolScalars.get(index));
  const { id, type, functionName, functionArguments, customName, customInput } = delta;
  let changed = false;

  const save = () => {
    if (changed || unresolved.size > 0 || !hadToolCall) state.toolCalls.set(index, current);

    if (unresolved.size > 0) state.unresolvedToolScalars.set(index, unresolved);
    else state.unresolvedToolScalars.delete(index);
  };

  if (
    id === undefined &&
    type === undefined &&
    functionName === undefined &&
    functionArguments === undefined &&
    customName === undefined &&
    customInput === undefined
  )
    return;

  if (!state.toolCalls.has(index)) {
    const firstToolCall = state.toolCalls.size === 0;

    if (
      !reserveChatStructure(
        budget,
        2 +
          (firstToolCall ? JSON.stringify("tool_calls").length + 4 : 1) +
          (firstToolCall && state.content.length === 0 && state.functionCall === undefined
            ? JSON.stringify("content").length + 6
            : 0),
        1 +
          (firstToolCall ? 1 : 0) +
          (firstToolCall && state.content.length === 0 && state.functionCall === undefined ? 1 : 0),
      )
    ) {
      return;
    }
  }

  if (id !== undefined) {
    const currentId = readString(current.id);

    const capturedId =
      currentId === undefined
        ? captureChatString(id, budget, JSON.stringify("id").length + 2, true, true, true)
        : replaceChatString(currentId, id, budget);

    if (capturedId !== undefined) {
      current.id = capturedId;
      unresolved.delete("id");
      changed = true;
    } else {
      if (currentId !== undefined) {
        releaseCapturedChatString(currentId, budget, JSON.stringify("id").length + 2, true, true);
        delete current.id;
        changed = true;
      }

      unresolved.add("id");
    }
  }

  if (type !== undefined) {
    const currentType = readString(current.type);

    const capturedType =
      currentType === undefined
        ? captureChatString(type, budget, JSON.stringify("type").length + 2, true, true, true)
        : replaceChatString(currentType, type, budget);

    if (capturedType !== undefined) {
      current.type = capturedType;
      unresolved.delete("type");
      changed = true;
    } else {
      if (currentType !== undefined) {
        releaseCapturedChatString(
          currentType,
          budget,
          JSON.stringify("type").length + 2,
          true,
          true,
        );
        delete current.type;
        changed = true;
      }

      unresolved.add("type");
    }
  }

  changed =
    captureToolCallPayload(
      current,
      "function",
      functionName,
      "arguments",
      functionArguments,
      unresolved,
      budget,
    ) || changed;

  changed =
    captureToolCallPayload(
      current,
      "custom",
      customName,
      "input",
      customInput,
      unresolved,
      budget,
    ) || changed;

  save();
}

function captureFunctionCallDelta(
  state: ChatChoiceState,
  name: string | undefined,
  argumentsText: string | undefined,
  budget: ChatCaptureBudget,
): void {
  if (name === undefined && argumentsText === undefined) return;

  const hadFunctionCall = state.functionCall !== undefined;
  const current = { ...state.functionCall };

  if (!hadFunctionCall) {
    const reserveNullContent = state.content.length === 0 && state.toolCalls.size === 0;

    if (
      !reserveChatStructure(
        budget,
        JSON.stringify("function_call").length +
          4 +
          (reserveNullContent ? JSON.stringify("content").length + 6 : 0),
        1 + (reserveNullContent ? 1 : 0),
      )
    )
      return;
    state.functionCall = current;
  }

  if (name !== undefined) {
    const currentName = readString(current.name);

    const capturedName =
      currentName === undefined
        ? captureChatString(name, budget, JSON.stringify("name").length + 2, true, true, true)
        : replaceChatString(currentName, name, budget);

    if (capturedName !== undefined) {
      current.name = capturedName;
      state.unresolvedFunctionScalars.delete("name");
    } else {
      if (currentName !== undefined) {
        releaseCapturedChatString(
          currentName,
          budget,
          JSON.stringify("name").length + 2,
          true,
          true,
        );
        delete current.name;
      }

      state.unresolvedFunctionScalars.add("name");
    }
  }

  if (!budget.truncated && argumentsText !== undefined) {
    const existingArguments = readString(current.arguments);

    const capturedArguments = captureChatString(
      argumentsText,
      budget,
      existingArguments === undefined ? JSON.stringify("arguments").length + 2 : 0,
      existingArguments === undefined,
      existingArguments === undefined,
    );

    if (capturedArguments !== undefined)
      current.arguments = `${existingArguments ?? ""}${capturedArguments}`;
  }

  state.functionCall = current;
}

function chatMessage(state: ChatChoiceState) {
  const fields: ChatMessageFields = {};

  if (state.roleResolved) fields.role = state.role ?? "assistant";

  if (state.content.length > 0) fields.content = state.content;
  else if (state.functionCall || state.toolCalls.size > 0) fields.content = null;

  if (state.refusal.length > 0) fields.refusal = state.refusal;

  if (state.functionCall) {
    const functionCall = { ...state.functionCall };

    if (state.unresolvedFunctionScalars.has("name")) delete functionCall.name;
    fields.function_call = functionCall;
  }

  if (state.toolCalls.size > 0) {
    fields.tool_calls = [...state.toolCalls.entries()]
      .sort(([left], [right]) => left - right)
      .map(([index, toolCall]) => {
        const captured = { ...toolCall };
        const unresolved = state.unresolvedToolScalars.get(index);

        if (unresolved?.has("id")) delete captured.id;

        if (unresolved?.has("type")) delete captured.type;

        if (unresolved?.has("function.name")) {
          const fn = { ...asRecord(captured.function) };
          delete fn.name;
          captured.function = fn;
        }

        if (unresolved?.has("custom.name")) {
          const custom = { ...asRecord(captured.custom) };
          delete custom.name;
          captured.custom = custom;
        }

        return captured;
      });
  }

  return fields;
}

function chatOutput(states: Map<number, ChatChoiceState>): ChatMessageFields[] {
  return [...states.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, state]) => chatMessage(state));
}

function chatStateHasFieldBesidesRole(state: ChatChoiceState): boolean {
  return (
    state.content.length > 0 ||
    state.refusal.length > 0 ||
    state.functionCall !== undefined ||
    state.toolCalls.size > 0
  );
}

function chatPartialFields(
  states: Map<number, ChatChoiceState>,
  finishReasonStates: Map<number, string>,
  usage: SpanFields["usage"],
  captureOutput: boolean,
  captureTruncated: boolean,
): SpanFields {
  const finishReasons = [...finishReasonStates.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, reason]) => reason);

  const fields: SpanFields = {
    output: captureOutput && states.size > 0 ? chatOutput(states) : undefined,
    usage,
    finishReason: finishReasons[0],
  };

  if (finishReasons.length > 1 || captureTruncated) {
    const attributes: NonNullable<SpanFields["attributes"]> = {};

    if (finishReasons.length > 1) attributes["gen_ai.response.finish_reasons"] = finishReasons;

    if (captureTruncated) attributes["telemetry.dev.capture.truncated"] = true;

    fields.attributes = attributes;
  }

  return fields;
}

function recordChatChunk<T>(
  chunk: T,
  states: Map<number, ChatChoiceState>,
  finishReasonStates: Map<number, string>,
  finishReasonReservations: Map<number, ChatCaptureReservation>,
  rejectedFinishReasons: Set<number>,
  unterminatedChoices: Set<number>,
  outputBudget: ChatCaptureBudget,
  finishReasonBudget: ChatCaptureBudget,
  captureOutput: boolean,
  reportError: (cause: unknown) => void,
) {
  const c = asRecord(chunk) ?? {};
  const fieldReadFailed = { value: false };
  let hasOutput = false;
  let remainingToolCalls = CHAT_STREAM_CAPTURE_MAX_ITEMS;
  const choices = asArray(readOwn(c, "choices", outputBudget, fieldReadFailed, reportError)) ?? [];

  for (const choice of choices.slice(0, CHAT_STREAM_CAPTURE_MAX_ITEMS)) {
    const choiceRecord = asRecord(choice) ?? {};
    const captureBudget = captureOutput ? outputBudget : undefined;

    const choiceIndex =
      readNumber(readOwn(choiceRecord, "index", captureBudget, fieldReadFailed, reportError)) ?? 0;

    const delta =
      asRecord(readOwn(choiceRecord, "delta", captureBudget, fieldReadFailed, reportError)) ?? {};

    const content = readString(
      readOwn(delta, "content", captureBudget, fieldReadFailed, reportError),
    );

    const refusal = readString(
      readOwn(delta, "refusal", captureBudget, fieldReadFailed, reportError),
    );

    const audio = asRecord(readOwn(delta, "audio", undefined, fieldReadFailed, reportError));

    const audioData = audio
      ? readString(readOwn(audio, "data", undefined, fieldReadFailed, reportError))
      : undefined;

    const legacyFunction = asRecord(
      readOwn(delta, "function_call", captureBudget, fieldReadFailed, reportError),
    );

    const legacyName = legacyFunction
      ? readString(readOwn(legacyFunction, "name", captureBudget, fieldReadFailed, reportError))
      : undefined;

    const legacyArguments = legacyFunction
      ? readString(
          readOwn(legacyFunction, "arguments", captureBudget, fieldReadFailed, reportError),
        )
      : undefined;

    const toolCallDeltas =
      asArray(readOwn(delta, "tool_calls", captureBudget, fieldReadFailed, reportError)) ?? [];

    if (
      (content !== undefined && content.length > 0) ||
      (refusal !== undefined && refusal.length > 0) ||
      (audioData !== undefined && audioData.length > 0) ||
      (legacyArguments !== undefined && legacyArguments.length > 0)
    ) {
      hasOutput = true;
    }

    const finishReason = readString(
      readOwn(choiceRecord, "finish_reason", finishReasonBudget, fieldReadFailed, reportError),
    );

    if (finishReason !== undefined) {
      unterminatedChoices.delete(choiceIndex);
      replaceFinishReason(
        choiceIndex,
        finishReason,
        finishReasonStates,
        finishReasonReservations,
        rejectedFinishReasons,
        finishReasonBudget,
      );
    } else if (!finishReasonStates.has(choiceIndex) && !rejectedFinishReasons.has(choiceIndex)) {
      if (
        unterminatedChoices.has(choiceIndex) ||
        unterminatedChoices.size < CHAT_STREAM_CAPTURE_MAX_ITEMS
      ) {
        unterminatedChoices.add(choiceIndex);
      } else {
        finishReasonBudget.truncated = true;
      }
    }

    const hadState = states.has(choiceIndex);
    const retainedState = captureOutput ? states.get(choiceIndex) : undefined;

    if (retainedState && finishReason !== undefined) retainedState.terminal = true;
    let state = retainedState;

    if (captureOutput && !outputBudget.truncated && !state) {
      if (reserveChatStructure(outputBudget, states.size > 0 ? 1 : 0)) {
        const initialState: ChatChoiceState = {
          content: "",
          refusal: "",
          roleResolved: true,
          terminal: finishReason !== undefined,
          unresolvedFunctionScalars: new Set(),
          toolCalls: new Map(),
          unresolvedToolScalars: new Map(),
        };

        const initialCapture = boundedCaptureDetails(chatMessage(initialState), {
          maxBytes: outputBudget.remainingBytes,
          maxItems: outputBudget.remainingItems,
        });

        if (initialCapture.truncated || initialCapture.value === undefined) {
          outputBudget.truncated = true;
        } else {
          outputBudget.remainingBytes -= initialCapture.bytes;
          outputBudget.remainingItems -= initialCapture.items;
          state = initialState;
          states.set(choiceIndex, state);
        }
      }
    }

    const role = state
      ? readString(readOwn(delta, "role", outputBudget, fieldReadFailed, reportError))
      : undefined;

    if (
      state &&
      role !== undefined &&
      (!state.roleResolved || role !== (state.role ?? "assistant"))
    ) {
      const capturedRole = state.roleResolved
        ? replaceChatString(state.role ?? "assistant", role, outputBudget)
        : captureChatString(
            role,
            outputBudget,
            JSON.stringify("role").length + (chatStateHasFieldBesidesRole(state) ? 2 : 1),
            true,
            true,
            true,
          );

      if (capturedRole === undefined) {
        if (state.roleResolved)
          releaseCapturedChatField(
            "role",
            state.role ?? "assistant",
            outputBudget,
            chatStateHasFieldBesidesRole(state),
          );
        state.role = undefined;
        state.roleResolved = false;
      } else {
        state.role = capturedRole;
        state.roleResolved = true;
      }
    }

    if (state && !outputBudget.truncated && content) {
      const hasContent = state.content.length > 0;

      const capturedContent =
        !hasContent && (state.functionCall !== undefined || state.toolCalls.size > 0)
          ? replaceChatNullWithString(content, outputBudget)
          : captureChatString(
              content,
              outputBudget,
              hasContent ? 0 : JSON.stringify("content").length + 2,
              !hasContent,
              !hasContent,
            );

      if (capturedContent !== undefined) state.content += capturedContent;
    }

    if (state && !outputBudget.truncated && refusal) {
      const hasRefusal = state.refusal.length > 0;

      const capturedRefusal = captureChatString(
        refusal,
        outputBudget,
        hasRefusal ? 0 : JSON.stringify("refusal").length + 2,
        !hasRefusal,
        !hasRefusal,
      );

      if (capturedRefusal !== undefined) state.refusal += capturedRefusal;
    }

    if (state) captureFunctionCallDelta(state, legacyName, legacyArguments, outputBudget);

    const retainedToolCallDeltas = toolCallDeltas.slice(0, remainingToolCalls);
    remainingToolCalls -= retainedToolCallDeltas.length;

    if (retainedToolCallDeltas.length < toolCallDeltas.length) {
      hasOutput = true;

      if (captureOutput) outputBudget.truncated = true;
    }

    for (const rawToolCall of retainedToolCallDeltas) {
      if (hasOutput && !state) break;

      const toolCall = readChatToolCallDelta(
        asRecord(rawToolCall) ?? {},
        captureBudget,
        reportError,
      );

      if (toolCall.readFailed) {
        if (
          !hadState &&
          state &&
          state.role === undefined &&
          state.content.length === 0 &&
          state.refusal.length === 0 &&
          state.functionCall === undefined &&
          state.toolCalls.size === 0
        )
          states.delete(choiceIndex);
        break;
      }

      if (toolCall.functionArguments || toolCall.customInput) hasOutput = true;

      if (state) captureToolCallDelta(state, toolCall, outputBudget);
    }
  }

  if (choices.length > CHAT_STREAM_CAPTURE_MAX_ITEMS) {
    hasOutput = true;

    if (captureOutput) outputBudget.truncated = true;
    finishReasonBudget.truncated = true;
  }

  const responseId = readString(readOwn(c, "id", undefined, fieldReadFailed, reportError));
  const responseModel = readString(readOwn(c, "model", undefined, fieldReadFailed, reportError));
  const usage = chatUsage(readOwn(c, "usage", undefined, fieldReadFailed, reportError));

  if (fieldReadFailed.value) {
    outputBudget.truncated = true;
    finishReasonBudget.truncated = true;
  }

  return {
    responseId,
    responseModel,
    usage,
    hasOutput,
  };
}

// Older cores enforce captureOutput and masking when the span ends but expose no policy, so
// withhold incomplete output from their mask. Retention uses the fixed stream bounds either way;
// maxAttributeLength is only here to satisfy the policy type.
const LEGACY_CAPTURE_POLICY: NonNullable<SpanHandle["capturePolicy"]> = {
  output: true,
  mask: true,
  maxAttributeLength: 65_536,
};

function reportSpanError(span: SpanHandle, cause: unknown): void {
  try {
    span.reportError?.(cause);
  } catch {
    return;
  }
}

function createSpanErrorReporter(span: SpanHandle): (cause: unknown) => void {
  let reported = false;

  return (cause) => {
    if (reported) return;
    reported = true;
    reportSpanError(span, cause);
  };
}

function isSyntheticUsageChunk<T>(chunk: T): boolean {
  const c = asRecord(chunk);

  return !!c?.usage && (asArray(c.choices)?.length ?? 0) === 0;
}

function createObservedChatStream(
  source: Stream<JsonValue>,
  span: SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
  hideSyntheticUsage: boolean,
): Stream<JsonValue> {
  const capturePolicy = span.capturePolicy ?? LEGACY_CAPTURE_POLICY;
  const captureOutput = capturePolicy.output;
  const maskOutputWhenIncomplete = capturePolicy.mask;
  const captureLimit = CHAT_STREAM_CAPTURE_MAX_BYTES;
  const reportError = createSpanErrorReporter(span);

  async function* iterator() {
    const states = new Map<number, ChatChoiceState>();
    const finishReasonStates = new Map<number, string>();
    const finishReasonReservations = new Map<number, ChatCaptureReservation>();
    const rejectedFinishReasons = new Set<number>();
    const unterminatedChoices = new Set<number>();

    const outputBudget: ChatCaptureBudget = {
      remainingBytes: Math.max(captureLimit - 2, 0),
      remainingItems: CHAT_STREAM_CAPTURE_MAX_ITEMS - 1,
      truncated: captureOutput && captureLimit < 2,
    };

    const finishReasonBudget: ChatCaptureBudget = {
      remainingBytes: CHAT_STREAM_CAPTURE_MAX_BYTES,
      remainingItems: CHAT_STREAM_CAPTURE_MAX_ITEMS,
      truncated: false,
    };

    let usage: SpanFields["usage"];
    let sawFirst = false;
    let completedNormally = false;
    let terminalError: Error | undefined;

    try {
      for await (const chunk of source) {
        const receivedAt = performance.now();
        let update: ReturnType<typeof recordChatChunk>;

        try {
          update = recordChatChunk(
            chunk,
            states,
            finishReasonStates,
            finishReasonReservations,
            rejectedFinishReasons,
            unterminatedChoices,
            outputBudget,
            finishReasonBudget,
            captureOutput,
            reportError,
          );
        } catch (cause) {
          outputBudget.truncated = true;
          finishReasonBudget.truncated = true;
          reportError(cause);
          update = {
            responseId: undefined,
            responseModel: undefined,
            usage: undefined,
            hasOutput: false,
          };
        }

        if (update.hasOutput) span.recordOutputChunk?.(receivedAt);

        if (!sawFirst) {
          sawFirst = true;
          span.update({
            timeToFirstChunkMs: performance.now() - startedAt,
            responseId: update.responseId,
            responseModel: update.responseModel,
          });
        }

        if (update.usage) usage = update.usage;

        let syntheticUsage = false;

        if (hideSyntheticUsage) {
          try {
            syntheticUsage = isSyntheticUsageChunk(chunk);
          } catch (cause) {
            outputBudget.truncated = true;
            finishReasonBudget.truncated = true;
            reportError(cause);
          }
        }

        if (!syntheticUsage) yield chunk;
      }

      completedNormally = true;
    } catch (error) {
      terminalError = asError(error);
      throw error;
    } finally {
      const outputIncomplete =
        !completedNormally ||
        unterminatedChoices.size > 0 ||
        (captureOutput &&
          [...states.values()].some(
            (state) =>
              !state.terminal ||
              !state.roleResolved ||
              state.unresolvedFunctionScalars.size > 0 ||
              [...state.unresolvedToolScalars.values()].some((fields) => fields.size > 0),
          ));

      const fields = chatPartialFields(
        states,
        finishReasonStates,
        usage,
        captureOutput &&
          !(maskOutputWhenIncomplete && (outputBudget.truncated || outputIncomplete)),
        outputBudget.truncated ||
          outputIncomplete ||
          finishReasonBudget.truncated ||
          rejectedFinishReasons.size > 0,
      );

      if (terminalError) fields.error = terminalError;
      end(fields);
    }
  }

  return new Stream(() => iterator(), source.controller);
}

function markResponsesCaptureIncomplete(fields: SpanFields, maskOutput: boolean): void {
  fields.attributes = {
    ...fields.attributes,
    "telemetry.dev.capture.truncated": true,
  };

  if (maskOutput) delete fields.output;
}

function createObservedResponsesStream(
  source: Stream<JsonValue>,
  span: SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
): Stream<JsonValue> {
  const capturePolicy = span.capturePolicy ?? LEGACY_CAPTURE_POLICY;
  const captureLimit = CHAT_STREAM_CAPTURE_MAX_BYTES;
  const maskOutputWhenIncomplete = capturePolicy.mask;
  const reportError = createSpanErrorReporter(span);

  async function* iterator() {
    let sawFirst = false;
    let sawTerminalSnapshot = false;
    let partial: SpanFields = {};
    let retainedOutput: SpanFields["output"];
    let terminalError: Error | undefined;

    try {
      for await (const event of source) {
        const receivedAt = performance.now();

        try {
          const e = asRecord(event) ?? {};

          if (responseEventHasOutput(e)) span.recordOutputChunk?.(receivedAt);
          const response = asRecord(e.response);

          if (!sawFirst) {
            sawFirst = true;
            span.update({ timeToFirstChunkMs: performance.now() - startedAt });
          }

          if (response) {
            const next = responsesResponse(response, capturePolicy.output, captureLimit);
            const truncated = next.attributes?.["telemetry.dev.capture.truncated"] === true;

            const fitsWithContent =
              !truncated &&
              next.output !== undefined &&
              !(Array.isArray(next.output) && next.output.length === 0);

            if (fitsWithContent) retainedOutput = next.output;

            partial = next;

            if (truncated && maskOutputWhenIncomplete) delete partial.output;
            else if (retainedOutput !== undefined) partial.output = retainedOutput;
          }

          const terminalSnapshot =
            response !== undefined &&
            (e.type === "response.completed" ||
              e.type === "response.failed" ||
              e.type === "response.incomplete");

          if (terminalSnapshot) sawTerminalSnapshot = true;

          if (e.type === "response.failed") {
            if (!terminalSnapshot)
              markResponsesCaptureIncomplete(partial, maskOutputWhenIncomplete);

            partial = { ...partial, error: responseFailedError(response) };
            end(partial);
          } else if (e.type === "error") {
            markResponsesCaptureIncomplete(partial, maskOutputWhenIncomplete);
            partial = { ...partial, error: responseStreamError(event) };
            end(partial);
          } else if (e.type === "response.completed" || e.type === "response.incomplete") {
            if (!terminalSnapshot)
              markResponsesCaptureIncomplete(partial, maskOutputWhenIncomplete);

            end(partial);
          }
        } catch (cause) {
          reportError(cause);
          markResponsesCaptureIncomplete(partial, maskOutputWhenIncomplete);
          const failure = responsesEventFailure(event);

          if (failure) end({ ...partial, error: failure });
        }

        yield event;
      }
    } catch (error) {
      terminalError = asError(error);
      throw error;
    } finally {
      if (!sawTerminalSnapshot) markResponsesCaptureIncomplete(partial, maskOutputWhenIncomplete);

      if (terminalError) partial.error = terminalError;
      end(partial);
    }
  }

  return new Stream(() => iterator(), source.controller);
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
      "response.shell_call_command.delta",
      "response.output_audio.delta",
      "response.audio.delta",
      "response.audio.transcript.delta",
    ].includes(event.type) &&
    typeof event.delta === "string" &&
    event.delta.length > 0
  );
}

function serializedJsonCharacterLength(character: string): number {
  const code = character.charCodeAt(0);

  if (character === '"' || character === "\\") return 2;

  if (code === 0x08 || code === 0x09 || code === 0x0a || code === 0x0c || code === 0x0d) return 2;

  if (code <= 0x1f || (character.length === 1 && code >= 0xd800 && code <= 0xdfff)) return 6;

  return character.length;
}

function boundedTranscriptText(value: string, maxContentLength = MAX_CAPTURE_LENGTH - 2) {
  let cutoff = 0;
  let serializedLength = 0;

  for (const character of value) {
    const characterLength = serializedJsonCharacterLength(character);

    if (serializedLength + characterLength > maxContentLength) {
      return { value: value.slice(0, cutoff), serializedLength, truncated: true };
    }

    cutoff += character.length;
    serializedLength += characterLength;
  }

  return { value, serializedLength, truncated: false };
}

function createObservedMediaStream(
  source: Stream<JsonValue>,
  operation: "images" | "transcriptions",
  span: SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
): Stream<JsonValue> {
  const capturePolicy = span.capturePolicy ?? LEGACY_CAPTURE_POLICY;
  const captureOutput = capturePolicy.output;
  const maskOutputWhenIncomplete = capturePolicy.mask;
  const captureLimit = MAX_CAPTURE_LENGTH;
  const reportError = createSpanErrorReporter(span);

  async function* iterator() {
    let sawFirst = false;
    let sawTranscriptDelta = false;
    const textChunks: string[] = [];
    let serializedTextLength = 2;
    let deltaCaptureStopped = false;
    let captureTruncated = false;
    let sawTerminal = false;
    let terminalText: string | undefined;
    let usage: SpanFields["usage"];
    let error: Error | undefined;

    try {
      for await (const event of source) {
        try {
          const e = asRecord(event) ?? {};
          const delta = readString(e.delta);
          sawTerminal ||=
            (operation === "transcriptions" && e.type === "transcript.text.done") ||
            (operation === "images" &&
              (e.type === "image_generation.completed" ||
                e.type === "image_generation.failed" ||
                e.type === "image_edit.completed" ||
                e.type === "image_edit.failed"));

          if (operation === "transcriptions" && e.type === "transcript.text.delta") {
            sawTranscriptDelta = true;
          }

          const segmentText =
            operation === "transcriptions" && e.type === "transcript.text.segment"
              ? readString(e.text)
              : undefined;

          const completeText =
            captureOutput && operation === "transcriptions" && e.type === "transcript.text.done"
              ? readString(e.text)
              : undefined;

          const hasOutput =
            operation === "images"
              ? typeof e.b64_json === "string" && e.b64_json.length > 0
              : !!delta || !!segmentText || !!completeText;

          if (hasOutput) {
            span.recordOutputChunk?.(performance.now());

            if (!sawFirst) {
              sawFirst = true;
              span.update({ timeToFirstChunkMs: performance.now() - startedAt });
            }
          }

          const incrementalText =
            delta ?? (!sawTranscriptDelta && segmentText ? segmentText : undefined);

          if (
            captureOutput &&
            operation === "transcriptions" &&
            incrementalText &&
            !deltaCaptureStopped
          ) {
            const capture = boundedTranscriptText(
              incrementalText,
              Math.max(captureLimit - serializedTextLength, 0),
            );

            if (capture.value) textChunks.push(capture.value);
            serializedTextLength += capture.serializedLength;
            deltaCaptureStopped = capture.truncated;
            captureTruncated ||= capture.truncated;
          }

          if (completeText !== undefined) {
            const capture = boundedTranscriptText(completeText, Math.max(captureLimit - 2, 0));
            terminalText = capture.value;
            captureTruncated = capture.truncated;
          }

          if (e.usage) {
            usage = modalityUsage(e.usage, operation === "images" ? "image" : "text");
          }
        } catch (cause) {
          captureTruncated = true;
          reportError(cause);
        }

        yield event;
      }
    } catch (cause) {
      error = asError(cause);
      throw cause;
    } finally {
      const text = captureOutput ? (terminalText ?? textChunks.join("")) : "";
      const captureIncomplete = captureTruncated || !sawTerminal;
      end({
        output:
          operation === "transcriptions" && text && !(maskOutputWhenIncomplete && captureIncomplete)
            ? text
            : undefined,
        usage,
        error,
        attributes: captureIncomplete ? { "telemetry.dev.capture.truncated": true } : undefined,
      });
    }
  }

  return new Stream(() => iterator(), source.controller);
}

function wrapStream<T>(
  value: T,
  operation: Operation,
  span: SpanHandle,
  startedAt: number,
  injectedUsage: boolean,
): T | Stream<JsonValue> {
  const candidate = asRecord(value);

  if (
    !(candidate?.iterator instanceof Function) ||
    !(candidate?.controller instanceof AbortController) ||
    !(candidate?.tee instanceof Function) ||
    !(candidate?.toReadableStream instanceof Function)
  ) {
    return value;
  }

  const end = endOnce(span);
  const source = value as Stream<JsonValue>;
  let observed: Stream<JsonValue>;

  if (operation === "chat") {
    observed = createObservedChatStream(source, span, startedAt, end, injectedUsage);
  } else if (operation === "images" || operation === "transcriptions") {
    observed = createObservedMediaStream(source, operation, span, startedAt, end);
  } else {
    observed = createObservedResponsesStream(source, span, startedAt, end);
  }

  const StreamConstructor = source.constructor as new (
    iterator: () => AsyncIterator<JsonValue>,
    controller: AbortController,
  ) => Stream<JsonValue>;

  const iterator = () => observed[Symbol.asyncIterator]();

  return new StreamConstructor(iterator, source.controller);
}

function withChatUsageInjection(body: JsonRecord) {
  const options = asRecord(body.stream_options);

  if (options?.include_usage === true) return { body, injected: false };

  return {
    body: {
      ...body,
      stream_options: { ...options, include_usage: true },
    },
    injected: true,
  };
}

function wrapCreate<Fn extends (...args: never[]) => JsonValue | Promise<JsonValue>>(
  original: Fn,
  operation: Operation,
  mapRequest: (body: JsonRecord) => RequestMapping,
  mapResponse: <R>(response: R) => SpanFields,
  provider: ProviderResolver,
  injectUsage: boolean,
): Fn {
  if (isWrapped(original)) return original;

  const wrapped = function <This>(this: This, ...args: JsonValue[]) {
    const originalBody = asRecord(args[0]) ?? {};
    const streaming = originalBody.stream === true;

    const { body, injected } =
      operation === "chat" && streaming && injectUsage
        ? withChatUsageInjection(originalBody)
        : { body: originalBody, injected: false };

    const callArgs = body === originalBody ? args : [body, ...args.slice(1)];
    const request = mapRequest(body);
    const span = startSpan(request.name, { ...request.fields, provider: provider(this) });
    const end = endOnce(span);
    const startedAt = performance.now();

    try {
      const result = original.apply(this, callArgs as never[]);
      const promise = result instanceof Promise ? result : Promise.resolve(result);

      return makeTracedPromise(promise, {
        end,
        span,
        startedAt,
        streaming,
        operation,
        injectedUsage: injected,
        mapResponse,
      });
    } catch (cause) {
      end({ error: asError(cause) });
      throw cause;
    }
  };

  const marked = markWrapped(wrapped as WrappedFunction, original);

  return marked as Fn;
}

function wrapLifecycleMethod<Fn extends (...args: never[]) => JsonValue | Promise<JsonValue>>(
  original: Fn,
  request: (args: JsonValue[]) => RequestMapping,
  response: <R>(value: R) => SpanFields,
  provider: ProviderResolver,
): Fn {
  if (isWrapped(original)) return original;

  const wrapped = function <This>(this: This, ...args: JsonValue[]) {
    const mapping = request(args);
    const span = startSpan(mapping.name, { ...mapping.fields, provider: provider(this) });
    const end = endOnce(span);

    try {
      const result = original.apply(this, args as never[]);
      const promise = result instanceof Promise ? result : Promise.resolve(result);

      return makeTracedPromise(promise, {
        end,
        span,
        startedAt: performance.now(),
        streaming: false,
        operation: "responses",
        injectedUsage: false,
        mapResponse: response,
      });
    } catch (cause) {
      end({ error: asError(cause) });
      throw cause;
    }
  };

  return markWrapped(wrapped as WrappedFunction, original) as Fn;
}

function responseRetrieveRequest<T>(responseId: T): RequestMapping {
  return {
    name: "chat unknown",
    fields: {
      type: "generation",
      responseId: readString(responseId),
    },
  };
}

function wrapResponseRetrieve<Fn extends (...args: never[]) => JsonValue | Promise<JsonValue>>(
  original: Fn,
  provider: ProviderResolver,
): Fn {
  if (isWrapped(original)) return original;

  const wrapped = function <This>(this: This, ...args: JsonValue[]) {
    const query = asRecord(args[1]);
    const options = asRecord(args[2]);
    const streaming = query?.stream === true || options?.stream === true;

    if (!streaming) {
      return original.apply(this, args as never[]);
    }

    const request = responseRetrieveRequest(args[0]);
    const span = startSpan(request.name, { ...request.fields, provider: provider(this) });
    const end = endOnce(span);
    const startedAt = performance.now();

    try {
      const result = original.apply(this, args as never[]);
      const promise = result instanceof Promise ? result : Promise.resolve(result);

      return makeTracedPromise(promise, {
        end,
        span,
        startedAt,
        streaming: true,
        operation: "responses",
        injectedUsage: false,
        mapResponse: responsesResponse,
      });
    } catch (cause) {
      end({ error: asError(cause) });
      throw cause;
    }
  };

  return markWrapped(wrapped as WrappedFunction, original) as Fn;
}

function patchInstanceMethod<T extends JsonRecord>(
  target: T,
  key: keyof T,
  operation: Operation,
  request: (body: JsonRecord) => RequestMapping,
  response: <R>(value: R) => SpanFields,
  injectUsage: boolean,
): void {
  const current = target[key];

  // An own wrapper means this instance is already instrumented. A wrapper
  // inherited from instrumentOpenAI()'s prototype patch must still be shadowed
  // by an own wrapper over the underlying original, so this client stays
  // instrumented after uninstrumentOpenAI() restores the prototype.
  if (isWrapped(current) && Object.prototype.hasOwnProperty.call(target, key)) return;
  const original = isWrapped(current) ? current[ORIGINAL] : current;

  if (!(original instanceof Function)) return;
  const bound: (...args: never[]) => JsonValue | Promise<JsonValue> = original.bind(target);

  const wrapped = wrapCreate(
    bound,
    operation,
    request,
    response,
    () => providerForResource(target),
    injectUsage,
  );

  Object.assign(target, { [key]: wrapped });
}

function patchResponseRetrieveInstance(target: JsonRecord & { retrieve?: WrappedFunction }): void {
  const current = target["retrieve"];

  if (isWrapped(current) && Object.prototype.hasOwnProperty.call(target, "retrieve")) return;
  const original = isWrapped(current) ? current[ORIGINAL] : current;

  if (!(original instanceof Function)) return;
  target["retrieve"] = wrapResponseRetrieve(
    original.bind(target) as (...args: never[]) => JsonValue,
    () => providerForResource(target),
  );
}

function patchLifecycleInstance(
  target: JsonRecord,
  key: string,
  request: (args: JsonValue[]) => RequestMapping,
  response: <R>(value: R) => SpanFields,
): void {
  const current = target[key];

  if (isWrapped(current) && Object.prototype.hasOwnProperty.call(target, key)) return;
  const original = isWrapped(current) ? current[ORIGINAL] : current;

  if (!(original instanceof Function)) return;
  Object.assign(target, {
    [key]: wrapLifecycleMethod(original.bind(target) as WrappedFunction, request, response, () =>
      providerForResource(target),
    ),
  });
}

function patchResponseRetrievePrototype(): () => void {
  const rawPrototype: unknown = Responses.prototype;
  const prototype = rawPrototype as JsonRecord & { retrieve?: WrappedFunction };
  const original = prototype.retrieve;

  if (!(original instanceof Function)) return () => {};

  const wrapped = wrapResponseRetrieve(original, providerForResource);
  prototype.retrieve = wrapped;

  return () => {
    if (prototype.retrieve === wrapped) prototype.retrieve = original;
  };
}

function patchPrototype(
  klass: { prototype: object },
  key: string,
  operation: Operation,
  request: (body: JsonRecord) => RequestMapping,
  response: <R>(value: R) => SpanFields,
  injectUsage: boolean,
): () => void {
  const prototype = klass.prototype as JsonRecord & { [key: string]: WrappedFunction | undefined };
  const original = prototype[key];

  if (!(original instanceof Function)) return () => {};

  const wrapped = wrapCreate(
    original,
    operation,
    request,
    response,
    providerForResource,
    injectUsage,
  );

  prototype[key] = wrapped;

  return () => {
    if (prototype[key] === wrapped) prototype[key] = original;
  };
}

function patchLifecyclePrototype(
  klass: { prototype: object },
  key: string,
  request: (args: JsonValue[]) => RequestMapping,
  response: <R>(value: R) => SpanFields,
): () => void {
  const prototype = klass.prototype as Record<string, WrappedFunction | undefined>;
  const original = prototype[key];

  if (!(original instanceof Function)) return () => {};

  const wrapped = wrapLifecycleMethod(original, request, response, providerForResource);
  prototype[key] = wrapped;

  return () => {
    if (prototype[key] === wrapped) prototype[key] = original;
  };
}

let installed = false;
let restorePatches: Array<() => void> = [];

export interface InstrumentOpenAIOptions {
  /**
   * Inject `stream_options: { include_usage: true }` into streamed chat completion
   * requests so token usage is reported in the final chunk. Disabled by default
   * because it changes the request shape and some providers (for example Azure
   * OpenAI "on your data" with `data_sources`) reject `stream_options`.
   */
  injectStreamUsage?: boolean;
}

export interface OpenAIRealtimeEmitter {
  send(event: unknown): void;
  close(...args: unknown[]): void;
  on(event: string, listener: (...args: unknown[]) => void): unknown;
  off(event: string, listener: (...args: unknown[]) => void): unknown;
}

export interface WrapOpenAIRealtimeOptions {
  model?: string;
  provider?: string;
  traceTimeoutMs?: number;
  maxInFlight?: number;
}

interface RealtimeOutputState {
  items: Map<number, JsonRecord>;
  itemBytes: Map<number, number>;
  bytes: number;
  truncated: boolean;
  stopped: boolean;
}

interface RealtimeTrace {
  span: SpanHandle;
  end: (fields?: SpanFields) => void;
  sequence: number;
  startedAt: number;
  sawOutput: boolean;
  requestEventId?: string;
  correlationId: string;
  correlationKey: string;
  timer: ReturnType<typeof setTimeout>;
  output?: RealtimeOutputState;
}

type RealtimeListener = (...args: unknown[]) => void;

interface RealtimeSocket {
  on?: (event: string, listener: RealtimeListener) => unknown;
  off?: (event: string, listener: RealtimeListener) => unknown;
  listenerCount?: (event: string) => number;
  addEventListener?: (event: string, listener: RealtimeListener) => unknown;
  removeEventListener?: (event: string, listener: RealtimeListener) => unknown;
}

const realtimeOutputIndex = (event: JsonRecord): number | undefined => {
  const index = readNumber(event.output_index);

  return Number.isSafeInteger(index) && index! >= 0 && index! < REALTIME_CAPTURE_MAX_ITEMS
    ? index
    : undefined;
};

const stopRealtimeOutputCapture = (state: RealtimeOutputState) => {
  state.truncated = true;
  state.stopped = true;
};

const jsonByteLength = (value: unknown) => new TextEncoder().encode(JSON.stringify(value)).length;

const setRealtimeItem = (state: RealtimeOutputState, index: number, item: JsonRecord) => {
  const previousBytes = state.itemBytes.get(index) ?? 0;
  const separatorBytes = state.items.has(index) || state.items.size === 0 ? 0 : 1;
  const itemBytes = jsonByteLength(item);
  const nextBytes = state.bytes - previousBytes + itemBytes + separatorBytes;

  if (nextBytes > REALTIME_CAPTURE_MAX_BYTES) {
    stopRealtimeOutputCapture(state);

    return false;
  }

  state.items.set(index, item);
  state.itemBytes.set(index, itemBytes);
  state.bytes = nextBytes;

  return true;
};

const refreshRealtimeItemBytes = (state: RealtimeOutputState, index: number) => {
  const item = state.items.get(index);

  return item ? setRealtimeItem(state, index, item) : false;
};

const serializedJsonUtf8Length = (character: string) => {
  const code = character.codePointAt(0)!;

  if (character === '"' || character === "\\") return 2;

  if (code === 0x08 || code === 0x09 || code === 0x0a || code === 0x0c || code === 0x0d) return 2;

  if (code <= 0x1f || (code >= 0xd800 && code <= 0xdfff)) return 6;

  if (code <= 0x7f) return 1;

  if (code <= 0x7ff) return 2;

  if (code <= 0xffff) return 3;

  return 4;
};

const appendRealtimeText = (
  state: RealtimeOutputState,
  index: number,
  current: string | undefined,
  key: string,
  value: string,
) => {
  const overhead = current === undefined ? jsonByteLength({ [key]: "" }) - 1 : 0;

  if (state.bytes + overhead > REALTIME_CAPTURE_MAX_BYTES) {
    stopRealtimeOutputCapture(state);

    return current;
  }

  const available = Math.max(0, REALTIME_CAPTURE_MAX_BYTES - state.bytes - overhead);
  let used = 0;
  let cutoff = 0;

  for (const character of value) {
    const length = serializedJsonUtf8Length(character);

    if (used + length > available) break;
    used += length;
    cutoff += character.length;
  }

  state.bytes += overhead + used;
  state.itemBytes.set(index, (state.itemBytes.get(index) ?? 0) + overhead + used);

  if (cutoff < value.length) stopRealtimeOutputCapture(state);

  return `${current ?? ""}${value.slice(0, cutoff)}`;
};

const realtimeOutputCapture = (state: RealtimeOutputState | undefined) => {
  if (!state || state.items.size === 0) {
    return { output: undefined, truncated: state?.truncated === true };
  }

  const output = [...state.items.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, item]) => item);

  const capture = sanitizeResponsesCapture(output);

  return { output: capture.value, truncated: state.truncated || capture.truncated };
};

const capturedRealtimeRecord = (value: unknown) => {
  const capture = sanitizeResponsesCapture(value);
  const record = asRecord(capture.value);

  return {
    record: Array.isArray(capture.value) ? undefined : record,
    truncated: capture.truncated,
  };
};

const realtimeItem = (
  state: RealtimeOutputState,
  event: JsonRecord,
  type: "message" | "function_call" | "mcp_call",
) => {
  const index = realtimeOutputIndex(event);

  if (index === undefined) {
    stopRealtimeOutputCapture(state);

    return undefined;
  }

  let item = state.items.get(index);

  if (!item) {
    item = {
      type,
      ...(type === "message" ? { role: "assistant", content: [] } : undefined),
      ...(readString(event.item_id) ? { id: readString(event.item_id) } : undefined),
    };

    if (!setRealtimeItem(state, index, item)) return undefined;
  }

  return item;
};

const realtimeContentPart = (state: RealtimeOutputState, event: JsonRecord) => {
  const item = realtimeItem(state, event, "message");
  const contentIndex = readNumber(event.content_index);

  if (
    !item ||
    !Number.isSafeInteger(contentIndex) ||
    contentIndex! < 0 ||
    contentIndex! >= REALTIME_CAPTURE_MAX_ITEMS
  ) {
    stopRealtimeOutputCapture(state);

    return undefined;
  }

  const currentContent = asArray(item.content);
  const currentPart = asRecord(currentContent?.[contentIndex!]);

  if (currentPart && !Array.isArray(currentContent?.[contentIndex!])) return currentPart;

  if (state.stopped) return undefined;

  const content = [...(currentContent ?? [])];

  while (content.length <= contentIndex!) content.push(null);

  const part: JsonRecord = {};
  content[contentIndex!] = part;
  const index = realtimeOutputIndex(event)!;

  if (!setRealtimeItem(state, index, { ...item, content })) return undefined;

  return part;
};

const updateRealtimeOutput = (state: RealtimeOutputState, event: JsonRecord) => {
  if (typeof event.type !== "string") return;
  const type = event.type;

  if (type === "response.output_item.added" || type === "response.output_item.done") {
    const index = realtimeOutputIndex(event);
    const snapshot = capturedRealtimeRecord(event.item);

    if (index === undefined) {
      stopRealtimeOutputCapture(state);

      return;
    }

    if (snapshot.record && (!snapshot.truncated || !state.items.has(index))) {
      if (!setRealtimeItem(state, index, snapshot.record)) return;
    }

    if (snapshot.truncated) stopRealtimeOutputCapture(state);
  } else if (type === "response.content_part.added" || type === "response.content_part.done") {
    const part = realtimeContentPart(state, event);
    const snapshot = capturedRealtimeRecord(event.part);

    if (part && snapshot.record && !snapshot.truncated) {
      const previousPart = { ...part };

      for (const key of Object.keys(part)) delete part[key];
      Object.assign(part, snapshot.record);

      const index = realtimeOutputIndex(event)!;

      if (!refreshRealtimeItemBytes(state, index)) {
        for (const key of Object.keys(part)) delete part[key];
        Object.assign(part, previousPart);

        return;
      }
    }

    if (snapshot.truncated) stopRealtimeOutputCapture(state);
  } else if (type === "response.output_text.delta" || type === "response.output_text.done") {
    if (state.stopped && type.endsWith(".delta")) return;
    const part = realtimeContentPart(state, event);

    if (part) {
      if (part.type !== "output_text") {
        part.type = "output_text";

        if (!refreshRealtimeItemBytes(state, realtimeOutputIndex(event)!)) return;
      }

      const text = type.endsWith(".done") ? readString(event.text) : readString(event.delta);

      if (text !== undefined) {
        const index = realtimeOutputIndex(event)!;

        if (type.endsWith(".done")) {
          const previousText = part.text;
          part.text = text;

          if (!refreshRealtimeItemBytes(state, index)) part.text = previousText;
        } else {
          part.text = appendRealtimeText(state, index, readString(part.text), "text", text);
        }
      }
    }
  } else if (
    type === "response.output_audio_transcript.delta" ||
    type === "response.output_audio_transcript.done"
  ) {
    if (state.stopped && type.endsWith(".delta")) return;
    const part = realtimeContentPart(state, event);

    if (part) {
      if (part.type !== "output_audio") {
        part.type = "output_audio";

        if (!refreshRealtimeItemBytes(state, realtimeOutputIndex(event)!)) return;
      }

      const transcript = type.endsWith(".done")
        ? readString(event.transcript)
        : readString(event.delta);

      if (transcript !== undefined) {
        const index = realtimeOutputIndex(event)!;

        if (type.endsWith(".done")) {
          const previousTranscript = part.transcript;
          part.transcript = transcript;

          if (!refreshRealtimeItemBytes(state, index)) part.transcript = previousTranscript;
        } else {
          part.transcript = appendRealtimeText(
            state,
            index,
            readString(part.transcript),
            "transcript",
            transcript,
          );
        }
      }
    }
  } else if (
    type === "response.function_call_arguments.delta" ||
    type === "response.function_call_arguments.done"
  ) {
    if (state.stopped && type.endsWith(".delta")) return;
    const item = realtimeItem(state, event, "function_call");

    if (item) {
      const index = realtimeOutputIndex(event)!;
      const name = readString(event.name);
      const callId = readString(event.call_id);
      const previousName = item.name;
      const previousCallId = item.call_id;
      let metadataChanged = false;

      if (name && item.name !== name) {
        item.name = name;
        metadataChanged = true;
      }

      if (callId && item.call_id !== callId) {
        item.call_id = callId;
        metadataChanged = true;
      }

      if (metadataChanged && !refreshRealtimeItemBytes(state, index)) {
        item.name = previousName;
        item.call_id = previousCallId;

        return;
      }

      const value = type.endsWith(".done") ? readString(event.arguments) : readString(event.delta);

      if (value !== undefined) {
        if (type.endsWith(".done")) {
          const previousArguments = item.arguments;
          item.arguments = value;

          if (!refreshRealtimeItemBytes(state, index)) item.arguments = previousArguments;
        } else {
          item.arguments = appendRealtimeText(
            state,
            index,
            readString(item.arguments),
            "arguments",
            value,
          );
        }
      }
    }
  } else if (
    type === "response.mcp_call_arguments.delta" ||
    type === "response.mcp_call_arguments.done"
  ) {
    if (state.stopped && type.endsWith(".delta")) return;
    const item = realtimeItem(state, event, "mcp_call");

    if (item) {
      const index = realtimeOutputIndex(event)!;
      const value = type.endsWith(".done") ? readString(event.arguments) : readString(event.delta);

      if (value !== undefined) {
        if (type.endsWith(".done")) {
          const previousArguments = item.arguments;
          item.arguments = value;

          if (!refreshRealtimeItemBytes(state, index)) item.arguments = previousArguments;
        } else {
          item.arguments = appendRealtimeText(
            state,
            index,
            readString(item.arguments),
            "arguments",
            value,
          );
        }
      }
    }
  } else {
    return;
  }
};

const realtimeItemHasOutput = (value: unknown): boolean => {
  const item = asRecord(value);

  if (!item || Array.isArray(value)) return false;

  if (readString(item.arguments) || readString(item.output)) return true;

  return (asArray(item.content) ?? []).some((value) => {
    const part = asRecord(value);

    return !!readString(part?.text) || !!readString(part?.transcript);
  });
};

const realtimeEventHasOutput = (event: JsonRecord): boolean => {
  if (
    typeof event.type === "string" &&
    event.type.endsWith(".delta") &&
    typeof event.delta === "string" &&
    event.delta.length > 0
  ) {
    return true;
  }

  return (
    realtimeItemHasOutput(event.item) ||
    realtimeItemHasOutput({ content: [event.part] }) ||
    !!readString(event.text) ||
    !!readString(event.transcript) ||
    !!readString(event.arguments)
  );
};

export function wrapOpenAIRealtime<T extends OpenAIRealtimeEmitter>(
  connection: T,
  options?: WrapOpenAIRealtimeOptions,
): T {
  if (wrappedRealtimeConnections.has(connection)) return connection;

  if (
    options?.maxInFlight !== undefined &&
    (!Number.isInteger(options.maxInFlight) || options.maxInFlight <= 0)
  ) {
    throw new RangeError("maxInFlight must be a positive integer");
  }

  if (
    options?.traceTimeoutMs !== undefined &&
    (!Number.isFinite(options.traceTimeoutMs) || options.traceTimeoutMs <= 0)
  ) {
    throw new RangeError("traceTimeoutMs must be a positive finite number");
  }

  const pending = new Map<string, RealtimeTrace>();
  const pendingByEventId = new Map<string, RealtimeTrace>();
  const active = new Map<string, RealtimeTrace>();
  const originalSend = connection.send.bind(connection);
  const originalClose = connection.close.bind(connection);
  const socket = (connection as T & { socket?: RealtimeSocket }).socket;
  let listenersAttached = true;
  let closed = false;
  let traceSequence = 0;

  const removeTrace = (trace: RealtimeTrace) => {
    clearTimeout(trace.timer);
    pending.delete(trace.correlationId);

    if (trace.requestEventId) pendingByEventId.delete(trace.requestEventId);

    for (const [responseId, candidate] of active) {
      if (candidate === trace) active.delete(responseId);
    }
  };

  const finishTrace = (trace: RealtimeTrace, fields?: SpanFields, responseDone = false) => {
    removeTrace(trace);
    const hasAuthoritativeOutput = fields?.output !== undefined;
    const partialCapture = hasAuthoritativeOutput ? undefined : realtimeOutputCapture(trace.output);
    const partialOutput = partialCapture?.output;

    const partialTruncated =
      !hasAuthoritativeOutput &&
      (partialCapture?.truncated === true || (!responseDone && partialOutput !== undefined));

    trace.end({
      ...fields,
      output: hasAuthoritativeOutput ? fields.output : partialOutput,
      attributes: partialTruncated
        ? { ...fields?.attributes, "telemetry.dev.capture.truncated": true }
        : fields?.attributes,
    });
  };

  const finishAll = (error: Error) => {
    for (const trace of new Set([...pending.values(), ...active.values()]))
      finishTrace(trace, { error });
  };

  const onEvent = (value: unknown) => {
    const event = asRecord(value) ?? {};

    if (event.type === "error") {
      const error = asRecord(event.error);
      const requestEventId = readString(error?.event_id);

      if (!requestEventId) return;

      const trace = pendingByEventId.get(requestEventId);

      if (trace) {
        finishTrace(trace, {
          error: asError(
            [readString(error?.code), readString(error?.message)].filter(Boolean).join(": "),
          ),
        });
      }

      return;
    }

    const response = asRecord(event.response);

    if (event.type === "response.created" && response) {
      const metadata = asRecord(response.metadata);
      let trace: RealtimeTrace | undefined;

      for (let index = 0; index < 16; index += 1) {
        const key = index === 0 ? REALTIME_CORRELATION_KEY : `${REALTIME_CORRELATION_KEY}_${index}`;
        const correlationId = readString(metadata?.[key]);
        const candidate = correlationId ? pending.get(correlationId) : undefined;

        if (candidate?.correlationKey === key) {
          trace = candidate;
          break;
        }
      }

      const id = readString(response.id);

      if (trace && id) {
        const existing = active.get(id);

        if (existing && existing !== trace) {
          finishTrace(trace, {
            error: new Error(`OpenAI Realtime returned duplicate response id ${id}`),
          });

          return;
        }

        pending.delete(trace.correlationId);
        active.set(id, trace);
      }

      return;
    }

    const responseId = readString(event.response_id) ?? readString(response?.id);
    const trace = responseId ? active.get(responseId) : undefined;

    if (!trace) return;

    if (trace.output) updateRealtimeOutput(trace.output, event);

    if (realtimeEventHasOutput(event)) {
      trace.span.recordOutputChunk?.(performance.now());

      if (!trace.sawOutput) {
        trace.sawOutput = true;
        trace.span.update({ timeToFirstChunkMs: performance.now() - trace.startedAt });
      }
    }

    if (event.type === "response.done" && response) {
      try {
        const status = readString(response.status);
        const statusDetails = asRecord(response.status_details);

        const capture = captureEnabled("output")
          ? sanitizeResponsesCapture(response.output)
          : undefined;

        finishTrace(
          trace,
          {
            responseId,
            responseModel: readString(response.model),
            output: capture?.value,
            usage: modalityUsage(response.usage),
            finishReason: status,
            attributes: capture?.truncated
              ? { "telemetry.dev.capture.truncated": true }
              : undefined,
            error:
              status === "failed"
                ? asError(
                    readString(asRecord(statusDetails?.error)?.message) ??
                      "Realtime response failed",
                  )
                : undefined,
          },
          true,
        );
      } catch (cause) {
        finishTrace(trace, { error: asError(cause) }, true);
      }
    }
  };

  const onError = (cause: unknown) => {
    const nested = asRecord(cause)?.error;
    const error = asError(nested ?? cause);
    finishAll(error);

    if (socket?.listenerCount?.("error") === 1) throw error;
  };

  const detachListeners = () => {
    if (!listenersAttached) return;
    listenersAttached = false;
    connection.off("event", onEvent);

    if (socket?.off) socket.off("close", onClose);
    else socket?.removeEventListener?.("close", onClose);

    if (socket?.off) socket.off("error", onError);
    else socket?.removeEventListener?.("error", onError);
  };

  const onClose = () => {
    closed = true;
    finishAll(new Error("OpenAI Realtime connection closed before response.done"));
    detachListeners();
  };

  connection.send = (value: unknown) => {
    const event = asRecord(value);

    if (event?.type !== "response.create") {
      originalSend(value);

      return;
    }

    if (closed) {
      originalSend(value);

      return;
    }

    const response = asRecord(event.response) ?? {};
    const model = readString(response.model) ?? options?.model;
    const metadata = asRecord(response.metadata) ?? {};

    const correlationKey = Array.from({ length: 16 }, (_, index) =>
      index === 0 ? REALTIME_CORRELATION_KEY : `${REALTIME_CORRELATION_KEY}_${index}`,
    ).find((key) => !(key in metadata));

    if (!correlationKey || Object.keys(metadata).length >= 16) {
      const span = startSpan(`chat ${model ?? "unknown"}`, {
        type: "generation",
        model,
        provider: options?.provider ?? "openai",
      });

      span.end({
        error: new Error("OpenAI Realtime response metadata has no room for telemetry correlation"),
      });
      originalSend(value);

      return;
    }

    const suppliedEventId = readString(event.event_id);

    const requestEventId =
      suppliedEventId && !pendingByEventId.has(suppliedEventId)
        ? suppliedEventId
        : `event_telemetry_${Date.now().toString(36)}_${(realtimeEventSequence += 1).toString(36)}`;

    const correlationId = globalThis.crypto.randomUUID();

    const inputCapture = captureEnabled("input")
      ? sanitizeResponsesCapture(response.instructions)
      : undefined;

    const span = startSpan(`chat ${model ?? "unknown"}`, {
      type: "generation",
      model,
      provider: options?.provider ?? "openai",
      input: inputCapture?.value,
      attributes: inputCapture?.truncated ? { "telemetry.dev.capture.truncated": true } : undefined,
    });

    const trace = {} as RealtimeTrace;

    const timer = setTimeout(() => {
      finishTrace(trace, { error: new Error("OpenAI Realtime trace expired") });
    }, options?.traceTimeoutMs ?? REALTIME_TRACE_TIMEOUT_MS);

    timer.unref?.();
    Object.assign(trace, {
      span,
      end: endOnce(span),
      sequence: (traceSequence += 1),
      startedAt: performance.now(),
      sawOutput: false,
      requestEventId,
      correlationId,
      correlationKey,
      timer,
      output: captureEnabled("output")
        ? {
            items: new Map(),
            itemBytes: new Map(),
            bytes: 2,
            truncated: false,
            stopped: false,
          }
        : undefined,
    });

    const oldest = [...pending.values(), ...active.values()].reduce<RealtimeTrace | undefined>(
      (candidate, trace) => (!candidate || trace.sequence < candidate.sequence ? trace : candidate),
      undefined,
    );

    if (pending.size + active.size >= (options?.maxInFlight ?? REALTIME_MAX_IN_FLIGHT) && oldest) {
      finishTrace(oldest, { error: new Error("OpenAI Realtime in-flight trace limit exceeded") });
    }

    pending.set(correlationId, trace);
    pendingByEventId.set(requestEventId, trace);

    try {
      originalSend({
        ...event,
        event_id: requestEventId,
        response: {
          ...response,
          metadata: {
            ...metadata,
            [correlationKey]: correlationId,
          },
        },
      });
    } catch (cause) {
      finishTrace(trace, { error: asError(cause) });
      throw cause;
    }
  };

  connection.close = (...args: unknown[]) => {
    try {
      originalClose(...args);
    } finally {
      onClose();
    }
  };

  connection.on("event", onEvent);

  if (socket?.on) socket.on("close", onClose);
  else socket?.addEventListener?.("close", onClose);

  if (socket?.on) socket.on("error", onError);
  else socket?.addEventListener?.("error", onError);
  wrappedRealtimeConnections.add(connection);

  return connection;
}

export function wrapOpenAI<T extends OpenAIClient>(
  client: T,
  options?: InstrumentOpenAIOptions,
): T {
  if (wrappedClients.has(client)) return client;
  const injectUsage = options?.injectStreamUsage === true;
  const rawCompletions: unknown = client.chat.completions;
  const completions = rawCompletions as JsonRecord & { create?: WrappedFunction };
  const rawResponses: unknown = client.responses;

  const responses = rawResponses as JsonRecord & {
    create?: WrappedFunction;
    retrieve?: WrappedFunction;
  };

  const rawEmbeddings: unknown = client.embeddings;
  const embeddings = rawEmbeddings as JsonRecord & { create?: WrappedFunction };
  patchInstanceMethod(completions, "create", "chat", chatRequest, chatResponse, injectUsage);
  patchInstanceMethod(
    responses,
    "create",
    "responses",
    responsesRequest,
    responsesResponse,
    injectUsage,
  );
  patchResponseRetrieveInstance(responses);
  patchInstanceMethod(
    embeddings,
    "create",
    "embeddings",
    embeddingsRequest,
    embeddingsResponse,
    injectUsage,
  );
  const images = asRecord(client.images);

  if (images) {
    for (const key of ["generate", "edit", "createVariation"] as const) {
      patchInstanceMethod(images, key, "images", imageRequest, imageResponse, injectUsage);
    }
  }

  const audio = client.audio;
  const speech = asRecord(audio?.speech);
  const transcriptions = asRecord(audio?.transcriptions);
  const translations = asRecord(audio?.translations);

  if (speech) {
    patchInstanceMethod(
      speech,
      "create",
      "responses",
      (body) => audioRequest(body, "speech", "speech"),
      () => ({}),
      injectUsage,
    );
  }

  if (transcriptions) {
    patchInstanceMethod(
      transcriptions,
      "create",
      "transcriptions",
      (body) => audioRequest(body, "transcription", "text"),
      transcriptionResponse,
      injectUsage,
    );
  }

  if (translations) {
    patchInstanceMethod(
      translations,
      "create",
      "responses",
      (body) => audioRequest(body, "translation", "text"),
      transcriptionResponse,
      injectUsage,
    );
  }

  const videos = asRecord(client.videos);

  if (videos) {
    patchInstanceMethod(
      videos,
      "create",
      "responses",
      videoCreateRequest,
      videoResponse,
      injectUsage,
    );
    patchLifecycleInstance(
      videos,
      "retrieve",
      (args) => generationRequest({ model: undefined }, "video", "video", { video_id: args[0] }),
      videoResponse,
    );
  }

  const batches = asRecord(client.batches);

  if (batches) {
    patchInstanceMethod(
      batches,
      "create",
      "responses",
      (body) => batchRequest(body, "create"),
      batchResponse,
      injectUsage,
    );

    for (const key of ["retrieve", "cancel"] as const) {
      patchLifecycleInstance(
        batches,
        key,
        (args) => batchRequest({ batch_id: args[0] }, key),
        batchResponse,
      );
    }
  }

  wrappedClients.add(client);

  return client;
}

export function instrumentOpenAI(options?: InstrumentOpenAIOptions): void {
  if (installed) return;
  const injectUsage = options?.injectStreamUsage === true;
  const videos = (OpenAI as { Videos?: unknown }).Videos;

  const videoClass = typeof videos === "function" ? videos : undefined;

  restorePatches = [
    patchPrototype(Completions, "create", "chat", chatRequest, chatResponse, injectUsage),
    patchPrototype(
      Responses,
      "create",
      "responses",
      responsesRequest,
      responsesResponse,
      injectUsage,
    ),
    patchResponseRetrievePrototype(),
    patchPrototype(
      Embeddings,
      "create",
      "embeddings",
      embeddingsRequest,
      embeddingsResponse,
      injectUsage,
    ),
    patchPrototype(Images, "generate", "images", imageRequest, imageResponse, injectUsage),
    patchPrototype(Images, "edit", "images", imageRequest, imageResponse, injectUsage),
    patchPrototype(Images, "createVariation", "images", imageRequest, imageResponse, injectUsage),
    patchPrototype(
      Speech,
      "create",
      "responses",
      (body) => audioRequest(body, "speech", "speech"),
      () => ({}),
      injectUsage,
    ),
    patchPrototype(
      Transcriptions,
      "create",
      "transcriptions",
      (body) => audioRequest(body, "transcription", "text"),
      transcriptionResponse,
      injectUsage,
    ),
    patchPrototype(
      Translations,
      "create",
      "responses",
      (body) => audioRequest(body, "translation", "text"),
      transcriptionResponse,
      injectUsage,
    ),
    ...(videoClass
      ? [
          patchPrototype(
            videoClass,
            "create",
            "responses",
            videoCreateRequest,
            videoResponse,
            injectUsage,
          ),
          patchLifecyclePrototype(
            videoClass,
            "retrieve",
            (args) =>
              generationRequest({ model: undefined }, "video", "video", { video_id: args[0] }),
            videoResponse,
          ),
        ]
      : []),
    patchPrototype(
      Batches,
      "create",
      "responses",
      (body) => batchRequest(body, "create"),
      batchResponse,
      injectUsage,
    ),
    patchLifecyclePrototype(
      Batches,
      "retrieve",
      (args) => batchRequest({ batch_id: args[0] }, "retrieve"),
      batchResponse,
    ),
    patchLifecyclePrototype(
      Batches,
      "cancel",
      (args) => batchRequest({ batch_id: args[0] }, "cancel"),
      batchResponse,
    ),
  ];
  installed = true;
}

export function uninstrumentOpenAI(): void {
  if (!installed) return;

  for (const restore of restorePatches.reverse()) restore();
  restorePatches = [];
  installed = false;
}
