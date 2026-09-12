import {
  startSpan,
  type SpanFields,
  type SpanHandle,
  type StartSpanOptions,
} from "@telemetry-dev/sdk";
import { AzureOpenAI } from "openai";
import { Stream } from "openai/core/streaming";
import { Completions } from "openai/resources/chat/completions/completions";
import { Embeddings } from "openai/resources/embeddings";
import { Responses } from "openai/resources/responses/responses";

const WRAPPED = Symbol("telemetry.dev.openai.wrapped");
const ORIGINAL = Symbol("telemetry.dev.openai.original");
const wrappedClients = new WeakSet<object>();

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

type Operation = "chat" | "responses" | "embeddings";

interface OpenAIClient {
  chat: { completions: object };
  responses: object;
  embeddings: object;
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

  return {
    name: `chat ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      model,
      input: body.messages,
      temperature: readNumber(body.temperature),
      topP: readNumber(body.top_p),
      maxTokens: readNumber(body.max_completion_tokens) ?? readNumber(body.max_tokens),
      stopSequences: stopSequences(body.stop),
      seed: readNumber(body.seed),
      frequencyPenalty: readNumber(body.frequency_penalty),
      presencePenalty: readNumber(body.presence_penalty),
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
    reasoningOutputTokens: readNumber(completionDetails?.reasoning_tokens),
  });
}

function chatResponse<T>(response: T): SpanFields {
  const r = asRecord(response) ?? {};
  const choices = asArray(r.choices) ?? [];

  const finishReasons = choices
    .map((choice) => readString(asRecord(choice)?.finish_reason))
    .filter((reason): reason is string => reason !== undefined);

  const fields: SpanFields = {
    responseModel: readString(r.model),
    responseId: readString(r.id),
    finishReason: finishReasons[0],
    output: choices
      .map((choice) => asRecord(choice)?.message)
      .filter((message) => message !== undefined),
    usage: chatUsage(r.usage),
  };

  if (finishReasons.length > 1) {
    fields.attributes = { "gen_ai.response.finish_reasons": finishReasons };
  }

  return fields;
}

function responsesRequest(body: JsonRecord): RequestMapping {
  const model = readString(body.model);

  return {
    name: `chat ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      model,
      input: body.input,
      systemInstructions: body.instructions,
      temperature: readNumber(body.temperature),
      topP: readNumber(body.top_p),
      maxTokens: readNumber(body.max_output_tokens),
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
    reasoningOutputTokens: readNumber(outputDetails?.reasoning_tokens),
  });
}

function responsesResponse<T>(response: T): SpanFields {
  const r = asRecord(response) ?? {};
  const status = readString(r.status);
  const incompleteDetails = asRecord(r.incomplete_details);

  const fields: SpanFields = {
    responseModel: readString(r.model),
    responseId: readString(r.id),
    output: r.output,
    usage: responsesUsage(r.usage),
    finishReason:
      status === "completed" ? "stop" : (readString(incompleteDetails?.reason) ?? status),
  };

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
  if (client instanceof AzureOpenAI) return "azure.ai.openai";
  const host = baseURLHost(asRecord(client)?.baseURL)?.replace(/\.$/, "");

  if (host === "openrouter.ai" || host?.endsWith(".openrouter.ai")) return "openrouter";

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
    return wrapStream(value, ctx.operation, ctx.span, ctx.startedAt, ctx.injectedUsage);
  }

  ctx.end(ctx.mapResponse(value));

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
  content: string;
  refusal: string;
  toolCalls: Map<number, JsonRecord>;
  finishReason?: string;
}

function getChoiceState(states: Map<number, ChatChoiceState>, index: number): ChatChoiceState {
  let state = states.get(index);

  if (!state) {
    state = { content: "", refusal: "", toolCalls: new Map() };
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
      currentFunction.arguments = `${readString(currentFunction.arguments) ?? ""}${
        readString(incomingFunction.arguments) ?? ""
      }`;
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
      const fields = Object.assign(message, {} as JsonRecord);

      if (state.content.length > 0) fields.content = state.content;
      else if (state.toolCalls.size > 0) fields.content = null;

      if (state.refusal.length > 0) fields.refusal = state.refusal;

      if (state.toolCalls.size > 0) {
        fields.tool_calls = [...state.toolCalls.entries()]
          .sort(([left], [right]) => left - right)
          .map(([, toolCall]) => toolCall);
      }

      return fields;
    });
}

function chatPartialFields(
  states: Map<number, ChatChoiceState>,
  usage: SpanFields["usage"],
): SpanFields {
  const finishReasons = [...states.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, state]) => state.finishReason)
    .filter((reason): reason is string => reason !== undefined);

  const fields: SpanFields = {
    output: states.size > 0 ? chatOutput(states) : undefined,
    usage,
    finishReason: finishReasons[0],
  };

  if (finishReasons.length > 1) {
    fields.attributes = { "gen_ai.response.finish_reasons": finishReasons };
  }

  return fields;
}

function recordChatChunk<T>(chunk: T, states: Map<number, ChatChoiceState>) {
  const c = asRecord(chunk) ?? {};
  let hasOutput = false;

  for (const choice of asArray(c.choices) ?? []) {
    const choiceRecord = asRecord(choice) ?? {};
    const state = getChoiceState(states, readNumber(choiceRecord.index) ?? 0);
    const delta = asRecord(choiceRecord.delta) ?? {};
    const audio = asRecord(delta.audio);
    const legacyFunction = asRecord(delta.function_call);

    if (
      (typeof delta.content === "string" && delta.content.length > 0) ||
      (typeof delta.refusal === "string" && delta.refusal.length > 0) ||
      (typeof audio?.data === "string" && audio.data.length > 0) ||
      (typeof legacyFunction?.arguments === "string" && legacyFunction.arguments.length > 0) ||
      (asArray(delta.tool_calls) ?? []).some((toolCall) => {
        const fn = asRecord(asRecord(toolCall)?.function);

        return typeof fn?.arguments === "string" && fn.arguments.length > 0;
      })
    ) {
      hasOutput = true;
    }

    if (typeof delta.role === "string") state.role = delta.role;

    if (typeof delta.content === "string") state.content += delta.content;

    if (typeof delta.refusal === "string") state.refusal += delta.refusal;

    for (const toolCall of asArray(delta.tool_calls) ?? []) {
      const toolCallRecord = asRecord(toolCall);

      if (toolCallRecord) mergeToolCall(state, toolCallRecord);
    }

    const finishReason = readString(choiceRecord.finish_reason);

    if (finishReason) state.finishReason = finishReason;
  }

  return {
    responseId: readString(c.id),
    responseModel: readString(c.model),
    usage: chatUsage(c.usage),
    hasOutput,
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
  async function* iterator() {
    const states = new Map<number, ChatChoiceState>();
    let usage: SpanFields["usage"];
    let sawFirst = false;
    let terminalError: Error | undefined;

    try {
      for await (const chunk of source) {
        const receivedAt = performance.now();
        const update = recordChatChunk(chunk, states);

        if (update.hasOutput) span.recordOutputChunk?.(receivedAt);

        if (!sawFirst) {
          sawFirst = true;
          span.update({
            timeToFirstChunkMs: Date.now() - startedAt,
            responseId: update.responseId,
            responseModel: update.responseModel,
          });
        }

        if (update.usage) usage = update.usage;

        if (!hideSyntheticUsage || !isSyntheticUsageChunk(chunk)) yield chunk;
      }
    } catch (error) {
      terminalError = asError(error);
      throw error;
    } finally {
      const fields = chatPartialFields(states, usage);

      if (terminalError) fields.error = terminalError;
      end(fields);
    }
  }

  return new Stream(() => iterator(), source.controller);
}

function createObservedResponsesStream(
  source: Stream<JsonValue>,
  span: SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
): Stream<JsonValue> {
  async function* iterator() {
    let sawFirst = false;
    let partial: SpanFields = {};
    let terminalError: Error | undefined;

    try {
      for await (const event of source) {
        const receivedAt = performance.now();
        const e = asRecord(event) ?? {};

        if (responseEventHasOutput(e)) span.recordOutputChunk?.(receivedAt);
        const response = asRecord(e.response);

        if (!sawFirst) {
          sawFirst = true;
          span.update({ timeToFirstChunkMs: Date.now() - startedAt });
        }

        if (response) partial = responsesResponse(response);

        if (e.type === "response.failed") {
          partial = { ...partial, error: responseFailedError(response) };
          end(partial);
        } else if (e.type === "error") {
          partial = { ...partial, error: responseStreamError(event) };
          end(partial);
        } else if (e.type === "response.completed" || e.type === "response.incomplete") {
          end(partial);
        }

        yield event;
      }
    } catch (error) {
      terminalError = asError(error);
      throw error;
    } finally {
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
      "response.output_audio.delta",
      "response.audio.delta",
      "response.audio.transcript.delta",
    ].includes(event.type) &&
    typeof event.delta === "string" &&
    event.delta.length > 0
  );
}

function wrapStream<T>(
  value: T,
  operation: Operation,
  span: SpanHandle,
  startedAt: number,
  injectedUsage: boolean,
): T | Stream<JsonValue> {
  if (!(value instanceof Stream)) return value;
  const end = endOnce(span);
  const source = value as Stream<JsonValue>;

  if (operation === "chat") {
    return createObservedChatStream(source, span, startedAt, end, injectedUsage);
  }

  return createObservedResponsesStream(source, span, startedAt, end);
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
    const startedAt = Date.now();

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
    const startedAt = Date.now();

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

function patchResponseRetrievePrototype(): () => void {
  const rawPrototype: unknown = Responses.prototype;
  const prototype = rawPrototype as JsonRecord & { retrieve?: WrappedFunction };
  const original = prototype.retrieve;

  if (!(original instanceof Function)) return () => {};

  prototype.retrieve = wrapResponseRetrieve(original, providerForResource);

  return () => {
    prototype.retrieve = original;
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

  prototype[key] = wrapCreate(
    original,
    operation,
    request,
    response,
    providerForResource,
    injectUsage,
  );

  return () => {
    prototype[key] = original;
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
  wrappedClients.add(client);

  return client;
}

export function instrumentOpenAI(options?: InstrumentOpenAIOptions): void {
  if (installed) return;
  const injectUsage = options?.injectStreamUsage === true;
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
  ];
  installed = true;
}

export function uninstrumentOpenAI(): void {
  if (!installed) return;

  for (const restore of restorePatches.reverse()) restore();
  restorePatches = [];
  installed = false;
}
