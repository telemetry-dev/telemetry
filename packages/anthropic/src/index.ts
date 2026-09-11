import {
  startSpan,
  type SpanFields,
  type SpanHandle,
  type StartSpanOptions,
} from "@telemetry-dev/sdk";
import { Stream } from "@anthropic-ai/sdk/core/streaming";
import { Messages } from "@anthropic-ai/sdk/resources/messages";

export type { SpanFields, SpanHandle, StartSpanOptions } from "@telemetry-dev/sdk";

const WRAPPED = Symbol("telemetry.dev.anthropic.wrapped");
const WRAPPED_ORIGINAL = Symbol("telemetry.dev.anthropic.original");
const wrappedClients = new WeakSet<object>();

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
interface OpaqueValue {}
type Value = string | number | boolean | null | undefined | OpaqueValue;
interface ValueRecord {
  [key: string]: JsonValue | undefined;
  [key: symbol]: JsonValue | undefined;
}
interface CallableRecord {
  [key: string]: JsonValue | AnyFunction | undefined;
}
type AnyFunction = (...args: never[]) => OpaqueValue;
type WrappedFunction = AnyFunction & {
  [WRAPPED]?: true;
  [WRAPPED_ORIGINAL]?: AnyFunction;
};
type ProviderResolver = (resource: Value) => string;

interface AnthropicClient {
  messages: object;
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
interface ToolBlockState {
  data: ValueRecord;
  inputJson: string;
}
interface StreamState {
  blocks: Map<number, ValueRecord | ToolBlockState>;
  usage?: SpanFields["usage"];
  finishReason?: string;
}

function asRecord<T>(value: T): (T & ValueRecord) | undefined {
  if (value === null || value === undefined || value instanceof Function) return undefined;
  return Object(value) === value ? (value as T & ValueRecord) : undefined;
}
function readString<T>(value: T): string | undefined {
  return String(value) === value ? (value as string) : undefined;
}
function readNumber<T>(value: T): number | undefined {
  return Number(value) === value ? (value as number) : undefined;
}
function compactUsage(usage: SpanFields["usage"]): SpanFields["usage"] {
  if (!usage) return undefined;
  return Object.values(usage).some((value) => value !== undefined) ? usage : undefined;
}
function stopSequences<T>(value: T): string[] | undefined {
  const single = readString(value);
  if (single !== undefined) return [single];
  if (!Array.isArray(value)) return undefined;
  const strings = value.flatMap((item) => {
    const string = readString(item);
    return string === undefined ? [] : [string];
  });
  return strings.length > 0 ? strings : undefined;
}
function messagesRequest(body: ValueRecord): RequestMapping {
  const model = readString(body.model);
  const input =
    body.tools !== undefined || body.tool_choice !== undefined
      ? { messages: body.messages, tools: body.tools, tool_choice: body.tool_choice }
      : body.messages;
  return {
    name: `chat ${model ?? "unknown"}`,
    fields: {
      type: "generation",
      model,
      input,
      systemInstructions: body.system,
      temperature: readNumber(body.temperature),
      topP: readNumber(body.top_p),
      topK: readNumber(body.top_k),
      maxTokens: readNumber(body.max_tokens),
      stopSequences: stopSequences(body.stop_sequences),
    },
  };
}
function messagesUsage<T>(usage: T): SpanFields["usage"] {
  const record = asRecord(usage);
  const details = asRecord(record?.output_tokens_details);
  return compactUsage({
    inputTokens: readNumber(record?.input_tokens),
    outputTokens: readNumber(record?.output_tokens),
    cacheReadInputTokens: readNumber(record?.cache_read_input_tokens),
    cacheCreationInputTokens: readNumber(record?.cache_creation_input_tokens),
    reasoningOutputTokens: readNumber(details?.thinking_tokens),
  });
}
function mergeUsage(
  current: SpanFields["usage"],
  incoming: SpanFields["usage"],
): SpanFields["usage"] {
  if (!incoming) return current;
  return compactUsage({
    inputTokens: incoming.inputTokens ?? current?.inputTokens,
    outputTokens: incoming.outputTokens ?? current?.outputTokens,
    cacheReadInputTokens: incoming.cacheReadInputTokens ?? current?.cacheReadInputTokens,
    cacheCreationInputTokens:
      incoming.cacheCreationInputTokens ?? current?.cacheCreationInputTokens,
    reasoningOutputTokens: incoming.reasoningOutputTokens ?? current?.reasoningOutputTokens,
  });
}
function messagesResponse<T>(response: T): SpanFields {
  const record: ValueRecord = asRecord(response) ?? {};
  const role = readString(record.role) ?? "assistant";
  return {
    responseModel: readString(record.model),
    responseId: readString(record.id),
    finishReason: readString(record.stop_reason),
    output: record.content !== undefined ? [{ role, content: record.content }] : undefined,
    usage: messagesUsage(record.usage),
  };
}
function constructorName<T>(value: T): string | undefined {
  if (value === null || value === undefined) return undefined;
  const ctor = Object(value).constructor;
  return ctor instanceof Function ? ctor.name : undefined;
}
function providerForClient<T>(client: T): string {
  switch (constructorName(client)) {
    case "AnthropicBedrock":
    case "AsyncAnthropicBedrock":
      return "aws.bedrock";
    case "AnthropicVertex":
    case "AsyncAnthropicVertex":
      return "gcp.vertex_ai";
    default:
      return "anthropic";
  }
}
function providerForResource<T>(resource: T): string {
  return providerForClient(asRecord(resource)?._client);
}
function isWrapped<T>(fn: T): fn is T & WrappedFunction {
  if (!(fn instanceof Function)) return false;
  return (fn as WrappedFunction)[WRAPPED] === true;
}
function markWrapped<T extends WrappedFunction>(fn: T, original: AnyFunction): T {
  Object.defineProperty(fn, WRAPPED, { value: true });
  Object.defineProperty(fn, WRAPPED_ORIGINAL, { value: original });
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
  asResponse(): Promise<Response>;
  withResponse(): Promise<WrappedResponse<T>>;
  _thenUnwrap<U>(transform: (data: T, ...args: Value[]) => U): TracedResult<U>;
};
interface TracePromiseContext {
  end: (fields?: SpanFields) => void;
  span: SpanHandle;
  startedAt: number;
  streaming: boolean;
  mapResponse: (response: Value) => SpanFields;
}
interface StreamLike<T> extends AsyncIterable<T> {
  controller: AbortController;
}
interface InnerAPIPromise {
  then: Promise<Value>["then"];
  catch: Promise<Value>["catch"];
  finally: Promise<Value>["finally"];
  asResponse?: () => Promise<Response>;
  _thenUnwrap?: <U>(transform: (data: Value, ...args: Value[]) => U) => InnerAPIPromise;
}
function mapRawResponse(response: Response): SpanFields {
  const fields: SpanFields = { attributes: { "http.response.status_code": response.status } };
  const requestId = response.headers.get("request-id");
  if (requestId) fields.responseId = requestId;
  return fields;
}
function finalizeParsedValue(value: Value, ctx: TracePromiseContext): Value {
  if (ctx.streaming) {
    const wrapped = wrapStream(value, ctx.span, ctx.startedAt, ctx.end);
    if (wrapped !== value) return wrapped;
  }
  ctx.end(ctx.mapResponse(value));
  return value;
}
function rejectMissing(method: string): Promise<never> {
  return Promise.reject(new TypeError(`wrapped result has no ${method}`));
}
function makeTracedPromise<T>(inner: InnerAPIPromise, ctx: TracePromiseContext): TracedResult<T> {
  const source = inner;
  const originalThen = source.then.bind(source);
  const originalAsResponse = source.asResponse?.bind(source);
  const originalThenUnwrap = source._thenUnwrap?.bind(source);
  const handleError = (error: Value): never => {
    ctx.end({ error: error instanceof Error ? error : new Error(String(error)) });
    throw error;
  };
  const onParsed = (value: Value) => finalizeParsedValue(value, ctx) as T;
  const parsed = () => originalThen(onParsed, handleError);
  let traced: TracedResult<T>;
  if (constructorName(source) === "APIPromise") {
    traced = source as TracedResult<T>;
  } else {
    traced = new Promise<T>((resolve, reject) => parsed().then(resolve, reject)) as TracedResult<T>;
  }
  if (traced === source) {
    // oxlint-disable-next-line unicorn/no-thenable -- preserve Anthropic APIPromise subclass contract.
    traced.then = ((onfulfilled, onrejected) =>
      parsed().then(onfulfilled, onrejected)) as TracedResult<T>["then"];
    traced.catch = ((onrejected) => parsed().catch(onrejected)) as TracedResult<T>["catch"];
    traced.finally = ((onfinally) => parsed().finally(onfinally)) as TracedResult<T>["finally"];
  }
  traced.asResponse = () => {
    if (!originalAsResponse) return rejectMissing("asResponse");
    return originalAsResponse().then((response) => {
      ctx.end(mapRawResponse(response));
      return response;
    }, handleError);
  };
  traced.withResponse = () => {
    if (!originalAsResponse) return rejectMissing("withResponse");
    return Promise.all([parsed(), originalAsResponse()]).then(
      ([data, response]) => ({ data, response, request_id: response.headers.get("request-id") }),
      handleError,
    );
  };
  traced._thenUnwrap = <U>(transform: (data: T, ...args: Value[]) => U): TracedResult<U> => {
    if (!originalThenUnwrap) {
      return rejectMissing("_thenUnwrap") as Promise<never> & TracedResult<U>;
    }
    return makeTracedPromise(
      originalThenUnwrap((data, ...args) => {
        return transform(data as T, ...args);
      }),
      ctx,
    );
  };
  return traced;
}

function isToolBlockState(value: ValueRecord | ToolBlockState): value is ToolBlockState {
  return "data" in value && "inputJson" in value;
}
function setFirstStreamUpdate(
  span: SpanHandle,
  startedAt: number,
  sawFirst: { value: boolean },
  fields: SpanFields,
): void {
  if (sawFirst.value) return;
  sawFirst.value = true;
  span.update({
    timeToFirstChunkMs: Date.now() - startedAt,
    responseId: fields.responseId,
    responseModel: fields.responseModel,
  });
}
function parseToolInput(inputJson: string): JsonValue {
  if (inputJson.length === 0) return {};
  try {
    return JSON.parse(inputJson);
  } catch {
    return inputJson;
  }
}
function finalizeBlock(block: ValueRecord | ToolBlockState) {
  if (!isToolBlockState(block)) return block;
  const input = parseToolInput(block.inputJson);
  if (block.data.input === undefined) return { ...block.data, input };
  if (block.inputJson.length === 0) return block.data;
  const existingInput: ValueRecord | undefined = asRecord(block.data.input);
  const parsedInput: ValueRecord | undefined = asRecord(input);
  return {
    ...block.data,
    input:
      existingInput && parsedInput && !Array.isArray(existingInput) && !Array.isArray(parsedInput)
        ? { ...existingInput, ...parsedInput }
        : input,
  };
}
function streamOutput(state: StreamState): ValueRecord[] | undefined {
  if (state.blocks.size === 0) return undefined;
  const content = [...state.blocks.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, block]) => finalizeBlock(block));
  return [{ role: "assistant", content: content as JsonValue[] }];
}
function streamPartialFields(state: StreamState): SpanFields {
  return { output: streamOutput(state), usage: state.usage, finishReason: state.finishReason };
}
function recordContentBlockStart(event: ValueRecord, state: StreamState): void {
  const index = readNumber(event.index) ?? state.blocks.size;
  const contentBlock: ValueRecord = asRecord(event.content_block) ?? {};
  const type = readString(contentBlock.type);
  if (type === "text") {
    state.blocks.set(index, { type, text: readString(contentBlock.text) ?? "" });
    return;
  }
  if (type === "tool_use" || type === "server_tool_use") {
    const data = { ...contentBlock, type };
    state.blocks.set(index, { data, inputJson: "" });
    return;
  }
  if (type === "thinking") {
    state.blocks.set(index, { type, thinking: readString(contentBlock.thinking) ?? "" });
    return;
  }
  state.blocks.set(index, { ...contentBlock });
}
function blockForDelta(
  index: number,
  deltaType: string | undefined,
  state: StreamState,
): ValueRecord | ToolBlockState {
  const existing = state.blocks.get(index);
  if (existing) return existing;
  if (deltaType === "input_json_delta") {
    const block = { data: { type: "tool_use" }, inputJson: "" };
    state.blocks.set(index, block);
    return block;
  }
  const block =
    deltaType === "thinking_delta"
      ? { type: "thinking", thinking: "" }
      : { type: "text", text: "" };
  state.blocks.set(index, block);
  return block;
}
function appendStringField(target: ValueRecord, key: string, value: string | undefined): void {
  if (value === undefined) return;
  const existing = readString(target[key]) ?? "";
  target[key] = `${existing}${value}`;
}
function appendArrayField(target: ValueRecord, key: string, value: JsonValue | undefined): void {
  if (value === undefined) return;
  const existing = Array.isArray(target[key]) ? target[key] : [];
  target[key] = [...existing, value];
}
function recordContentBlockDelta(event: ValueRecord, state: StreamState): void {
  const index = readNumber(event.index) ?? 0;
  const delta: ValueRecord = asRecord(event.delta) ?? {};
  const deltaType = readString(delta.type);
  const block = blockForDelta(index, deltaType, state);
  if (deltaType === "input_json_delta") {
    if (isToolBlockState(block)) block.inputJson += readString(delta.partial_json) ?? "";
    return;
  }
  const data = isToolBlockState(block) ? block.data : block;
  if (deltaType === "text_delta") appendStringField(data, "text", readString(delta.text));
  if (deltaType === "citations_delta") appendArrayField(data, "citations", delta.citation);
  if (deltaType === "thinking_delta")
    appendStringField(data, "thinking", readString(delta.thinking));
  if (deltaType === "signature_delta" && delta.signature !== undefined)
    data.signature = delta.signature;
}
function recordStreamEvent<T>(event: T, state: StreamState): SpanFields {
  const record: ValueRecord = asRecord(event) ?? {};
  const type = readString(record.type);
  const fields: SpanFields = {};
  if (type === "message_start") {
    const message: ValueRecord = asRecord(record.message) ?? {};
    fields.responseId = readString(message.id);
    fields.responseModel = readString(message.model);
    state.usage = mergeUsage(state.usage, messagesUsage(message.usage));
  }
  if (type === "content_block_start") recordContentBlockStart(record, state);
  if (type === "content_block_delta") recordContentBlockDelta(record, state);
  if (type === "message_delta") {
    const delta = asRecord(record.delta) ?? {};
    state.finishReason = readString(delta.stop_reason) ?? state.finishReason;
    state.usage = mergeUsage(state.usage, messagesUsage(record.usage));
  }
  return fields;
}
function isStreamLike<T>(value: T): value is T & StreamLike<Value> {
  const stream = asRecord(value);
  if (!stream) return false;
  const controller = asRecord(stream.controller);
  const signal = asRecord(controller?.signal);
  if (
    !controller ||
    !(controller.abort instanceof Function) ||
    !signal ||
    typeof signal.aborted !== "boolean" ||
    !(signal.addEventListener instanceof Function)
  )
    return false;
  return stream[Symbol.asyncIterator] instanceof Function;
}
function createObservedMessagesStream(
  source: StreamLike<Value>,
  span: SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
): Stream<Value> {
  const state: StreamState = { blocks: new Map() };
  let consumed = false;
  const signal = source.controller.signal;
  const endAborted = () => {
    if (consumed) return;
    end({
      ...streamPartialFields(state),
      error: signal.reason instanceof Error ? signal.reason : new Error("Request aborted"),
    });
  };
  if (signal.aborted) endAborted();
  else signal.addEventListener("abort", endAborted, { once: true });
  async function* iterator() {
    consumed = true;
    const sawFirst = { value: false };
    let terminalError: Value;
    try {
      for await (const event of source) {
        const receivedAt = performance.now();
        const fields = recordStreamEvent(event, state);
        setFirstStreamUpdate(span, startedAt, sawFirst, fields);

        if (streamEventHasOutput(event)) span.recordOutputChunk?.(receivedAt);
        yield event;
      }
    } catch (error) {
      terminalError = error as Value;
      throw error;
    } finally {
      const fields = streamPartialFields(state);
      if (terminalError !== undefined)
        fields.error =
          terminalError instanceof Error ? terminalError : new Error(String(terminalError));
      end(fields);
    }
  }
  return new Stream(() => iterator(), source.controller);
}

function streamEventHasOutput(event: unknown): boolean {
  const record = asRecord(event);
  const type = record?.type;

  if (!record || (type !== "content_block_start" && type !== "content_block_delta")) return false;
  const value = asRecord(type === "content_block_delta" ? record.delta : record.content_block);
  const input = asRecord(value?.input);

  return (
    [value?.text, value?.thinking, value?.partial_json].some(
      (part) => typeof part === "string" && part.length > 0,
    ) ||
    (input !== undefined && Object.keys(input).length > 0)
  );
}

function wrapStream<T>(
  value: T,
  span: SpanHandle,
  startedAt: number,
  end: (fields?: SpanFields) => void,
): T | Stream<Value> {
  if (!isStreamLike(value)) return value;
  return createObservedMessagesStream(value, span, startedAt, end);
}
function wrapCreate<Fn extends AnyFunction>(
  original: Fn,
  mapRequest: (body: ValueRecord) => RequestMapping,
  mapResponse: (response: Value) => SpanFields,
  provider: ProviderResolver,
): Fn {
  if (isWrapped(original)) return original;
  const wrapped: WrappedFunction = function (this: Value, ...args: Value[]): OpaqueValue {
    const body: ValueRecord = asRecord(args[0]) ?? {};
    const streaming = body.stream === true;
    const request = mapRequest(body);
    const span = startSpan(request.name, { ...request.fields, provider: provider(this) });
    const end = endOnce(span);
    const startedAt = Date.now();
    try {
      const result = original.apply(this, args as never[]);
      return makeTracedPromise(result as InnerAPIPromise, {
        end,
        span,
        startedAt,
        streaming,
        mapResponse,
      }) as OpaqueValue;
    } catch (error) {
      end({ error: error instanceof Error ? error : new Error(String(error)) });
      throw error;
    }
  };
  return markWrapped(wrapped, original) as AnyFunction & Fn;
}
function patchInstanceMethod<T extends ValueRecord>(
  target: T,
  key: keyof T,
  request: (body: ValueRecord) => RequestMapping,
  response: (value: Value) => SpanFields,
  provider: string,
): void {
  const current = target[key];
  if (isWrapped(current) && Object.hasOwn(target, key)) return;
  const original = isWrapped(current) ? current[WRAPPED_ORIGINAL] : current;
  if (!(original instanceof Function)) return;
  const callable = original as AnyFunction;
  target[key] = wrapCreate(
    callable.bind(target),
    request,
    response,
    () => provider,
  ) as AnyFunction & T[keyof T];
}
function patchPrototype(
  klass: { prototype: object },
  key: string,
  request: (body: ValueRecord) => RequestMapping,
  response: (value: Value) => SpanFields,
): () => void {
  const prototype = klass.prototype as CallableRecord;
  const original = prototype[key];
  if (!(original instanceof Function) || isWrapped(original)) return () => {};
  prototype[key] = wrapCreate(original, request, response, providerForResource);
  return () => {
    prototype[key] = original;
  };
}

let installed = false;
let restorePatches: Array<() => void> = [];

export function wrapAnthropic<T extends AnthropicClient>(client: T): T {
  if (wrappedClients.has(client)) return client;
  const provider = providerForClient(client);
  const messages = asRecord(client.messages);
  if (messages)
    patchInstanceMethod(messages, "create", messagesRequest, messagesResponse, provider);
  wrappedClients.add(client);
  return client;
}
export function instrumentAnthropic(): void {
  if (installed) return;
  restorePatches = [patchPrototype(Messages, "create", messagesRequest, messagesResponse)];
  installed = true;
}
export function uninstrumentAnthropic(): void {
  if (!installed) return;
  for (const restore of restorePatches.reverse()) restore();
  restorePatches = [];
  installed = false;
}
