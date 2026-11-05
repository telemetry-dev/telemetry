import {
  boundedCaptureDetails,
  startSpan,
  type SpanFields,
  type SpanHandle,
  type StartSpanOptions,
  StreamCapture,
} from "@telemetry-dev/sdk";
import * as telemetrySdk from "@telemetry-dev/sdk";
import { Stream } from "@anthropic-ai/sdk/core/streaming";
import { Messages as BetaMessages } from "@anthropic-ai/sdk/resources/beta/messages";
import { Messages } from "@anthropic-ai/sdk/resources/messages";

export type { SpanFields, SpanHandle, StartSpanOptions } from "@telemetry-dev/sdk";

const WRAPPED = Symbol("telemetry.dev.anthropic.wrapped");
const WRAPPED_ORIGINAL = Symbol("telemetry.dev.anthropic.original");
const wrappedClients = new WeakSet<object>();

type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

interface OpaqueValue {}

type Value = string | number | boolean | null | undefined | OpaqueValue;

function unknownErrorMessage(value: Value): string {
  // oxlint-disable-next-line typescript/no-base-to-string -- error payloads must not be JSON-emitted.
  return String(value);
}

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
  beta?: { messages?: object };
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
  captureOutput: boolean;
  maskOutputWhenIncomplete: boolean;
  reservations: Map<number, number>;
  capture: StreamCapture;
  captureTruncated: boolean;
  metadataTruncated: boolean;
  expectMessageStop: boolean;
  messageStopped: boolean;
  unresolvedReplacements: Set<number>;
  unresolvedEncryptedContent: Set<number>;
  usage?: SpanFields["usage"];
  oneHourCacheCreationTokens?: number;
  iterations?: IterationsUsage;
  finishReason?: string;
  responseModel?: string;
}

const STREAM_CAPTURE_MAX_ITEMS = 1_000;
const MAX_UNRESOLVED_REPLACEMENTS = STREAM_CAPTURE_MAX_ITEMS;

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

/**
 * The type and name of each request tool as `gen_ai.tool.definitions`. Names are recorded
 * regardless of capture flags, as on tool spans; descriptions and schemas stay in the input.
 */
function toolDefinitions<T>(tools: T): Record<string, string> | undefined {
  if (!Array.isArray(tools)) return undefined;

  const definitions = tools.flatMap((tool) => {
    const record = asRecord(tool);
    const name = readString(record?.name);

    if (name === undefined) return [];
    const type = readString(record?.type);

    return [{ type: type === undefined || type === "custom" ? "function" : type, name }];
  });

  return definitions.length > 0
    ? { "gen_ai.tool.definitions": JSON.stringify(definitions) }
    : undefined;
}

// Cores older than promptAttributesFor record a prompt only inside withPrompt. A namespace read
// keeps them loadable without the export.
function promptAttributes(messages: unknown): SpanFields["attributes"] {
  const read = (
    telemetrySdk as { promptAttributesFor?: (messages: unknown) => SpanFields["attributes"] }
  ).promptAttributesFor;

  return typeof read === "function" ? read(messages) : undefined;
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
      attributes: {
        ...toolDefinitions(body.tools),
        ...promptAttributes(body.messages),
      },
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

// Anthropic's input_tokens leaves out cache reads and writes; span input tokens include them.
function cacheInclusiveUsage(usage: SpanFields["usage"]): SpanFields["usage"] {
  if (usage?.inputTokens === undefined) return usage;

  return {
    ...usage,
    inputTokens:
      usage.inputTokens + (usage.cacheReadInputTokens ?? 0) + (usage.cacheCreationInputTokens ?? 0),
  };
}

function addTokens(current: number | undefined, incoming: number | undefined): number | undefined {
  return current === undefined || incoming === undefined
    ? (current ?? incoming)
    : current + incoming;
}

function addUsage(
  current: SpanFields["usage"],
  incoming: SpanFields["usage"],
): SpanFields["usage"] {
  if (!incoming) return current;

  return compactUsage({
    inputTokens: addTokens(current?.inputTokens, incoming.inputTokens),
    outputTokens: addTokens(current?.outputTokens, incoming.outputTokens),
    cacheReadInputTokens: addTokens(current?.cacheReadInputTokens, incoming.cacheReadInputTokens),
    cacheCreationInputTokens: addTokens(
      current?.cacheCreationInputTokens,
      incoming.cacheCreationInputTokens,
    ),
    reasoningOutputTokens: addTokens(
      current?.reasoningOutputTokens,
      incoming.reasoningOutputTokens,
    ),
  });
}

function oneHourCacheCreationTokens<T>(usage: T): number | undefined {
  return readNumber(asRecord(asRecord(usage)?.cache_creation)?.ephemeral_1h_input_tokens);
}

interface IterationsUsage {
  compaction: SpanFields["usage"];
  advisor?: SpanFields["usage"];
  advisorModel?: string;
  compactionOneHourCacheCreationTokens?: number;
  executorOneHourCacheCreationTokens?: number;
  truncated: boolean;
}

// The top-level usage fields sum the executor iterations (message and fallback_message) and leave
// out compaction, so compaction is added from here. Like the fallback lookup, only the bounded
// tail of usage.iterations is read.
function iterationsUsage<T>(usage: T): IterationsUsage | undefined {
  const iterations = asRecord(usage)?.iterations;

  if (!Array.isArray(iterations)) return undefined;

  const result: IterationsUsage = {
    compaction: undefined,
    truncated: iterations.length > STREAM_CAPTURE_MAX_ITEMS,
  };

  const firstIndex = Math.max(0, iterations.length - STREAM_CAPTURE_MAX_ITEMS);
  // A stream uses the executor sum in place of message_start, so the sum is kept only when every
  // executor iteration reports the one-hour split. Iterations past the bound go unseen.
  let executorSplitComplete = !result.truncated;
  let advisorModelComplete = true;

  for (let index = firstIndex; index < iterations.length; index += 1) {
    const entry = asRecord(iterations[index]);

    if (entry?.type === "compaction") {
      result.compaction = addUsage(result.compaction, messagesUsage(entry));

      result.compactionOneHourCacheCreationTokens = addTokens(
        result.compactionOneHourCacheCreationTokens,
        oneHourCacheCreationTokens(entry),
      );
    } else if (entry?.type === "advisor_message") {
      result.advisor = addUsage(result.advisor, messagesUsage(entry));
      const model = readString(entry.model);
      if (
        model === undefined ||
        model.trim() === "" ||
        (result.advisorModel !== undefined && model !== result.advisorModel)
      )
        advisorModelComplete = false;
      result.advisorModel = model ?? result.advisorModel;
    } else if (entry?.type === "message" || entry?.type === "fallback_message") {
      const oneHour = oneHourCacheCreationTokens(entry);

      if (oneHour === undefined) executorSplitComplete = false;

      result.executorOneHourCacheCreationTokens = addTokens(
        result.executorOneHourCacheCreationTokens,
        oneHour,
      );
    }
  }

  if (!executorSplitComplete) result.executorOneHourCacheCreationTokens = undefined;
  // An aggregate must not be priced at one model's rates when its identity is ambiguous.
  if (!advisorModelComplete) {
    result.advisorModel = undefined;
    result.truncated = true;
  }

  return result;
}

function usageAttributes(
  oneHourCacheCreation: number | undefined,
  captureIncomplete: boolean,
  iterations: IterationsUsage | undefined,
): SpanFields["attributes"] {
  const attributes: NonNullable<SpanFields["attributes"]> = {};

  if (iterations?.advisorModel !== undefined)
    attributes["anthropic.advisor.model"] = iterations.advisorModel;

  const advisor = iterations?.advisor;
  if (advisor?.inputTokens !== undefined)
    attributes["anthropic.advisor.usage.input_tokens"] = advisor.inputTokens;
  if (advisor?.outputTokens !== undefined)
    attributes["anthropic.advisor.usage.output_tokens"] = advisor.outputTokens;
  if (advisor?.cacheReadInputTokens !== undefined)
    attributes["anthropic.advisor.usage.cache_read_input_tokens"] = advisor.cacheReadInputTokens;
  if (advisor?.cacheCreationInputTokens !== undefined)
    attributes["anthropic.advisor.usage.cache_creation_input_tokens"] =
      advisor.cacheCreationInputTokens;

  if (oneHourCacheCreation !== undefined)
    attributes["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"] = oneHourCacheCreation;

  if (captureIncomplete) attributes["telemetry.dev.capture.truncated"] = true;

  return Object.keys(attributes).length > 0 ? attributes : undefined;
}

function messagesResponse<T>(response: T): SpanFields {
  const record: ValueRecord = asRecord(response) ?? {};
  const role = readString(record.role) ?? "assistant";
  const iterations = iterationsUsage(record.usage);

  const fields: SpanFields = {
    responseModel: readString(record.model),
    responseId: readString(record.id),
    finishReason: readString(record.stop_reason),
    output: record.content !== undefined ? [{ role, content: record.content }] : undefined,
    usage: cacheInclusiveUsage(addUsage(messagesUsage(record.usage), iterations?.compaction)),
  };

  const attributes = usageAttributes(
    addTokens(
      oneHourCacheCreationTokens(record.usage),
      iterations?.compactionOneHourCacheCreationTokens,
    ),
    iterations?.truncated === true,
    iterations,
  );

  if (attributes) fields.attributes = attributes;

  return fields;
}

function constructorName<T>(value: T): string | undefined {
  if (value === null || value === undefined) return undefined;
  const ctor = Object(value).constructor;

  return ctor instanceof Function ? ctor.name : undefined;
}

const clientProviders = new Map([
  ["AnthropicBedrock", "aws.bedrock"],
  ["AnthropicBedrockMantle", "aws.bedrock"],
  ["AnthropicVertex", "gcp.vertex_ai"],
]);

// Walks the prototype chain so subclasses of a provider client keep its provider.
function providerForClient<T>(client: T): string {
  let prototype =
    client === null || client === undefined ? null : Object.getPrototypeOf(Object(client));

  while (prototype !== null && prototype !== Object.prototype) {
    const provider = clientProviders.get(constructorName(prototype) ?? "");

    if (provider) return provider;
    prototype = Object.getPrototypeOf(prototype);
  }

  return "anthropic";
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

/** The fields of the response props of `@anthropic-ai/sdk` that the wrapper reads. */
interface ResponseProps {
  response: Response;
  controller: AbortController;
}

/**
 * The bytes of a streamed body that the span's copy may hold before the span reads them. The span
 * reads as fast as the caller does unless its exporter stalls; past this bound the span stops
 * reading and its output is truncated, and the caller's body goes on.
 */
const COPY_QUEUE_BYTES = 16 * 1024 * 1024;

type TracedResult<T> = Promise<T> & {
  asResponse(): Promise<Response>;
  withResponse(): Promise<WrappedResponse<T>>;
  _thenUnwrap<U>(transform: (data: T, ...args: Value[]) => U): TracedResult<U>;
  _responseProps?: () => Promise<ResponseProps>;
};

interface TracePromiseContext {
  end: (fields?: SpanFields) => void;
  span: SpanHandle;
  startedAt: number;
  streaming: boolean;
  mapResponse: (response: Value) => SpanFields;
  /**
   * The client of the request, or undefined. `Stream.fromSSEResponse` reads only its logger, and
   * logs to the console without one.
   */
  client: Value;
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
  _responseProps?: () => Promise<ResponseProps>;
}

function mapRawResponse(response: Response): SpanFields {
  const fields: SpanFields = { attributes: { "http.response.status_code": response.status } };
  const requestId = response.headers.get("request-id");

  if (requestId) fields.responseId = requestId;

  return fields;
}

function finalizeParsedValue(value: Value, ctx: TracePromiseContext): Value {
  if (ctx.streaming) {
    try {
      const wrapped = wrapStream(value, ctx.span, ctx.startedAt, ctx.end);

      if (wrapped !== value) return wrapped;
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

/**
 * Gives the body of a streamed response to the span as well. A message thread of
 * `@anthropic-ai/sdk` 1.2 and later parses the stream of each turn from the response props of `create()`,
 * and never awaits the promise that `create()` returns, so the span never saw the stream.
 *
 * The returned response copies each chunk to a second body as the caller reads it, and the span
 * reads that second body, so it ends when the caller's stream ends, fails, or is aborted. The span
 * reads with its own controller: a failure of the span must not abort the caller's request.
 */
function observeStreamedResponse(props: ResponseProps, ctx: TracePromiseContext): ResponseProps {
  const body = props.response.body;

  if (!body) {
    ctx.end(mapRawResponse(props.response));

    return props;
  }

  // The copy is best effort: once the span stops reading it, or a write to it fails, the caller's
  // body goes on without it.
  let copyOpen = true;
  let copy: ReadableStreamDefaultController<Uint8Array> | undefined;
  const toCopy = (write: (controller: ReadableStreamDefaultController<Uint8Array>) => void) => {
    if (!copyOpen || !copy) return;

    try {
      write(copy);
    } catch {
      copyOpen = false;
    }
  };
  const closeCopy = (close: (controller: ReadableStreamDefaultController<Uint8Array>) => void) => {
    toCopy(close);
    copyOpen = false;
  };
  const observed = new ReadableStream<Uint8Array>(
    {
      start: (controller) => {
        copy = controller;
      },
      cancel: () => {
        copyOpen = false;
      },
    },
    new ByteLengthQueuingStrategy({ highWaterMark: COPY_QUEUE_BYTES }),
  );
  const reader = body.getReader();
  const signal = props.controller.signal;
  const onAbort = () =>
    closeCopy((controller) =>
      controller.error(
        signal.reason instanceof Error ? signal.reason : new Error("Request aborted"),
      ),
    );
  signal.addEventListener("abort", onAbort, { once: true });

  const read = new ReadableStream<Uint8Array>({
    async pull(controller) {
      let chunk: ReadableStreamReadResult<Uint8Array>;

      try {
        chunk = await reader.read();
      } catch (error) {
        closeCopy((copyController) => copyController.error(error));
        controller.error(error);

        return;
      }

      if (chunk.done) {
        signal.removeEventListener("abort", onAbort);
        closeCopy((copyController) => copyController.close());
        controller.close();

        return;
      }

      toCopy((copyController) => {
        copyController.enqueue(chunk.value);

        if ((copyController.desiredSize ?? 0) <= 0) closeCopy((full) => full.close());
      });
      controller.enqueue(chunk.value);
    },
    // A caller that stops reading, such as for a listener that throws, cancels the body before
    // the SDK aborts the request. The span then ends with what it read, as for an early stop.
    cancel(reason) {
      signal.removeEventListener("abort", onAbort);
      closeCopy((copyController) => copyController.close());

      return reader.cancel(reason);
    },
  });

  const stream = finalizeParsedValue(
    Stream.fromSSEResponse(
      new Response(observed, props.response),
      new AbortController(),
      ctx.client as never,
    ),
    ctx,
  );

  if (isStreamLike(stream)) {
    const iterator = stream[Symbol.asyncIterator]();
    const drain = async () => {
      while (!(await iterator.next()).done);
    };
    // The span records the error of the stream; the caller gets it from its own body.
    drain().catch(() => undefined);
  }

  const response = new Response(read, props.response);
  // The Response constructor takes no url or redirected; the caller and the SDK's logs read them.
  Object.defineProperties(response, {
    url: { value: props.response.url },
    redirected: { value: props.response.redirected },
  });

  return { ...props, response };
}

function rejectMissing(method: string): Promise<never> {
  return Promise.reject(new TypeError(`wrapped result has no ${method}`));
}

function makeTracedPromise<T>(inner: InnerAPIPromise, ctx: TracePromiseContext): TracedResult<T> {
  const source = inner;
  const originalThen = source.then.bind(source);
  const originalAsResponse = source.asResponse?.bind(source);
  const originalThenUnwrap = source._thenUnwrap?.bind(source);
  const originalResponseProps = source._responseProps?.bind(source);

  const handleError = (error: Value): never => {
    ctx.end({
      error: error instanceof Error ? error : new Error(unknownErrorMessage(error)),
    });
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

  if (originalResponseProps) {
    // The body of a stream can be read only once, so each call gets the same props.
    let props: Promise<ResponseProps> | undefined;

    traced._responseProps = () =>
      (props ??= originalResponseProps().then((value) => {
        if (!ctx.streaming) return value;

        try {
          return observeStreamedResponse(value, ctx);
        } catch (cause) {
          reportSpanError(ctx.span, cause);
          ctx.end({ attributes: { "telemetry.dev.capture.truncated": true } });

          return value;
        }
      }, handleError));
  }

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
  const retainedOutput = state.captureOutput ? streamOutput(state) : undefined;

  const boundedOutput =
    retainedOutput === undefined
      ? undefined
      : boundedCaptureDetails(retainedOutput, { maxBytes: state.capture.limit });

  const outputIncomplete =
    state.captureTruncated ||
    boundedOutput?.truncated === true ||
    (state.expectMessageStop && !state.messageStopped) ||
    state.unresolvedReplacements.size > 0 ||
    state.unresolvedEncryptedContent.size > 0;

  const captureIncomplete = outputIncomplete || state.metadataTruncated;

  const fields: SpanFields = {
    output: state.maskOutputWhenIncomplete && outputIncomplete ? undefined : boundedOutput?.value,
    usage: cacheInclusiveUsage(addUsage(state.usage, state.iterations?.compaction)),
    finishReason: state.finishReason,
  };

  if (state.responseModel !== undefined) fields.responseModel = state.responseModel;

  // Only message_start carries the one-hour split, so later turns come from the executor
  // iterations in the final usage when every one reports it.
  const attributes = usageAttributes(
    addTokens(
      state.iterations?.executorOneHourCacheCreationTokens ?? state.oneHourCacheCreationTokens,
      state.iterations?.compactionOneHourCacheCreationTokens,
    ),
    captureIncomplete || state.iterations?.truncated === true,
    state.iterations,
  );

  if (attributes) fields.attributes = attributes;

  return fields;
}

const APPENDED_DELTA_FIELDS = new Map([
  ["text_delta", "text"],
  ["thinking_delta", "thinking"],
  ["input_json_delta", "partial_json"],
]);

function reserveAppendedText(
  index: number,
  field: string,
  delta: ValueRecord,
  state: StreamState,
): ValueRecord | undefined {
  const text = readString(delta[field]) ?? "";

  const capture = boundedCaptureDetails(text, {
    maxBytes: Math.max(state.capture.limit - state.capture.bytes + 2, 0),
  });

  if (capture.truncated) {
    state.captureTruncated = true;

    return undefined;
  }

  const bytes = capture.bytes - 2;
  state.reservations.set(index, (state.reservations.get(index) ?? 0) + bytes);
  state.capture.bytes += bytes;

  return { [field]: text };
}

// The terminal output wraps blocks in [{"role":"assistant","content":[...]}] with a comma
// between blocks; charging both while streaming keeps an accepted state within the limit.
const STREAM_OUTPUT_ENVELOPE = boundedCaptureDetails([{ role: "assistant", content: [] }]);
const BLOCK_SEPARATOR_BYTES = 1;

// The bytes a value may take and still fit in the stream, so that a value past the room is
// measured only that far.
function streamRoom(state: StreamState, freed: number, separator: number): number {
  return Math.max(state.capture.limit - state.capture.bytes + freed - separator, 0);
}

function reserveStreamValue(index: number, value: unknown, state: StreamState): unknown {
  const separator = state.reservations.has(index) ? 0 : BLOCK_SEPARATOR_BYTES;
  const capture = boundedCaptureDetails(value, { maxBytes: streamRoom(state, 0, separator) });
  const bytes = capture.bytes + separator;

  if (capture.truncated || state.capture.bytes + bytes > state.capture.limit) {
    state.captureTruncated = true;

    return undefined;
  }

  state.reservations.set(index, (state.reservations.get(index) ?? 0) + bytes);
  state.capture.bytes += bytes;

  return capture.value;
}

function reserveReplacement(
  index: number,
  value: ValueRecord | ToolBlockState,
  state: StreamState,
  recoverUnseen: boolean,
): boolean {
  if (
    state.captureTruncated &&
    (!recoverUnseen || (!state.blocks.has(index) && !state.unresolvedReplacements.has(index)))
  )
    return false;

  const held = state.reservations.get(index) ?? 0;
  const capture = boundedCaptureDetails(value, {
    maxBytes: streamRoom(state, held, BLOCK_SEPARATOR_BYTES),
  });
  const bytes = capture.bytes + BLOCK_SEPARATOR_BYTES;

  if (capture.truncated || state.capture.bytes - held + bytes > state.capture.limit) {
    if (recoverUnseen || state.blocks.has(index) || state.unresolvedReplacements.has(index)) {
      if (
        !state.unresolvedReplacements.has(index) &&
        state.unresolvedReplacements.size >= MAX_UNRESOLVED_REPLACEMENTS
      )
        state.captureTruncated = true;
      else state.unresolvedReplacements.add(index);
    } else state.captureTruncated = true;

    return false;
  }

  state.capture.bytes += bytes - held;
  state.reservations.set(index, bytes);
  state.unresolvedReplacements.delete(index);

  return true;
}

function stripStaleSignature(index: number, state: StreamState): void {
  const block = state.blocks.get(index);

  if (!block) return;
  const data = isToolBlockState(block) ? { ...block.data } : { ...block };

  if (!("signature" in data)) return;
  delete data.signature;
  const stripped = isToolBlockState(block) ? { ...block, data } : data;
  const capture = boundedCaptureDetails(stripped);
  const bytes = capture.bytes + BLOCK_SEPARATOR_BYTES;
  const held = state.reservations.get(index) ?? 0;

  if (!capture.truncated && state.capture.bytes - held + bytes <= state.capture.limit) {
    state.capture.bytes += bytes - held;
    state.reservations.set(index, bytes);
  }

  state.blocks.set(index, stripped);
  state.unresolvedReplacements.add(index);
}

function dropBlock(index: number, state: StreamState): void {
  state.capture.bytes -= state.reservations.get(index) ?? 0;
  state.reservations.delete(index);
  state.blocks.delete(index);
}

function recordContentBlockStart(event: ValueRecord, state: StreamState): void {
  if (!state.captureOutput || state.captureTruncated) return;
  const index = readNumber(event.index) ?? state.blocks.size;
  const rawContentBlock = asRecord(event.content_block);

  if (
    rawContentBlock &&
    !Array.isArray(rawContentBlock) &&
    readString(rawContentBlock.type) === "compaction"
  ) {
    const contentBlock = rawContentBlock as ValueRecord;

    if (reserveReplacement(index, contentBlock, state, true)) {
      state.blocks.set(index, contentBlock);

      if ("encrypted_content" in contentBlock) state.unresolvedEncryptedContent.delete(index);
    } else {
      if ("encrypted_content" in contentBlock) {
        if (
          !state.unresolvedEncryptedContent.has(index) &&
          state.unresolvedEncryptedContent.size >= MAX_UNRESOLVED_REPLACEMENTS
        )
          state.captureTruncated = true;
        else state.unresolvedEncryptedContent.add(index);
      }

      dropBlock(index, state);
    }

    return;
  }

  const contentBlock = asRecord(reserveStreamValue(index, event.content_block, state));

  if (!contentBlock) return;
  const type = readString(contentBlock.type);

  if (type === "text") {
    state.blocks.set(index, { type, text: readString(contentBlock.text) ?? "" });

    return;
  }

  if (type === "tool_use" || type === "server_tool_use" || type === "mcp_tool_use") {
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

  const block = defaultBlock(deltaType);
  state.blocks.set(index, block);

  return block;
}

function defaultBlock(deltaType: string | undefined): ValueRecord | ToolBlockState {
  if (deltaType === "input_json_delta") {
    const block = { data: { type: "tool_use" }, inputJson: "" };

    return block;
  }

  if (deltaType === "thinking_delta") return { type: "thinking", thinking: "" };

  if (deltaType === "compaction_delta") return { type: "compaction" };

  return { type: "text", text: "" };
}

function appendStringField(target: ValueRecord, key: string, value: string | undefined): void {
  if (value === undefined) return;
  const existing = readString(target[key]) ?? "";
  target[key] = `${existing}${value}`;
}

// The arrays a stream built itself. A delta appends to one in place, so a long run of deltas
// costs each delta only its own value; an array from a provider event is copied once first.
const appendedArrays = new WeakSet<JsonValue[]>();

function appendArrayField(target: ValueRecord, key: string, value: JsonValue | undefined): void {
  if (value === undefined) return;
  const existing = target[key];

  if (Array.isArray(existing) && appendedArrays.has(existing)) {
    existing.push(value);

    return;
  }

  const appended = Array.isArray(existing) ? [...existing, value] : [value];
  appendedArrays.add(appended);
  target[key] = appended;
}

function recordContentBlockDelta(event: ValueRecord, state: StreamState): void {
  if (!state.captureOutput) return;
  const index = readNumber(event.index) ?? 0;
  const delta: ValueRecord = asRecord(event.delta) ?? {};
  const deltaType = readString(delta.type);

  if (deltaType === "compaction_delta" || deltaType === "signature_delta") {
    const block = state.blocks.get(index) ?? defaultBlock(deltaType);
    const data = isToolBlockState(block) ? { ...block.data } : { ...block };

    if (deltaType === "compaction_delta") {
      data.content = delta.content;

      if ("encrypted_content" in delta) data.encrypted_content = delta.encrypted_content;
    } else if (delta.signature !== undefined) {
      data.signature = delta.signature;
    } else {
      return;
    }

    const replacement = isToolBlockState(block) ? { ...block, data } : data;

    if (reserveReplacement(index, replacement, state, deltaType === "compaction_delta")) {
      state.blocks.set(index, replacement);

      if (deltaType === "compaction_delta" && "encrypted_content" in delta)
        state.unresolvedEncryptedContent.delete(index);
    } else if (deltaType === "compaction_delta") {
      if ("encrypted_content" in data) {
        if (
          !state.unresolvedEncryptedContent.has(index) &&
          state.unresolvedEncryptedContent.size >= MAX_UNRESOLVED_REPLACEMENTS
        )
          state.captureTruncated = true;
        else state.unresolvedEncryptedContent.add(index);
      }

      dropBlock(index, state);
    } else stripStaleSignature(index, state);

    return;
  }

  if (state.captureTruncated) return;

  const appendedField = deltaType === undefined ? undefined : APPENDED_DELTA_FIELDS.get(deltaType);

  const capturedDelta =
    appendedField !== undefined && state.blocks.has(index)
      ? reserveAppendedText(index, appendedField, delta, state)
      : asRecord(reserveStreamValue(index, delta, state));

  if (!capturedDelta) return;
  const block = blockForDelta(index, deltaType, state);

  if (deltaType === "input_json_delta") {
    if (isToolBlockState(block)) block.inputJson += readString(capturedDelta.partial_json) ?? "";

    return;
  }

  const data = isToolBlockState(block) ? block.data : block;

  if (deltaType === "text_delta") appendStringField(data, "text", readString(capturedDelta.text));

  if (deltaType === "citations_delta") appendArrayField(data, "citations", capturedDelta.citation);

  if (deltaType === "thinking_delta")
    appendStringField(data, "thinking", readString(capturedDelta.thinking));
}

// A fallback block only marks where a fallback was attempted. The model that served the
// response is the one on the fallback_message entry in the terminal usage.iterations.
function fallbackServingModel<T>(usage: T, state: StreamState): string | undefined {
  const iterations = asRecord(usage)?.iterations;

  if (!Array.isArray(iterations)) return undefined;

  if (iterations.length > STREAM_CAPTURE_MAX_ITEMS) state.metadataTruncated = true;
  const firstIndex = Math.max(0, iterations.length - STREAM_CAPTURE_MAX_ITEMS);

  for (let index = iterations.length - 1; index >= firstIndex; index -= 1) {
    const entry = asRecord(iterations[index]);

    if (entry?.type === "fallback_message") return readString(entry.model);
  }

  return undefined;
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

    state.oneHourCacheCreationTokens =
      oneHourCacheCreationTokens(message.usage) ?? state.oneHourCacheCreationTokens;
    // A stream interrupted before message_delta keeps the iterations message_start reported.
    state.iterations = iterationsUsage(message.usage) ?? state.iterations;
  }

  if (type === "content_block_start") recordContentBlockStart(record, state);

  if (type === "content_block_delta") recordContentBlockDelta(record, state);

  if (type === "message_delta") {
    const delta = asRecord(record.delta) ?? {};
    state.finishReason = readString(delta.stop_reason) ?? state.finishReason;
    state.usage = mergeUsage(state.usage, messagesUsage(record.usage));

    state.oneHourCacheCreationTokens =
      oneHourCacheCreationTokens(record.usage) ?? state.oneHourCacheCreationTokens;
    state.responseModel = fallbackServingModel(record.usage, state) ?? state.responseModel;
    state.iterations = iterationsUsage(record.usage) ?? state.iterations;
  }

  if (type === "message_stop") state.messageStopped = true;

  return fields;
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
  endSpan: (fields?: SpanFields) => void,
): Stream<Value> {
  const capturePolicy = span.capturePolicy ?? LEGACY_CAPTURE_POLICY;
  const reportError = createSpanErrorReporter(span);

  const end = (fields: () => SpanFields) => {
    try {
      endSpan(fields());
    } finally {
      state.capture.release();
    }
  };

  const state: StreamState = {
    blocks: new Map(),
    captureOutput: capturePolicy.output,
    maskOutputWhenIncomplete: capturePolicy.mask,
    reservations: new Map(),
    capture: new StreamCapture(STREAM_OUTPUT_ENVELOPE.bytes),
    captureTruncated: false,
    metadataTruncated: false,
    expectMessageStop: true,
    messageStopped: false,
    unresolvedReplacements: new Set(),
    unresolvedEncryptedContent: new Set(),
  };

  let consumed = false;
  const signal = source.controller.signal;

  const endAborted = () => {
    if (consumed) return;
    end(() => ({
      ...streamPartialFields(state),
      error: signal.reason instanceof Error ? signal.reason : new Error("Request aborted"),
    }));
  };

  if (signal.aborted) endAborted();
  else signal.addEventListener("abort", endAborted, { once: true });

  async function* iterator() {
    consumed = true;
    // From here the iterator ends the span. A caller that keeps the controller and drops the
    // stream must not keep this state, and its bytes in the stream bound, reachable.
    signal.removeEventListener("abort", endAborted);
    const sawFirst = { value: false };
    let terminalError: Value;

    try {
      for await (const event of source) {
        const receivedAt = performance.now();
        let fields: SpanFields = {};
        let hasOutput = false;

        try {
          fields = recordStreamEvent(event, state);
          hasOutput = streamEventHasOutput(event);
        } catch (cause) {
          state.captureTruncated = true;
          reportError(cause);
        }

        setFirstStreamUpdate(span, startedAt, sawFirst, fields);

        if (hasOutput) span.recordOutputChunk?.(receivedAt);
        yield event;
      }
    } catch (error) {
      terminalError = error as Value;
      throw error;
    } finally {
      end(() => {
        const fields = streamPartialFields(state);

        if (terminalError !== undefined)
          fields.error =
            terminalError instanceof Error
              ? terminalError
              : new Error(unknownErrorMessage(terminalError));

        return fields;
      });
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

  if (value?.type === "compaction" || value?.type === "compaction_delta")
    return typeof value.content === "string" && value.content.length > 0;

  let inputHasValue = false;

  if (Array.isArray(input)) inputHasValue = input.length > 0;
  else if (input !== undefined) {
    const objectInput = input as { [key: string]: JsonValue | undefined };

    for (const key in objectInput) {
      if (!Object.hasOwn(input, key)) continue;
      inputHasValue = true;
      break;
    }
  }

  return (
    [value?.text, value?.thinking, value?.partial_json].some(
      (part) => typeof part === "string" && part.length > 0,
    ) || inputHasValue
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
        client: asRecord(this)?._client,
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

  const wrapped = wrapCreate(original, request, response, providerForResource);
  prototype[key] = wrapped;

  return () => {
    if (prototype[key] === wrapped) prototype[key] = original;
  };
}

let installed = false;
let restorePatches: Array<() => void> = [];

export function wrapAnthropic<T extends AnthropicClient>(client: T): T {
  if (wrappedClients.has(client)) return client;
  const provider = providerForClient(client);

  // beta.messages.stream(), parse(), toolRunner(), and thread() all route through
  // beta.messages.create().
  for (const resource of [client.messages, client.beta?.messages]) {
    const messages = asRecord(resource);

    if (messages)
      patchInstanceMethod(messages, "create", messagesRequest, messagesResponse, provider);
  }

  wrappedClients.add(client);

  return client;
}

export function instrumentAnthropic(): void {
  if (installed) return;
  restorePatches = [
    patchPrototype(Messages, "create", messagesRequest, messagesResponse),
    patchPrototype(BetaMessages, "create", messagesRequest, messagesResponse),
  ];
  installed = true;
}

export function uninstrumentAnthropic(): void {
  if (!installed) return;

  for (const restore of restorePatches.reverse()) restore();
  restorePatches = [];
  installed = false;
}
