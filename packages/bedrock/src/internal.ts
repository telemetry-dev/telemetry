import {
  boundedCaptureDetails,
  startSpan,
  type SpanFields,
  type SpanHandle,
  type SpanType,
  type StartSpanOptions,
  StreamCapture,
} from "@telemetry-dev/sdk";
import * as telemetrySdk from "@telemetry-dev/sdk";

export const PROVIDER = "amazon-bedrock";

const STREAM_FINALIZER =
  typeof FinalizationRegistry === "function"
    ? new FinalizationRegistry<() => void>((finish) => finish())
    : undefined;

export interface BedrockInstrumentationOptions {
  captureAgentTrace?: boolean;
}

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | JsonValue[]
  | JsonRecord
  | ((...args: JsonValue[]) => JsonValue);

export interface JsonRecord {
  [key: string]: JsonValue;
}

export type SendFn = (...args: unknown[]) => JsonValue | object | Promise<JsonValue | object>;

export interface OpHandler {
  spanName(input: JsonRecord): string;
  spanType(input: JsonRecord): SpanType;
  requestFields(input: JsonRecord, options: BedrockInstrumentationOptions): SpanFields;
  onResult(
    result: JsonValue,
    span: SpanHandle,
    t0: number,
    input: JsonRecord,
    options: BedrockInstrumentationOptions,
    reportError: (cause: unknown) => void,
  ): JsonValue | object;
}

export function createInstrumentedSend(
  originalSend: SendFn,
  options: BedrockInstrumentationOptions,
  handlers: { [key: string]: OpHandler },
): SendFn {
  return function instrumentedSend<TThis, TCommand, TRest extends unknown[]>(
    this: TThis,
    command: TCommand,
    ...rest: TRest
  ) {
    const rawCommand = asJsonValue(command);
    const rawRest = rest.map(asJsonValue);
    const name = isRecord(rawCommand) ? rawCommand.constructor?.name : undefined;
    const handler = String(name) === name ? handlers[name] : undefined;

    if (!handler) {
      return originalSend.apply(this, [rawCommand, ...rawRest]);
    }

    const input = commandInput(rawCommand);
    const t0 = performance.now();
    const spanType = safe(() => handler.spanType(input)) ?? "span";
    const spanName = safe(() => handler.spanName(input)) ?? name ?? "bedrock";
    const fields = safe(() => handler.requestFields(input, options)) ?? {};
    const span = startRequestSpan(spanName, { type: spanType, ...fields });
    const reportError = createSpanErrorReporter(span);
    const callbackIndex = rawRest.findIndex((arg) => arg instanceof Function);

    if (callbackIndex !== -1) {
      const callback = rawRest[callbackIndex] as (cause: JsonValue, data?: JsonValue) => void;
      const wrappedRest = [...rawRest];
      wrappedRest[callbackIndex] = <TError, TData>(error: TError, data?: TData) => {
        const rawError: unknown = error;
        const rawData: unknown = data;

        if (rawError != null) {
          endSpan(span, { ...awsMetadataFields(metadataOf(rawError)), error: asError(rawError) });
          callback(asJsonValue(rawError), asJsonValue(rawData));

          return;
        }

        let nextData: JsonValue | object = asJsonValue(rawData);

        try {
          nextData = handler.onResult(asJsonValue(rawData), span, t0, input, options, reportError);
        } catch (cause) {
          endUnmappedSpan(span, rawData, cause, reportError);
        }

        callback(asJsonValue(rawError), asJsonValue(nextData));
      };

      try {
        return originalSend.apply(this, [rawCommand, ...wrappedRest]);
      } catch (error) {
        endSpan(span, { ...awsMetadataFields(metadataOf(error)), error: asError(error) });
        throw error;
      }
    }

    let result: JsonValue | object | Promise<JsonValue | object>;

    try {
      result = originalSend.apply(this, [rawCommand, ...rawRest]);
    } catch (error) {
      endSpan(span, { ...awsMetadataFields(metadataOf(error)), error: asError(error) });
      throw error;
    }

    return Promise.resolve(result).then(
      (resolved) => {
        try {
          return handler.onResult(asJsonValue(resolved), span, t0, input, options, reportError);
        } catch (cause) {
          endUnmappedSpan(span, resolved, cause, reportError);

          return resolved;
        }
      },
      (error) => {
        endSpan(span, { ...awsMetadataFields(metadataOf(error)), error: asError(error) });
        throw error;
      },
    );
  };
}

export function commandInput<T>(command: T): JsonRecord {
  if (!isRecord(command)) return {};

  return isRecord(command.input) ? command.input : {};
}

export function isRecord<T>(value: T): value is T & JsonRecord {
  return value !== null && Object(value) === value && !Array.isArray(value);
}

export function safe<T>(fn: () => T): T | undefined {
  try {
    return fn();
  } catch {
    return undefined;
  }
}

// reportError is optional on SpanHandle, so older cores without it still work.
function reportSpanError(span: SpanHandle, cause: unknown): void {
  try {
    span.reportError?.(cause);
  } catch {
    return;
  }
}

// One report per stream or call: a mapper that fails on one event likely fails on each one after it.
function createSpanErrorReporter(span: SpanHandle): (cause: unknown) => void {
  let reported = false;

  return (cause) => {
    if (reported) return;
    reported = true;
    reportSpanError(span, cause);
  };
}

// The call succeeded, so the span is not failed; its output and response fields are lost.
function endUnmappedSpan<T>(
  span: SpanHandle,
  response: T,
  cause: unknown,
  reportError: (cause: unknown) => void,
): void {
  reportError(cause);
  const metadata = safe(() => awsMetadataFields(metadataOf(response)));
  endSpan(span, mergeFields(metadata, { attributes: { [CAPTURE_TRUNCATED_ATTRIBUTE]: true } }));
}

/**
 * A stream response's own fields. When they fail to map, the stream still records its output
 * from its events.
 */
export function streamResponseFields<T>(
  response: T,
  reportError: (cause: unknown) => void,
  fields: () => SpanFields,
): SpanFields {
  try {
    return fields();
  } catch (cause) {
    reportError(cause);

    return safe(() => awsMetadataFields(metadataOf(response))) ?? {};
  }
}

// Bedrock reports input tokens without cache reads and writes; span input tokens include them.
function withCacheInclusiveInput(fields: SpanFields | undefined): SpanFields | undefined {
  const usage = fields?.usage;

  if (usage?.inputTokens === undefined) return fields;

  const inputTokens =
    usage.inputTokens + (usage.cacheReadInputTokens ?? 0) + (usage.cacheCreationInputTokens ?? 0);

  const totalTokens =
    usage.totalTokens !== undefined && usage.outputTokens !== undefined
      ? inputTokens + usage.outputTokens
      : usage.totalTokens;

  return { ...fields, usage: { ...usage, inputTokens, totalTokens } };
}

// A capture cut to the retention bound keeps its truncated flag, but a mask cannot be trusted to
// redact a value it only sees part of. Cores without a capture policy are treated as masked.
function capturesWithMask(span: SpanHandle): boolean {
  return span.capturePolicy?.mask ?? true;
}

function isCut(fields: SpanFields | undefined): boolean {
  return fields?.attributes?.["telemetry.dev.capture.truncated"] === true;
}

// Cores older than captureMasked expose no mask policy before a span starts; treat them as
// masked, like LEGACY_CAPTURE_POLICY. A namespace read keeps them loadable without the export.
export function captureMaskedBeforeStart(): boolean {
  const read = (telemetrySdk as { captureMasked?: () => boolean }).captureMasked;

  try {
    return typeof read === "function" ? read() : true;
  } catch {
    return true;
  }
}

// Cores older than promptAttributesFor record a prompt only inside withPrompt. A namespace read
// keeps them loadable without the export.
export function promptAttributes(messages: unknown): SpanFields["attributes"] {
  const read = (
    telemetrySdk as { promptAttributesFor?: (messages: unknown) => SpanFields["attributes"] }
  ).promptAttributesFor;

  return typeof read === "function" ? read(messages) : undefined;
}

// The decision is made before the span starts so samplers see the same attributes as before.
function startRequestSpan(name: string, options: StartSpanOptions): SpanHandle {
  const withheld = options.input !== undefined && isCut(options) && captureMaskedBeforeStart();

  return startSpan(name, withheld ? { ...options, input: undefined } : options);
}

/** Ends a non-streamed span, withholding output cut to the retention bound from a mask. */
export function endSpanWithBoundedOutput(span: SpanHandle, fields: SpanFields): void {
  const withheld = fields.output !== undefined && isCut(fields) && capturesWithMask(span);

  endSpan(span, withheld ? { ...fields, output: undefined } : fields);
}

export function endSpan(span: SpanHandle, fields?: SpanFields): void {
  try {
    span.end(withCacheInclusiveInput(fields));
  } catch {
    // Instrumentation must never throw into caller code.
  }
}

export function metadataOf<T>(value: T): JsonValue {
  if (!isRecord(value)) return undefined;

  return value.$metadata;
}

export function modeledStreamError(
  event: JsonRecord,
): { error: Error; fields: SpanFields } | undefined {
  const keys = {
    accessDeniedException: "AccessDeniedException",
    badGatewayException: "BadGatewayException",
    conflictException: "ConflictException",
    dependencyFailedException: "DependencyFailedException",
    internalServerException: "InternalServerException",
    modelNotReadyException: "ModelNotReadyException",
    modelStreamErrorException: "ModelStreamErrorException",
    modelTimeoutException: "ModelTimeoutException",
    resourceNotFoundException: "ResourceNotFoundException",
    serviceQuotaExceededException: "ServiceQuotaExceededException",
    serviceUnavailableException: "ServiceUnavailableException",
    throttlingException: "ThrottlingException",
    validationException: "ValidationException",
  } as const;

  for (const [key, name] of Object.entries(keys)) {
    const exception = isRecord(event[key]) ? event[key] : undefined;

    if (!exception) continue;

    const error = new Error(
      stringValue(exception.message) ?? stringValue(exception.originalMessage) ?? name,
    );

    error.name = stringValue(exception.name) ?? name;

    return { error, fields: awsMetadataFields(exception.$metadata) };
  }

  return undefined;
}

export function awsMetadataFields<T>(metadata: T): SpanFields {
  const raw: unknown = metadata;

  if (!isRecord(raw)) return {};
  const requestId = stringValue(raw.requestId);
  const attempts = numberValue(raw.attempts);
  const httpStatusCode = numberValue(raw.httpStatusCode);
  const totalRetryDelay = numberValue(raw.totalRetryDelay);
  const attributes: SpanFields["attributes"] = {};

  if (httpStatusCode !== undefined) attributes["aws.http.status_code"] = httpStatusCode;

  if (attempts !== undefined && attempts > 1) attributes["aws.request.attempts"] = attempts;

  if (totalRetryDelay !== undefined) {
    attributes["aws.request.total_retry_delay_ms"] = totalRetryDelay;
  }

  return omitUndefined({
    responseId: requestId,
    attributes: Object.keys(attributes).length > 0 ? attributes : undefined,
  });
}

export function mergeFields(...fields: Array<SpanFields | undefined>): SpanFields {
  const merged: SpanFields = {};

  for (const field of fields) {
    if (!field) continue;
    const { metadata, attributes, usage, ...rest } = field;
    Object.assign(merged, omitUndefined(rest));

    if (usage) merged.usage = { ...merged.usage, ...usage };

    if (metadata) merged.metadata = { ...merged.metadata, ...metadata };

    if (attributes) merged.attributes = { ...merged.attributes, ...attributes };
  }

  return merged;
}

export function omitUndefined<T extends object>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined)) as T;
}

export function stringValue<T>(value: T): string | undefined {
  const raw: unknown = value;

  return String(raw) === raw && raw.length > 0 ? raw : undefined;
}

export function numberValue<T>(value: T): number | undefined {
  const raw: unknown = value;

  return Number(raw) === raw && Number.isFinite(raw) ? raw : undefined;
}

export function arrayValue<T, TValue = unknown>(value: TValue): T[] | undefined {
  return Array.isArray(value) ? (value as T[]) : undefined;
}

/** Decodes UTF-8 bytes; with `fatal`, bytes that are not UTF-8 throw instead of decoding lossily. */
export function bytesToString<T>(value: T, fatal = false): string | undefined {
  const raw: unknown = value;

  if (String(raw) === raw) return raw;

  const decoder = new TextDecoder("utf-8", { fatal });

  if (raw instanceof Uint8Array) return decoder.decode(raw);

  if (ArrayBuffer.isView(raw)) {
    return decoder.decode(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
  }
}

/**
 * Returns undefined for a value that is not text or bytes, or is not valid JSON. With `strict`,
 * bytes that are not UTF-8 and non-blank text that is not valid JSON throw instead.
 */
export function parseJson<T>(value: T, strict = false): JsonValue {
  const text = bytesToString(value, strict);

  if (text === undefined) return undefined;

  try {
    return asJsonValue(JSON.parse(text));
  } catch (cause) {
    if (strict && text.trim()) throw cause;

    return undefined;
  }
}

// The usage iterations a stream keeps; the core applies maxAttributeLength after the mask.
export const STREAM_CAPTURE_MAX_ITEMS = 1_000;
const CAPTURE_TRUNCATED_ATTRIBUTE = "telemetry.dev.capture.truncated";

// Older cores enforce captureOutput and masking when the span ends but expose no policy, so
// withhold incomplete output from their mask. Retention uses the stream bounds either way;
// maxAttributeLength is only here to satisfy the policy type.
const LEGACY_CAPTURE_POLICY: NonNullable<SpanHandle["capturePolicy"]> = {
  output: true,
  mask: true,
  maxAttributeLength: 65_536,
};

/**
 * A stream state's share of the shared stream bound. After the first value that does not fit,
 * it accepts nothing more, so the kept output is a prefix of the stream.
 */
export class StreamRetention {
  private readonly capture = new StreamCapture();
  truncated = false;

  accept<T>(value: T): boolean {
    if (this.truncated) return false;

    const measured = boundedCaptureDetails(value, {
      maxBytes: this.capture.limit - this.capture.bytes,
    });

    if (measured.truncated) {
      this.truncated = true;

      return false;
    }

    this.capture.bytes += measured.bytes;

    return true;
  }

  release(): void {
    this.capture.release();
  }
}

export interface StreamState {
  feed<T>(event: T): boolean | void;
  finish(partial: boolean): SpanFields;
  readonly retention: StreamRetention;
  /** Whether the exported output itself was cut; defaults to the retention flag. */
  outputTruncated?(): boolean;
  /**
   * feed() sets it once the output of the event is kept, so a failure after it loses only
   * metadata. A state that does not set it has every failure count as lost output.
   */
  outputRecorded?: boolean;
  /** Set when kept metadata is lost: it flags the capture without cutting the output. */
  metadataLost?: boolean;
  /** Called after the retention is cut when an event failed before its output was kept. */
  outputLost?(): void;
  /** The stream that feeds the state sets it, for failures that feed() handles itself. */
  reportError?: (cause: unknown) => void;
}

function finishStream(
  state: StreamState,
  span: SpanHandle,
  partial: boolean,
  reportError: (cause: unknown) => void,
): SpanFields {
  let fields: SpanFields;

  try {
    fields = state.finish(partial);
  } catch (cause) {
    // The output that the stream kept is lost.
    reportError(cause);
    state.retention.truncated = true;
    fields = {};
  }

  const interrupted = partial || fields.error !== undefined;

  if (!interrupted && !state.retention.truncated && !state.metadataLost) return fields;

  fields.attributes = { ...fields.attributes, [CAPTURE_TRUNCATED_ATTRIBUTE]: true };
  const outputTruncated = state.outputTruncated?.() ?? state.retention.truncated;

  if ((span.capturePolicy ?? LEGACY_CAPTURE_POLICY).mask && (interrupted || outputTruncated)) {
    delete fields.output;
  }

  return fields;
}

export function wrapAsyncIterable<T>(
  iterable: T,
  state: StreamState,
  span: SpanHandle,
  t0: number,
  baseFields: SpanFields,
  reportError: (cause: unknown) => void,
  trackOutputChunks = true,
): T {
  if (!iterable || !isAsyncIterable(iterable)) {
    endSpan(span, baseFields);

    return iterable;
  }

  state.reportError = reportError;
  let ended = false;
  let first = false;
  const unregisterToken = {};
  const wrapperToken = {};

  const endOnce = (fields?: SpanFields) => {
    if (ended) return;
    ended = true;
    STREAM_FINALIZER?.unregister(unregisterToken);
    STREAM_FINALIZER?.unregister(wrapperToken);

    try {
      endSpan(span, mergeFields(baseFields, fields));
    } finally {
      state.retention.release();
    }
  };

  const generator = async function* () {
    let completed = false;

    try {
      for await (const event of iterable) {
        const receivedAt = performance.now();

        if (!first) {
          first = true;
          try {
            span.update({ timeToFirstChunkMs: receivedAt - t0 });
          } catch (cause) {
            reportError(cause);
          }
        }

        state.outputRecorded = false;
        let stateHasOutput: boolean | void | undefined;

        try {
          stateHasOutput = state.feed(event);
        } catch (cause) {
          reportError(cause);

          if (!state.outputRecorded) {
            // The event may have carried output that is now lost. The cut retention keeps no
            // later output, so the output stays a prefix of the stream.
            state.retention.truncated = true;
            state.outputLost?.();
          }
        }

        // Chunk timing must not reach the caller. A failure loses only the timing; when feed()
        // already failed, the fallback can fail the same way and the reporter drops the repeat.
        try {
          if (trackOutputChunks && (stateHasOutput ?? bedrockEventHasOutput(event))) {
            span.recordOutputChunk?.(receivedAt);
          }
        } catch (cause) {
          reportError(cause);
        }

        yield event;
      }

      completed = true;
    } catch (error) {
      endOnce(
        mergeFields(
          finishStream(state, span, true, reportError),
          awsMetadataFields(metadataOf(error)),
          { error: asError(error) },
        ),
      );
      throw error;
    } finally {
      try {
        if (!ended) endOnce(finishStream(state, span, !completed, reportError));
      } finally {
        // Also when finishing the stream throws before the span could end.
        state.retention.release();
      }
    }
  };

  const wrapped: { [key: PropertyKey]: JsonValue | (() => AsyncGenerator<unknown>) } = {};

  if (isRecord(iterable)) {
    for (const key of Reflect.ownKeys(iterable)) {
      wrapped[key] = (iterable as { [key: PropertyKey]: JsonValue })[key];
    }
  }

  const finishAbandoned = () => endOnce(safe(() => finishStream(state, span, true, reportError)));
  // A caller may keep only the iterator, so once one exists the wrapper is not watched.
  wrapped[Symbol.asyncIterator] = () => {
    const iterator = generator();
    STREAM_FINALIZER?.unregister(wrapperToken);
    STREAM_FINALIZER?.register(iterator, finishAbandoned, unregisterToken);

    return iterator;
  };

  STREAM_FINALIZER?.register(wrapped, finishAbandoned, wrapperToken);

  return wrapped as T;
}

function bedrockEventHasOutput(event: unknown): boolean {
  const record = isRecord(event) ? event : undefined;

  if (!record) return false;
  const output = isRecord(record.output) ? record.output : undefined;

  if (typeof output?.text === "string" && output.text.length > 0) return true;
  const block = isRecord(record.contentBlockDelta) ? record.contentBlockDelta : undefined;
  const delta = isRecord(block?.delta) ? block.delta : undefined;
  const reasoning = isRecord(delta?.reasoningContent) ? delta.reasoningContent : undefined;
  const toolUse = isRecord(delta?.toolUse) ? delta.toolUse : undefined;

  if (
    (typeof delta?.text === "string" && delta.text.length > 0) ||
    (typeof reasoning?.text === "string" && reasoning.text.length > 0) ||
    (typeof toolUse?.input === "string" && toolUse.input.length > 0)
  )
    return true;
  const chunk = isRecord(record.chunk) ? record.chunk : undefined;
  const raw = chunk?.bytes;

  if (raw === undefined) return false;

  const text =
    typeof raw === "string"
      ? raw
      : raw instanceof Uint8Array
        ? new TextDecoder().decode(raw)
        : undefined;

  if (!text) return false;
  const parsed = parseJson(text);

  if (parsed === undefined) return text.length > 0;

  if (!isRecord(parsed)) return false;
  const parsedDelta = isRecord(parsed.delta) ? parsed.delta : undefined;

  const contentDelta = isRecord(parsed.contentBlockDelta ?? parsed.content_block_delta)
    ? (parsed.contentBlockDelta ?? parsed.content_block_delta)
    : undefined;

  const nested =
    isRecord(contentDelta) && isRecord(contentDelta.delta) ? contentDelta.delta : undefined;

  const generations = Array.isArray(parsed.generations) ? parsed.generations : [];
  const generation = isRecord(generations[0]) ? generations[0] : undefined;
  const outputs = Array.isArray(parsed.outputs) ? parsed.outputs : [];
  const firstOutput = isRecord(outputs[0]) ? outputs[0] : undefined;

  return [
    parsed.outputText,
    parsed.generation,
    parsed.completion,
    parsed.text,
    parsedDelta?.text,
    parsedDelta?.thinking,
    parsedDelta?.partial_json,
    nested?.text,
    firstOutput?.text,
    generation?.text,
  ].some((value) => typeof value === "string" && value.length > 0);
}

function isAsyncIterable<T>(value: T): value is T & AsyncIterable<unknown> {
  return Object(value) === value && Symbol.asyncIterator in Object(value);
}

function asJsonValue<T>(value: T): JsonValue {
  return value as JsonValue;
}

function asError<T>(value: T): Error {
  return value instanceof Error ? value : new Error(String(value));
}

export function basenameArn<T>(value: T): string | undefined {
  const text = stringValue(value);

  if (!text) return undefined;

  return text.split(/[/:]/).filter(Boolean).at(-1) ?? text;
}
