import { startSpan, type SpanFields, type SpanHandle, type SpanType } from "@telemetry-dev/sdk";

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
    const span = startSpan(spanName, { type: spanType, ...fields });
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
          nextData = handler.onResult(asJsonValue(rawData), span, t0, input, options);
        } catch {
          endSpan(span, awsMetadataFields(metadataOf(rawData)));
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
          return handler.onResult(asJsonValue(resolved), span, t0, input, options);
        } catch {
          endSpan(span, awsMetadataFields(metadataOf(resolved)));

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

export function endSpan(span: SpanHandle, fields?: SpanFields): void {
  try {
    span.end(fields);
  } catch {
    // Instrumentation must never throw into caller code.
  }
}

export function updateSpan(span: SpanHandle, fields: SpanFields): void {
  try {
    span.update(fields);
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

export function bytesToString<T>(value: T): string | undefined {
  const raw: unknown = value;

  if (String(raw) === raw) return raw;

  if (raw instanceof Uint8Array) return new TextDecoder().decode(raw);

  if (ArrayBuffer.isView(raw)) {
    return new TextDecoder().decode(new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
  }
}

export function parseJson<T>(value: T): JsonValue {
  const text = bytesToString(value);

  if (text === undefined) return undefined;

  try {
    return asJsonValue(JSON.parse(text));
  } catch {
    return undefined;
  }
}

export interface StreamState {
  feed<T>(event: T): boolean | void;
  finish(partial: boolean): SpanFields;
}

export function wrapAsyncIterable<T>(
  iterable: T,
  state: StreamState,
  span: SpanHandle,
  t0: number,
  baseFields: SpanFields,
  trackOutputChunks = true,
): T {
  if (!iterable || !isAsyncIterable(iterable)) {
    endSpan(span, baseFields);

    return iterable;
  }

  let ended = false;
  let first = false;
  const unregisterToken = {};

  const endOnce = (fields?: SpanFields) => {
    if (ended) return;
    ended = true;
    STREAM_FINALIZER?.unregister(unregisterToken);
    endSpan(span, mergeFields(baseFields, fields));
  };

  const generator = async function* () {
    let completed = false;

    try {
      for await (const event of iterable) {
        const receivedAt = performance.now();

        if (!first) {
          first = true;
          updateSpan(span, { timeToFirstChunkMs: receivedAt - t0 });
        }

        const stateHasOutput = safe(() => state.feed(event));

        if (trackOutputChunks && (stateHasOutput ?? bedrockEventHasOutput(event))) {
          safe(() => span.recordOutputChunk?.(receivedAt));
        }

        yield event;
      }

      completed = true;
    } catch (error) {
      endOnce(
        mergeFields(state.finish(true), awsMetadataFields(metadataOf(error)), {
          error: asError(error),
        }),
      );
      throw error;
    } finally {
      if (!ended) endOnce(state.finish(!completed));
    }
  };

  const wrapped: { [key: PropertyKey]: JsonValue | (() => AsyncGenerator<unknown>) } = {};

  if (isRecord(iterable)) {
    for (const key of Reflect.ownKeys(iterable)) {
      wrapped[key] = (iterable as { [key: PropertyKey]: JsonValue })[key];
    }
  }

  const finishAbandoned = () => endOnce(safe(() => state.finish(true)));
  wrapped[Symbol.asyncIterator] = () => {
    const iterator = generator();
    STREAM_FINALIZER?.register(iterator, finishAbandoned, unregisterToken);

    return iterator;
  };

  STREAM_FINALIZER?.register(wrapped, finishAbandoned, unregisterToken);

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
