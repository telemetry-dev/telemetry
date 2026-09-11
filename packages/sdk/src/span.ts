import {
  type Context,
  defaultTextMapGetter,
  defaultTextMapSetter,
  type Link,
  ROOT_CONTEXT,
  type Span,
  type SpanContext,
  SpanKind,
  SpanStatusCode,
  trace,
} from "@opentelemetry/api";
import {
  CompositePropagator,
  parseTraceParent,
  W3CBaggagePropagator,
  W3CTraceContextPropagator,
} from "@opentelemetry/core";

import {
  activeContext,
  DURATION_BUCKETS,
  omitUndefined,
  OUTPUT_CHUNK_HISTOGRAM,
  type OutputChunkHistogram,
  PROPAGATED_KEY,
  propagatedFromContext,
  reportError,
  sessionIdOf,
  withContext,
  withSessionParent,
} from "@telemetry-dev/otel";

import {
  type FieldCaptureConfig,
  fieldsToAttributes,
  SPAN_TYPE_OPERATIONS,
  type SpanFields,
  type SpanType,
} from "./attrs.ts";
import { type ClientCore, currentClient } from "./client.ts";
import { NOOP_SPAN_HANDLE } from "./noop.ts";

export type ParentRef = string /* W3C traceparent */ | SpanHandle | SpanContext | Context;

export interface StartSpanOptions extends SpanFields {
  /** Default "span" → operation "function". */
  type?: SpanType;
  /** OpenTelemetry span kind; defaults to INTERNAL. */
  kind?: SpanKind;
  /** Parent override; defaults to the active context. */
  parent?: ParentRef;
  /** Links to associate with the span. */
  links?: Link[];
  startTime?: Date | number;
  /** Per-call override of the global captureInput. */
  captureInput?: boolean;
  /** Per-call override of the global captureOutput. */
  captureOutput?: boolean;
}

export interface SpanHandle {
  /** Raw OTel span (interop escape hatch). */
  readonly span: Span;
  /** Context with this span active, for manual parenting. */
  readonly context: Context;
  readonly traceId: string;
  readonly spanId: string;
  /** W3C traceparent for this span; null for the no-op handle. */
  readonly traceparent: string | null;
  readonly isRecording: boolean;
  update(fields: SpanFields): SpanHandle;
  /** Records an output chunk arrival. Pull-based streams include consumer delay between pulls. */
  recordOutputChunk(timestampMs?: number): void;
  end(fields?: SpanFields & { endTime?: Date | number }): void;
}

interface SpanMeta {
  type: SpanType;
  capture: FieldCaptureConfig;
  onError?: (cause: Error) => void;
}

// Remembers each span's type + capture flags so update()/end()/updateActiveSpan map
// input/output to the right gen_ai.* keys.
const SPAN_META = new WeakMap<Span, SpanMeta>();

const SEVERITY_ERROR = 17;

export function isThenable<T>(value: T): value is T & PromiseLike<Awaited<T>> {
  return (
    value != null &&
    (typeof value === "object" || typeof value === "function") &&
    "then" in value &&
    typeof value.then === "function"
  );
}

function isContext(value: Exclude<ParentRef, string>): value is Context {
  return "getValue" in value && typeof value.getValue === "function";
}

function isSpanHandle(value: Exclude<ParentRef, string>): value is SpanHandle {
  return "span" in value && "context" in value && "update" in value;
}

// Ambient propagated correlation attrs survive explicit parent overrides: contexts that lack
// PROPAGATED_KEY inherit it from the active context.
function withAmbientPropagated(ctx: Context): Context {
  if (propagatedFromContext(ctx)) return ctx;
  const ambient = propagatedFromContext(activeContext());
  return ambient ? ctx.setValue(PROPAGATED_KEY, ambient) : ctx;
}

function resolveParent(parent: ParentRef | undefined): Context {
  const base = activeContext();
  if (parent === undefined) return base;
  if (typeof parent === "string") {
    const parsed = parseTraceParent(parent);
    if (!parsed) return base;
    return trace.setSpanContext(base, { ...parsed, isRemote: true });
  }
  const ref = parent as Exclude<ParentRef, string>;
  if (isSpanHandle(ref)) return withAmbientPropagated(ref.context);
  if (isContext(ref)) return withAmbientPropagated(ref);
  return trace.setSpanContext(base, ref);
}

function applyError(span: Span, error: unknown): void {
  const err = error instanceof Error ? error : undefined;
  // Prefer the specific subclass name (Python parity: type(error).__name__) — many SDK
  // error classes (e.g. openai's BadRequestError) never override the inherited "Error" name.
  const errorType =
    err === undefined
      ? "Error"
      : err.name !== "Error" && err.name.length > 0
        ? err.name
        : (err.constructor?.name ?? "Error") || "Error";
  const message = err?.message ?? String(error);
  span.setStatus({ code: SpanStatusCode.ERROR });
  span.setAttribute("error.type", errorType);
  span.addEvent(
    "exception",
    omitUndefined({
      "exception.type": errorType,
      "exception.message": message,
      "exception.stacktrace": err?.stack,
      "log.severity_number": SEVERITY_ERROR,
    }),
  );
}

function applyFields(span: Span, meta: SpanMeta, fields: SpanFields): void {
  if (fields.name) span.updateName(fields.name);
  const attrs = fieldsToAttributes(fields, meta.type, meta.capture);
  if (Object.keys(attrs).length > 0) span.setAttributes(attrs);
  if (fields.error !== undefined) applyError(span, fields.error);
}

function metaFor(core: ClientCore, options?: StartSpanOptions): SpanMeta {
  const cfg = core.config;
  return {
    type: options?.type ?? "span",
    capture: {
      mask: cfg.mask,
      maxAttributeLength: cfg.maxAttributeLength,
      onError: cfg.onError,
      captureInput: options?.captureInput ?? cfg.captureInput,
      captureOutput: options?.captureOutput ?? cfg.captureOutput,
    },
    onError: cfg.onError,
  };
}

export function createSpanHandle(
  core: ClientCore,
  name: string,
  options: StartSpanOptions = {},
): SpanHandle {
  const meta = metaFor(core, options);
  const resolved = resolveParent(options.parent);
  const sessionId = sessionIdOf(resolved, options.attributes) ?? core.config.processSessionId;
  const sessionContext =
    sessionId === undefined
      ? resolved
      : resolved.setValue(PROPAGATED_KEY, {
          ...propagatedFromContext(resolved),
          "gen_ai.conversation.id": sessionId,
        });
  const parentCtx = withSessionParent(sessionContext, sessionId, core.config.apiKey);
  const attrs = fieldsToAttributes(options, meta.type, meta.capture);
  const span = core.tracer.startSpan(
    name,
    {
      kind: options.kind ?? SpanKind.INTERNAL,
      startTime: options.startTime,
      links: options.links,
      attributes: {
        "gen_ai.operation.name": SPAN_TYPE_OPERATIONS[meta.type],
        ...propagatedFromContext(parentCtx),
        ...attrs,
      },
    },
    parentCtx,
  );
  SPAN_META.set(span, meta);
  // The processor stamps propagation on start; explicit fields still win.
  if (Object.keys(attrs).length > 0) span.setAttributes(attrs);
  if (options.name) span.updateName(options.name);
  if (options.error !== undefined) applyError(span, options.error);

  const spanContext = span.spanContext();
  let previousOutputChunkTime: number | undefined;
  const flags = (spanContext.traceFlags & 0xff).toString(16).padStart(2, "0");
  const handle: SpanHandle = {
    span,
    context: trace.setSpan(parentCtx, span),
    traceId: spanContext.traceId,
    spanId: spanContext.spanId,
    traceparent: `00-${spanContext.traceId}-${spanContext.spanId}-${flags}`,
    get isRecording() {
      return span.isRecording();
    },
    update(fields) {
      try {
        applyFields(span, meta, fields);
      } catch (error) {
        reportError(meta.onError, error instanceof Error ? error : new Error(String(error)));
      }
      return handle;
    },
    recordOutputChunk(timestampMs) {
      const now = timestampMs ?? performance.now();

      if (
        !span.isRecording() ||
        !Number.isFinite(now) ||
        (previousOutputChunkTime !== undefined && now < previousOutputChunkTime)
      )
        return;

      if (previousOutputChunkTime !== undefined) {
        const seconds = Math.max(0, now - previousOutputChunkTime) / 1000;

        const target = span as Span & {
          [OUTPUT_CHUNK_HISTOGRAM]?: OutputChunkHistogram;
        };

        const histogram = (target[OUTPUT_CHUNK_HISTOGRAM] ??= {
          count: 0,
          sum: 0,
          min: seconds,
          max: seconds,
          bucketCounts: Array.from({ length: DURATION_BUCKETS.length + 1 }, () => 0),
        });

        histogram.count++;
        histogram.sum += seconds;
        histogram.min = Math.min(histogram.min, seconds);
        histogram.max = Math.max(histogram.max, seconds);
        const bucket = DURATION_BUCKETS.findIndex((boundary) => seconds <= boundary);
        histogram.bucketCounts[bucket < 0 ? DURATION_BUCKETS.length : bucket]++;
      }

      previousOutputChunkTime = now;
    },
    end(fields) {
      try {
        if (fields) applyFields(span, meta, fields);
      } catch (error) {
        reportError(meta.onError, error instanceof Error ? error : new Error(String(error)));
      }
      span.end(fields?.endTime);
    },
  };
  return handle;
}

/** Start a span WITHOUT activating context; the caller must call .end(). */
export function startSpan(name: string, options?: StartSpanOptions): SpanHandle {
  const core = currentClient().core;
  if (!core) return NOOP_SPAN_HANDLE;
  try {
    return createSpanHandle(core, name, options);
  } catch (error) {
    reportError(core.config.onError, error instanceof Error ? error : new Error(String(error)));
    return NOOP_SPAN_HANDLE;
  }
}

export function runWithHandle<T>(handle: SpanHandle, fn: (span: SpanHandle) => T): T {
  try {
    const result = withContext(handle.context, () => fn(handle));
    if (isThenable(result)) {
      return result.then(
        (value) => {
          handle.end();
          return value;
        },
        (error: unknown) => {
          handle.end({ error });
          throw error;
        },
      ) as T;
    }
    handle.end();
    return result;
  } catch (error) {
    handle.end({ error });
    throw error;
  }
}

export function startActiveSpan<T>(name: string, fn: (span: SpanHandle) => T): T;
export function startActiveSpan<T>(
  name: string,
  options: StartSpanOptions,
  fn: (span: SpanHandle) => T,
): T;
export function startActiveSpan<T>(
  name: string,
  optionsOrFn: StartSpanOptions | ((span: SpanHandle) => T),
  maybeFn?: (span: SpanHandle) => T,
): T {
  const isFn = typeof optionsOrFn === "function";
  const fn = isFn ? optionsOrFn : maybeFn!;
  const options = isFn ? undefined : optionsOrFn;
  const handle = startSpan(name, options);
  if (handle === NOOP_SPAN_HANDLE) return fn(handle);
  return runWithHandle(handle, fn);
}

/** Update the innermost active span; no-op when none is recording. */
export function updateActiveSpan(fields: SpanFields): void {
  const core = currentClient().core;
  if (!core) return;
  try {
    const span = trace.getSpan(activeContext());
    if (!span?.isRecording()) return;
    const meta = SPAN_META.get(span) ?? metaFor(core);
    applyFields(span, meta, fields);
  } catch (error) {
    reportError(core.config.onError, error instanceof Error ? error : new Error(String(error)));
  }
}

const TRACE_CONTEXT_PROPAGATOR = new W3CTraceContextPropagator();
const W3C_PROPAGATOR = new CompositePropagator({
  propagators: [TRACE_CONTEXT_PROPAGATOR, new W3CBaggagePropagator()],
});

export { activeContext, withContext };

/** Extract W3C trace context; baggage is omitted unless includeBaggage is true. */
export function extractW3cContext(
  carrier: Record<string, unknown>,
  options: { includeBaggage?: boolean } = {},
): Context {
  const propagator = options.includeBaggage === true ? W3C_PROPAGATOR : TRACE_CONTEXT_PROPAGATOR;
  return propagator.extract(ROOT_CONTEXT, carrier, defaultTextMapGetter);
}

/** Inject W3C trace context; baggage is omitted unless includeBaggage is true. */
export function injectW3cContext(
  context: Context,
  carrier: Record<string, unknown>,
  options: { includeBaggage?: boolean } = {},
): void {
  const propagator = options.includeBaggage === true ? W3C_PROPAGATOR : TRACE_CONTEXT_PROPAGATOR;
  for (const key of Object.keys(carrier)) {
    const normalized = key.toLowerCase();
    if (normalized === "traceparent" || normalized === "tracestate" || normalized === "baggage") {
      delete carrier[key];
    }
  }
  propagator.inject(context, carrier, defaultTextMapSetter);
}

/** Serialize the active span context as a W3C traceparent; null when there is none. */
export function getTraceparent(): string | null {
  const carrier = { traceparent: undefined };
  TRACE_CONTEXT_PROPAGATOR.inject(activeContext(), carrier, defaultTextMapSetter);
  return carrier.traceparent ?? null;
}
