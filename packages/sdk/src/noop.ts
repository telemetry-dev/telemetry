import { INVALID_SPAN_CONTEXT, ROOT_CONTEXT, trace } from "@opentelemetry/api";

import type { SpanHandle } from "./span.ts";

const noopSpan = trace.wrapSpanContext(INVALID_SPAN_CONTEXT);

/** Shared inert handle returned by span primitives before init() or when disabled. */
export const NOOP_SPAN_HANDLE: SpanHandle = {
  span: noopSpan,
  context: trace.setSpan(ROOT_CONTEXT, noopSpan),
  traceId: INVALID_SPAN_CONTEXT.traceId,
  spanId: INVALID_SPAN_CONTEXT.spanId,
  traceparent: null,
  isRecording: false,
  update: () => NOOP_SPAN_HANDLE,
  end: () => {},
};
