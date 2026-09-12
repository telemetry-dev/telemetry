import {
  type Attributes,
  type Context,
  context as apiContext,
  diag,
  isSpanContextValid,
  type Span,
  type SpanContext,
  type SpanOptions,
  trace,
  TraceFlags,
  type Tracer,
  type TracerProvider,
} from "@opentelemetry/api";
import { getNumberFromEnv, getStringFromEnv } from "@opentelemetry/core";
import {
  AlwaysOffSampler,
  AlwaysOnSampler,
  ParentBasedSampler,
  type Sampler,
  TraceIdRatioBasedSampler,
} from "@opentelemetry/sdk-trace-base";

import { propagatedFromContext } from "./context.ts";
import { reportError } from "./debug.ts";

const SESSION_PARENTS = new WeakMap<Span, Context>();

/**
 * Install on the provider used with withSessionParent or sessionRootTracerProvider.
 * Pass the provider's configured sampler explicitly; OTel cannot read it back from a provider.
 * Without an argument, uses OTEL_TRACES_SAMPLER / OTEL_TRACES_SAMPLER_ARG defaults.
 */
export function sessionSampler(inner: Sampler = samplerFromEnv()): Sampler {
  return {
    shouldSample(ctx, traceId, name, kind, attributes, links) {
      const parent = trace.getSpan(ctx);
      const original = parent && SESSION_PARENTS.get(parent);

      if (original) {
        const span = trace.getSpan(original);
        ctx = span ? trace.setSpan(ctx, span) : trace.deleteSpan(ctx);
      }

      return inner.shouldSample(ctx, traceId, name, kind, attributes, links);
    },
    toString: () => `SessionSampler{${inner.toString()}}`,
  };
}

function samplerFromEnv(): Sampler {
  const name = getStringFromEnv("OTEL_TRACES_SAMPLER") ?? "parentbased_always_on";
  let root: Sampler;

  switch (name) {
    case "always_off":
    case "parentbased_always_off":
      root = new AlwaysOffSampler();
      break;
    case "traceidratio":
    case "parentbased_traceidratio": {
      const ratio = getNumberFromEnv("OTEL_TRACES_SAMPLER_ARG");
      const valid = ratio !== undefined && Number.isFinite(ratio) && ratio >= 0 && ratio <= 1;

      if (!valid) diag.error("Invalid OTEL_TRACES_SAMPLER_ARG; using 1.");
      root = new TraceIdRatioBasedSampler(valid ? ratio : 1);
      break;
    }

    case "always_on":
    case "parentbased_always_on":
      root = new AlwaysOnSampler();
      break;
    default:
      diag.error(`Invalid OTEL_TRACES_SAMPLER "${name}"; using parentbased_always_on.`);

      return new ParentBasedSampler({ root: new AlwaysOnSampler() });
  }

  return name.startsWith("parentbased_") ? new ParentBasedSampler({ root }) : root;
}

/**
 * Deterministic remote parent for a session: every root span of the session joins one trace
 * (`traceId = SHA-256(apiKey ‖ 0x00 ‖ sessionId)[0:16]`, parent `spanId = digest[16:24]`).
 * The API key is part of the input so two projects with the same session id never share a trace
 * id. No span is ever emitted for the parent. Its flags are not a sampling decision:
 * use withSessionParent and install sessionSampler on the provider. Python mirrors these IDs.
 */
export function sessionSpanContext(apiKey: string | undefined, sessionId: string): SpanContext {
  const digest = sha256(new TextEncoder().encode(`${apiKey ?? ""}\0${sessionId}`));

  return {
    traceId: hex(digest.subarray(0, 16)),
    spanId: hex(digest.subarray(16, 24)),
    traceFlags: TraceFlags.NONE,
    isRemote: true,
  };
}

/** Returns the session id when a span must start a new turn in the session trace. */
export type SessionRootOf = (name: string, attributes: Attributes) => string | undefined;

/**
 * Wraps a provider so that spans `sessionRootOf` recognizes are parented under the session
 * parent even when a parent is active. Frameworks that run each turn inside their own
 * engine span (eve's "workflow" scope) otherwise give every turn its own trace.
 * The inner provider MUST use sessionSampler(configuredSampler) to retain sampling policy.
 */
export function sessionRootTracerProvider(
  inner: TracerProvider,
  apiKey: string | undefined,
  sessionRootOf: SessionRootOf,
  onError?: (error: Error) => void,
): TracerProvider {
  const reparent = (name: string, options: SpanOptions | undefined, ctx: Context): Context => {
    if (!apiKey) return ctx;

    try {
      const sessionId = sessionRootOf(name, options?.attributes ?? {});

      return sessionId
        ? setSessionParent(options?.root ? trace.deleteSpan(ctx) : ctx, sessionId, apiKey)
        : ctx;
    } catch (error) {
      reportError(onError, error);

      return ctx;
    }
  };

  return {
    getTracer(name, version, options): Tracer {
      const tracer = inner.getTracer(name, version, options);

      return {
        startSpan: (spanName, spanOptions, ctx = apiContext.active()) => {
          const parent = reparent(spanName, spanOptions, ctx);

          return tracer.startSpan(
            spanName,
            parent !== ctx && spanOptions?.root ? { ...spanOptions, root: false } : spanOptions,
            parent,
          );
        },
        startActiveSpan<F extends (span: Span) => unknown>(
          spanName: string,
          ...args: unknown[]
        ): ReturnType<F> {
          const fn = args.at(-1) as F;
          const spanOptions = args.length > 1 ? (args[0] as SpanOptions) : undefined;
          const ctx = args.length > 2 ? (args[1] as Context) : apiContext.active();
          const parent = reparent(spanName, spanOptions, ctx);

          return tracer.startActiveSpan(
            spanName,
            parent !== ctx && spanOptions?.root
              ? { ...spanOptions, root: false }
              : (spanOptions ?? {}),
            parent,
            fn,
          );
        },
      };
    },
  };
}

/**
 * Parent a would-be root under the session; nested spans keep their real parent.
 * Requires sessionSampler(configuredSampler) on the provider that starts the span.
 */
export function withSessionParent(
  ctx: Context,
  sessionId: string | undefined,
  apiKey: string | undefined,
): Context {
  if (!apiKey || !sessionId) return ctx;
  const current = trace.getSpanContext(ctx);

  if (current && isSpanContextValid(current)) return ctx;

  return setSessionParent(ctx, sessionId, apiKey);
}

function setSessionParent(ctx: Context, sessionId: string, apiKey: string): Context {
  const parent = trace.wrapSpanContext({
    ...sessionSpanContext(apiKey, sessionId),
    traceState: trace.getSpanContext(ctx)?.traceState,
  });

  SESSION_PARENTS.set(parent, ctx);

  return trace.setSpan(ctx, parent);
}

export function sessionIdOf(ctx: Context, attributes?: Attributes): string | undefined {
  const explicit = attributes?.["gen_ai.conversation.id"];

  if (typeof explicit === "string") return explicit;
  const propagated = propagatedFromContext(ctx)?.["gen_ai.conversation.id"];

  return typeof propagated === "string" ? propagated : undefined;
}

function hex(bytes: Uint8Array): string {
  let s = "";

  for (const b of bytes) s += b.toString(16).padStart(2, "0");

  return s;
}

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));

/** FIPS 180-4 SHA-256 in plain JS: the package must stay synchronous and runtime-neutral. */
export function sha256(bytes: Uint8Array): Uint8Array {
  const h = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);

  const padded = new Uint8Array(Math.ceil((bytes.length + 9) / 64) * 64);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  const bitLen = bytes.length * 8;
  view.setUint32(padded.length - 8, Math.floor(bitLen / 0x100000000));
  view.setUint32(padded.length - 4, bitLen >>> 0);
  const w = new Uint32Array(64);

  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i * 4);

    for (let i = 16; i < 64; i++) {
      const x = w[i - 15]!;
      const y = w[i - 2]!;
      const s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >>> 3);
      const s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >>> 10);
      w[i] = (w[i - 16]! + s0 + w[i - 7]! + s1) >>> 0;
    }

    let a = h[0]!;
    let b = h[1]!;
    let c = h[2]!;
    let d = h[3]!;
    let e = h[4]!;
    let f = h[5]!;
    let g = h[6]!;
    let hh = h[7]!;

    for (let i = 0; i < 64; i++) {
      const t1 =
        (hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i]! + w[i]!) >>>
        0;

      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      hh = g;
      g = f;
      f = e;
      e = (d + t1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) >>> 0;
    }

    h[0] = (h[0]! + a) >>> 0;
    h[1] = (h[1]! + b) >>> 0;
    h[2] = (h[2]! + c) >>> 0;
    h[3] = (h[3]! + d) >>> 0;
    h[4] = (h[4]! + e) >>> 0;
    h[5] = (h[5]! + f) >>> 0;
    h[6] = (h[6]! + g) >>> 0;
    h[7] = (h[7]! + hh) >>> 0;
  }

  const out = new Uint8Array(32);
  const outView = new DataView(out.buffer);

  for (let i = 0; i < 8; i++) outView.setUint32(i * 4, h[i]!);

  return out;
}
