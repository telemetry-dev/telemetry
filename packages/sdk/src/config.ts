import type { AttributeValue } from "@opentelemetry/api";
import type { ReadableSpan, Sampler } from "@opentelemetry/sdk-trace-base";
import {
  type BatchOptions,
  DEFAULT_BASE_URL,
  DEFAULT_BATCH,
  type ExportMode,
  resolveEnv,
  type SdkLogLevel,
  type SessionRootOf,
} from "@telemetry-dev/otel";

export type { BatchOptions, ExportMode, LogLevel, SdkLogLevel } from "@telemetry-dev/otel";

/** Redaction hook applied to captured content (input/output/log message) before JSON stringification. */
export type MaskFn<T = unknown> = (value: T, ctx: { key: string }) => T;

export interface TelemetryOptions {
  /** telemetry.dev ingest key (`td_live_…`). Falls back to `TELEMETRY_DEV_API_KEY`. When absent the SDK is a no-op. */
  apiKey?: string;
  /** Ingest base URL. Falls back to `TELEMETRY_DEV_BASE_URL`, then `https://ingest.telemetry.dev`. */
  baseUrl?: string;
  /** Deployment environment label. Falls back to `TELEMETRY_DEV_ENVIRONMENT`, then `production`. */
  environment?: string;
  /** Service name attached to every trace. Falls back to `OTEL_SERVICE_NAME`, then `unknown_service`. */
  serviceName?: string;
  /** Kill switch: `false` makes the SDK a complete no-op (tests). Default true. */
  enabled?: boolean;
  /** Group otherwise uncorrelated spans and logs for this SDK initialization. Default "explicit". */
  sessionMode?: "explicit" | "process";
  /** Register this SDK's TracerProvider/context manager/propagator globally. Default false (isolated). */
  registerGlobal?: boolean;
  /** "batched" (default) buffers spans; "immediate" exports per span (serverless). */
  exportMode?: ExportMode;
  /** OTel sampling policy. Defaults to OTEL_TRACES_SAMPLER / OTEL_TRACES_SAMPLER_ARG. */
  sampler?: Sampler;
  /** Capture span inputs (args, prompts) by default. Default true. */
  captureInput?: boolean;
  /** Capture span outputs (results, completions) by default. Default true. */
  captureOutput?: boolean;
  /** Redaction hook run on captured content before stringify/truncate. */
  mask?: MaskFn;
  /** Truncation cap (chars) for content attributes; also set as the provider's attributeValueLengthLimit. Default 65536. */
  maxAttributeLength?: number;
  batch?: BatchOptions;
  /** Export filter. Default when registerGlobal: only spans from this SDK's instrumentation scope. */
  spanFilter?: (span: ReadableSpan) => boolean;
  /**
   * With registerGlobal: spans from other tracers that this returns a session id for are
   * parented under the session trace even when a parent span is active. Use it for frameworks
   * that start each turn inside their own engine span.
   */
  sessionRootOf?: SessionRootOf;
  /** Extra resource attributes merged onto service.name / deployment.environment.name. */
  resourceAttributes?: Record<string, AttributeValue>;
  /** SDK self-diagnostics console level. Default "warn"; "silent" disables. */
  logLevel?: SdkLogLevel;
  /** Injected fetch implementation. Defaults to `globalThis.fetch`. */
  fetch?: typeof fetch;
  /** Serverless extender (e.g. Cloudflare `ctx.waitUntil`). When provided, `flush()` does not await exports. */
  waitUntil?: (p: Promise<unknown>) => void;
  /** Receives any error raised while emitting telemetry; the SDK never throws into user code. */
  onError?: (cause: Error) => void;
  /** Header value for a custom telemetry.dev integration. */
  sdkName?: string;
}

export interface ResolvedConfig {
  apiKey: string | undefined;
  baseUrl: string;
  environment: string;
  serviceName: string;
  enabled: boolean;
  sessionMode: "explicit" | "process";
  processSessionId?: string;
  registerGlobal: boolean;
  exportMode: ExportMode;
  sampler?: Sampler;
  captureInput: boolean;
  captureOutput: boolean;
  mask?: MaskFn;
  maxAttributeLength: number;
  batch: Required<BatchOptions>;
  spanFilter?: (span: ReadableSpan) => boolean;
  sessionRootOf?: SessionRootOf;
  resourceAttributes?: Record<string, AttributeValue>;
  logLevel: SdkLogLevel;
  fetchImpl: typeof fetch;
  waitUntil?: (p: Promise<unknown>) => void;
  onError?: (cause: Error) => void;
  sdkName: string;
}

export const DEFAULT_MAX_ATTRIBUTE_LENGTH = 65536;

export function resolveConfig(options: TelemetryOptions = {}): ResolvedConfig {
  const env = resolveEnv();

  if (
    options.sessionMode !== undefined &&
    options.sessionMode !== "explicit" &&
    options.sessionMode !== "process"
  ) {
    throw new TypeError(`Invalid sessionMode: ${String(options.sessionMode)}`);
  }

  const baseUrl = (options.baseUrl ?? env.TELEMETRY_DEV_BASE_URL ?? DEFAULT_BASE_URL).replace(
    /\/+$/,
    "",
  );

  return {
    apiKey: options.apiKey ?? env.TELEMETRY_DEV_API_KEY,
    baseUrl,
    environment: options.environment ?? env.TELEMETRY_DEV_ENVIRONMENT ?? "production",
    serviceName: options.serviceName ?? env.OTEL_SERVICE_NAME ?? "unknown_service",
    enabled: options.enabled ?? true,
    sessionMode: options.sessionMode ?? "explicit",
    registerGlobal: options.registerGlobal ?? false,
    exportMode: options.exportMode ?? "batched",
    sampler: options.sampler,
    captureInput: options.captureInput ?? true,
    captureOutput: options.captureOutput ?? true,
    mask: options.mask,
    maxAttributeLength: options.maxAttributeLength ?? DEFAULT_MAX_ATTRIBUTE_LENGTH,
    batch: { ...DEFAULT_BATCH, ...options.batch },
    spanFilter: options.spanFilter,
    sessionRootOf: options.sessionRootOf,
    resourceAttributes: options.resourceAttributes,
    logLevel: options.logLevel ?? "warn",
    fetchImpl: options.fetch ?? globalThis.fetch,
    waitUntil: options.waitUntil,
    onError: options.onError,
    sdkName: options.sdkName ?? "@telemetry-dev/sdk",
  };
}
