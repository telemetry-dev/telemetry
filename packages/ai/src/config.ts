import type { Sampler } from "@opentelemetry/sdk-trace-base";

export type TelemetryError =
  | Error
  | string
  | number
  | boolean
  | bigint
  | null
  | undefined
  | { [key: string]: TelemetryError }
  | TelemetryError[];

export interface TelemetryDevOptions {
  /** telemetry.dev ingest key (`td_live_…`). Falls back to `TELEMETRY_DEV_API_KEY`. When absent the integration is a no-op. */
  apiKey?: string;
  /** Ingest base URL. Falls back to `TELEMETRY_DEV_BASE_URL`, then `https://ingest.telemetry.dev`. */
  baseUrl?: string;
  /** Deployment environment label. Falls back to `TELEMETRY_DEV_ENVIRONMENT`, then `production`. */
  environment?: string;
  /** Service name attached to every trace. Falls back to `OTEL_SERVICE_NAME`, then `unknown_service`. */
  serviceName?: string;
  /** OTel sampling policy. Defaults to OTEL_TRACES_SAMPLER / OTEL_TRACES_SAMPLER_ARG. */
  sampler?: Sampler;
  /** Injected fetch implementation. Defaults to `globalThis.fetch`. */
  fetch?: typeof fetch;
  /** Serverless extender (e.g. Cloudflare `ctx.waitUntil`). When provided, `onFinish` does not await the POST. */
  waitUntil?: (p: Promise<unknown>) => void;
  /**
   * Redaction hook run on each piece of captured content before it is JSON-stringified: input and
   * output messages, tool arguments and results, detailed error messages, and model warning text.
   * `context.key` is the attribute being written (`exception.message` for error messages,
   * `log.message` for warnings). Returning undefined omits the content; a throwing mask omits it
   * too (error messages fall back to a generic text) and reports through onError. Not applied to
   * runtimeContext metadata.
   */
  mask?: (value: unknown, context: { key: string }) => unknown;
  /** Receives any error raised while emitting telemetry; the integration never throws into the SDK. */
  onError?: (error: TelemetryError) => void;
}

export interface ResolvedConfig {
  apiKey: string | undefined;
  baseUrl: string;
  environment: string;
  serviceName: string;
  sampler?: Sampler;
  fetchImpl: typeof fetch;
  waitUntil?: (p: Promise<unknown>) => void;
  mask?: TelemetryDevOptions["mask"];
  onError?: (error: TelemetryError) => void;
}

const DEFAULT_BASE_URL = "https://ingest.telemetry.dev";

export function resolveConfig(options: TelemetryDevOptions = {}): ResolvedConfig {
  const env = globalThis.process?.env ?? {};

  const baseUrl = (options.baseUrl ?? env.TELEMETRY_DEV_BASE_URL ?? DEFAULT_BASE_URL).replace(
    /\/+$/,
    "",
  );

  return {
    apiKey: options.apiKey ?? env.TELEMETRY_DEV_API_KEY,
    baseUrl,
    environment: options.environment ?? env.TELEMETRY_DEV_ENVIRONMENT ?? "production",
    serviceName: options.serviceName ?? env.OTEL_SERVICE_NAME ?? "unknown_service",
    sampler: options.sampler,
    fetchImpl: options.fetch ?? globalThis.fetch,
    waitUntil: options.waitUntil,
    mask: options.mask,
    onError: options.onError,
  };
}
