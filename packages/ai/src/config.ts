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
    onError: options.onError,
  };
}
