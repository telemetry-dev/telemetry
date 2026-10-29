import { DEFAULT_BASE_URL, resolveEnv } from "@telemetry-dev/otel";

import { currentClient } from "./client.ts";

/** Where the calls to the telemetry.dev API go, and with which key. */
export interface ApiOptions {
  /** API key. Falls back to the initialized client, then `TELEMETRY_DEV_API_KEY`. */
  apiKey?: string;
  /** Falls back to the initialized client, then `TELEMETRY_DEV_BASE_URL`, then `https://ingest.telemetry.dev`. */
  baseUrl?: string;
  /** Falls back to the initialized client's `fetch`, then `globalThis.fetch`. */
  fetch?: typeof fetch;
}

export interface Api {
  apiKey: string | undefined;
  baseUrl: string;
  sdkName: string;
  fetchImpl: typeof fetch;
  onError: ((cause: Error) => void) | undefined;
}

/** An error class with the shape of `PromptError`: a code, a message, and the HTTP status. */
export type ApiErrorType<E extends Error> = new (
  code: string,
  message: string,
  status?: number,
  options?: { cause?: unknown },
) => E;

interface RequestOptions {
  body?: unknown;
  query?: Record<string, string | undefined>;
  timeoutMs: number;
  /**
   * The server takes a repeat of this request at most once, so a retry after a timeout, a network
   * error, or a 5xx is safe. Other requests try again only after a 429 or a 503: the server did
   * not take the request then.
   */
  idempotent: boolean;
  /** Default 3. */
  attempts?: number;
}

type Attempt =
  | { ok: true; body: unknown }
  | { ok: false; error: Error; status?: number; retryAfterMs?: number };

/** A longer Retry-After waits only this long: the API counts requests by the minute. */
const MAX_RETRY_AFTER_MS = 60_000;
/** IMF-fixdate, the HTTP date that servers send: `Sun, 06 Nov 1994 08:49:37 GMT`. */
const HTTP_DATE = /^[A-Z][a-z]{2}, \d{2} [A-Z][a-z]{2} \d{4} \d{2}:\d{2}:\d{2} GMT$/;

/** The key, base URL, and fetch of a call: its options, then the initialized client, then the env. */
export function resolveApi(options: ApiOptions): Api {
  const config = currentClient().core?.config;
  const env = resolveEnv();

  return {
    apiKey: options.apiKey ?? config?.apiKey ?? env.TELEMETRY_DEV_API_KEY,
    // An empty value counts as unset, as in the Python SDK.
    baseUrl: (
      options.baseUrl ||
      config?.baseUrl ||
      env.TELEMETRY_DEV_BASE_URL ||
      DEFAULT_BASE_URL
    ).replace(/\/+$/, ""),
    sdkName: config?.sdkName ?? "@telemetry-dev/sdk",
    fetchImpl: options.fetch ?? config?.fetchImpl ?? globalThis.fetch,
    onError: config?.onError,
  };
}

/**
 * Sends one JSON request to the telemetry.dev API and returns the JSON of the response. A response
 * that is not 2xx fails with the `{ error: { code, message } }` of its body, or `http_<status>`.
 * A failure that `options.idempotent` allows tries again, after the Retry-After of the response
 * or after 1 and then 2 seconds.
 */
export async function request<E extends Error>(
  api: Api & { apiKey: string },
  ErrorType: ApiErrorType<E>,
  method: "GET" | "POST",
  path: string,
  options: RequestOptions,
): Promise<unknown> {
  const url = new URL(`${api.baseUrl}${path}`);

  for (const [key, value] of Object.entries(options.query ?? {})) {
    if (value !== undefined) url.searchParams.set(key, value);
  }

  // Outside send(), so that a value JSON cannot hold fails as itself, not as a network error.
  const body = options.body === undefined ? undefined : JSON.stringify(options.body);
  const attempts = options.attempts ?? 3;

  for (let attempt = 1; ; attempt++) {
    const result = await send(api, ErrorType, method, url, body, options.timeoutMs);

    if (result.ok) return result.body;

    if (attempt >= attempts || !retryable(result.status, options.idempotent)) throw result.error;

    await sleep(Math.min(result.retryAfterMs ?? 1000 * attempt, MAX_RETRY_AFTER_MS));
  }
}

/** The wait that the Retry-After of a 429 or a 503 asks for, in seconds or as an HTTP date. */
export function retryAfterMs(res: Response): number | undefined {
  if (res.status !== 429 && res.status !== 503) return undefined;

  const value = res.headers.get("retry-after")?.trim();

  if (!value) return undefined;

  let ms: number;

  if (/^\d+$/.test(value)) ms = Number(value) * 1000;
  else if (HTTP_DATE.test(value)) ms = Date.parse(value) - Date.now();
  else return undefined;

  return ms > 0 ? ms : undefined;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function send<E extends Error>(
  api: Api & { apiKey: string },
  ErrorType: ApiErrorType<E>,
  method: "GET" | "POST",
  url: URL,
  body: string | undefined,
  timeoutMs: number,
): Promise<Attempt> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await api.fetchImpl(url, {
      method,
      headers: {
        authorization: `Bearer ${api.apiKey}`,
        "x-telemetry-dev-sdk": api.sdkName,
        ...(body === undefined ? {} : { "content-type": "application/json" }),
      },
      body,
      signal: controller.signal,
      // A redirect to another host must not get the API key, so a 3xx fails below like any status
      // that is not OK. Cloudflare Workers reject redirect: "error".
      redirect: "manual",
    });

    const json: unknown = await res.json().catch(() => undefined);

    if (res.ok) return { ok: true, body: json };

    const error = isRecord(json) && isRecord(json.error) ? json.error : {};

    return {
      ok: false,
      status: res.status,
      retryAfterMs: retryAfterMs(res),
      error: new ErrorType(
        typeof error.code === "string" ? error.code : `http_${res.status}`,
        typeof error.message === "string" ? error.message : `telemetry.dev returned ${res.status}`,
        res.status,
      ),
    };
  } catch (error) {
    // The timeout can also cut the body, which then reads as an invalid response.
    if (controller.signal.aborted) {
      return {
        ok: false,
        error: new ErrorType("timeout", `${method} ${url.pathname} took more than ${timeoutMs} ms`),
      };
    }

    return {
      ok: false,
      error: new ErrorType("network_error", "Could not reach telemetry.dev", undefined, {
        cause: error,
      }),
    };
  } finally {
    clearTimeout(timer);
  }
}

function retryable(status: number | undefined, idempotent: boolean): boolean {
  if (status === 429 || status === 503) return true;

  return idempotent && (status === undefined || status >= 500);
}
