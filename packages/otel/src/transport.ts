import { type ExportResult, ExportResultCode } from "@opentelemetry/core";
import {
  ProtobufLogsSerializer,
  ProtobufMetricsSerializer,
  ProtobufTraceSerializer,
} from "@opentelemetry/otlp-transformer";
import type { LogRecordExporter, ReadableLogRecord } from "@opentelemetry/sdk-logs";
import {
  AggregationTemporality,
  type PushMetricExporter,
  type ResourceMetrics,
} from "@opentelemetry/sdk-metrics";
import type { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";

import { DEFAULT_BATCH } from "./config.ts";
import { reportError } from "./debug.ts";

export interface Transport {
  fetchImpl: typeof fetch;
  onError?: (error: Error) => void;
  exportTimeoutMillis?: number;
}

export interface OtlpTarget {
  url: string;
  headers: Record<string, string>;
}

const RETRY_DELAYS_MS = [100, 500] as const;
const MAX_RETRY_AFTER_MS = 2_000;
const MAX_TIMER_DELAY_MS = 2_147_483_647;

const RETRYABLE_STATUSES = new Set([429, 502, 503, 504]);

const isRetryableStatus = (status: number) => RETRYABLE_STATUSES.has(status);

const HTTP_MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];
const HTTP_WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const HTTP_WEEKDAYS_LONG = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];

const httpDateTimestamp = (
  weekday: string,
  dayText: string,
  monthText: string,
  yearText: string,
  hourText: string,
  minuteText: string,
  secondText: string,
  weekdays: string[],
): number | undefined => {
  const day = Number(dayText);
  const month = HTTP_MONTHS.indexOf(monthText);
  const year = Number(yearText);
  const hour = Number(hourText);
  const minute = Number(minuteText);
  const second = Number(secondText);
  if (hour > 23 || minute > 59 || second > 59) return undefined;
  const date = new Date(0);
  date.setUTCFullYear(year, month, day);
  date.setUTCHours(hour, minute, second, 0);
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month ||
    date.getUTCDate() !== day ||
    weekdays[date.getUTCDay()] !== weekday
  ) {
    return undefined;
  }
  return date.getTime();
};

const parseHttpDate = (value: string): number | undefined => {
  const imf =
    /^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d{2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4}) (\d{2}):(\d{2}):(\d{2}) GMT$/.exec(
      value,
    );
  if (imf)
    return httpDateTimestamp(
      ...(imf.slice(1) as [string, string, string, string, string, string, string]),
      HTTP_WEEKDAYS,
    );

  const rfc850 =
    /^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), (\d{2})-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2}) (\d{2}):(\d{2}):(\d{2}) GMT$/.exec(
      value,
    );
  if (rfc850) {
    const fields = rfc850.slice(1) as [string, string, string, string, string, string, string];
    const currentYear = new Date().getUTCFullYear();
    let year = Math.floor(currentYear / 100) * 100 + Number(fields[3]);
    if (year > currentYear + 50) year -= 100;
    fields[3] = String(year);
    return httpDateTimestamp(...fields, HTTP_WEEKDAYS_LONG);
  }

  const asctime =
    /^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) ( [1-9]|[12]\d|3[01]) (\d{2}):(\d{2}):(\d{2}) (\d{4})$/.exec(
      value,
    );
  if (!asctime) return undefined;
  const [, weekday, month, day, hour, minute, second, year] = asctime;
  return httpDateTimestamp(
    weekday!,
    day!.trim(),
    month!,
    year!,
    hour!,
    minute!,
    second!,
    HTTP_WEEKDAYS,
  );
};

const retryAfterMs = (res: Response): number | undefined => {
  const header = res.headers.get("retry-after")?.trim();
  if (header === undefined) return undefined;
  if (/^\d+$/.test(header)) return Number(header) * 1000;
  const at = parseHttpDate(header);
  return at === undefined ? undefined : Math.max(0, at - Date.now());
};

const delay = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason ?? new Error("telemetry.dev export aborted"));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    if (signal?.aborted) onAbort();
    else signal?.addEventListener("abort", onAbort, { once: true });
  });

const cancelBody = async (res: Response) => {
  try {
    await res.body?.cancel();
  } catch {}
};

export const postOtlp = async ({
  fetchImpl,
  url,
  headers,
  body,
  signal,
}: {
  fetchImpl: typeof fetch;
  url: string;
  headers: Record<string, string>;
  body: Uint8Array;
  signal?: AbortSignal;
}) => {
  for (let attempt = 0; ; attempt += 1) {
    let retryAfter: number | undefined;
    try {
      const res = await fetchImpl(url, {
        method: "POST",
        headers,
        body: body as RequestInit["body"],
        signal,
      });
      retryAfter = retryAfterMs(res);
      if (
        res.ok ||
        !isRetryableStatus(res.status) ||
        attempt === RETRY_DELAYS_MS.length ||
        (retryAfter !== undefined && retryAfter > MAX_RETRY_AFTER_MS)
      ) {
        await cancelBody(res);
        return res;
      }
      await cancelBody(res);
    } catch (error) {
      if (attempt === RETRY_DELAYS_MS.length) throw error;
    }

    await delay(retryAfter ?? RETRY_DELAYS_MS[attempt]!, signal);
  }
};

const GZIP_THRESHOLD_BYTES = 1024;

export async function maybeGzip(
  body: Uint8Array,
): Promise<{ body: Uint8Array; contentEncoding?: "gzip" }> {
  if (body.byteLength <= GZIP_THRESHOLD_BYTES || globalThis.CompressionStream === undefined) {
    return { body };
  }
  try {
    const stream = new Blob([body as Uint8Array<ArrayBuffer>])
      .stream()
      .pipeThrough(new CompressionStream("gzip"));
    const compressed = new Uint8Array(await new Response(stream).arrayBuffer());
    return { body: compressed, contentEncoding: "gzip" };
  } catch {
    // Fail-open: ship uncompressed rather than lose the batch.
    return { body };
  }
}

async function postSerialized(
  body: Uint8Array,
  target: OtlpTarget,
  transport: Transport,
  label: string,
  post: typeof postOtlp = postOtlp,
  signal?: AbortSignal,
): Promise<void> {
  const { body: finalBody, contentEncoding } = await maybeGzip(body);
  const headers = contentEncoding
    ? { ...target.headers, "content-encoding": contentEncoding }
    : target.headers;
  const res = await post({
    fetchImpl: transport.fetchImpl,
    url: target.url,
    headers,
    body: finalBody,
    signal,
  });
  if (!res.ok) {
    const retryAfter = retryAfterMs(res);
    const retryAfterHeader = res.headers.get("retry-after")?.trim();
    const retryDetail =
      retryAfter !== undefined && retryAfter > MAX_RETRY_AFTER_MS && retryAfterHeader !== undefined
        ? `; retry-after ${retryAfterHeader} exceeds ${MAX_RETRY_AFTER_MS / 1000}s retry cap`
        : "";
    throw new Error(`telemetry.dev ${label} ingest failed: ${res.status}${retryDetail}`);
  }
}

const createExportLifecycle = (transport: Transport) => {
  const inFlight = new Set<Promise<void>>();
  let shutDown = false;
  const configuredTimeoutMillis =
    transport.exportTimeoutMillis ?? DEFAULT_BATCH.exportTimeoutMillis;
  const timeoutMillis =
    Number.isFinite(configuredTimeoutMillis) &&
    configuredTimeoutMillis >= 0 &&
    configuredTimeoutMillis <= MAX_TIMER_DELAY_MS
      ? configuredTimeoutMillis
      : DEFAULT_BATCH.exportTimeoutMillis;

  const callback = (resultCallback: (result: ExportResult) => void, result: ExportResult) => {
    try {
      resultCallback(result);
    } catch (error) {
      reportError(transport.onError, error);
    }
  };

  const exportBatch = (
    send: (signal: AbortSignal) => Promise<void>,
    resultCallback: (result: ExportResult) => void,
  ) => {
    if (shutDown) {
      const error = new Error("telemetry.dev exporter is shut down");
      callback(resultCallback, { code: ExportResultCode.FAILED, error });
      return;
    }
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new Error(`telemetry.dev export timed out after ${timeoutMillis}ms`));
      }, timeoutMillis);
    });
    let tracked: Promise<void>;
    tracked = Promise.race([Promise.resolve().then(() => send(controller.signal)), timeout])
      .then(
        () => callback(resultCallback, { code: ExportResultCode.SUCCESS }),
        (cause: unknown) => {
          const error = cause instanceof Error ? cause : new Error(String(cause));
          reportError(transport.onError, error);
          callback(resultCallback, { code: ExportResultCode.FAILED, error });
        },
      )
      .finally(() => {
        clearTimeout(timer);
        inFlight.delete(tracked);
      });
    inFlight.add(tracked);
  };

  const forceFlush = () => Promise.all(inFlight).then(() => undefined);
  const shutdown = () => {
    shutDown = true;
    return forceFlush();
  };

  return { exportBatch, forceFlush, shutdown };
};

// Stay under the ingest's 4 MiB raw body limit with headroom; oversized batches are halved.
const MAX_BODY_BYTES = 3_500_000;

interface Serializer<T> {
  serializeRequest(items: T[]): Uint8Array | undefined;
}

function createOtlpBatchSender<T>(
  serializer: Serializer<T>,
  target: OtlpTarget,
  transport: Transport,
  label: string,
): (items: T[], signal: AbortSignal) => Promise<void> {
  const send = async (items: T[], signal: AbortSignal): Promise<void> => {
    const body = serializer.serializeRequest(items);
    if (!body || body.byteLength === 0) return;
    if (body.byteLength > MAX_BODY_BYTES) {
      if (items.length > 1) {
        const mid = Math.ceil(items.length / 2);
        await send(items.slice(0, mid), signal);
        await send(items.slice(mid), signal);
        return;
      }
      // A single record beyond the limit can never be accepted; drop it instead of wedging the batch.
      reportError(
        transport.onError,
        new Error(`telemetry.dev: ${label} record exceeds max export size, dropped`),
      );
      return;
    }
    await postSerialized(body, target, transport, label, postOtlp, signal);
  };
  return send;
}

export function createTraceExporter(target: OtlpTarget, transport: Transport): SpanExporter {
  const send = createOtlpBatchSender<ReadableSpan>(
    ProtobufTraceSerializer,
    target,
    transport,
    "trace",
  );
  const lifecycle = createExportLifecycle(transport);
  return {
    export(spans, resultCallback) {
      lifecycle.exportBatch((signal) => send(spans, signal), resultCallback);
    },
    forceFlush: lifecycle.forceFlush,
    shutdown: lifecycle.shutdown,
  };
}

export function createLogExporter(target: OtlpTarget, transport: Transport): LogRecordExporter {
  const send = createOtlpBatchSender<ReadableLogRecord>(
    ProtobufLogsSerializer,
    target,
    transport,
    "log",
  );
  const lifecycle = createExportLifecycle(transport);
  return {
    export(logs, resultCallback) {
      lifecycle.exportBatch((signal) => send(logs, signal), resultCallback);
    },
    forceFlush: lifecycle.forceFlush,
    shutdown: lifecycle.shutdown,
  };
}

export function createMetricExporter(target: OtlpTarget, transport: Transport): PushMetricExporter {
  const lifecycle = createExportLifecycle(transport);
  return {
    export(resourceMetrics: ResourceMetrics, resultCallback) {
      lifecycle.exportBatch(async (signal) => {
        const hasData = resourceMetrics.scopeMetrics.some((scope) =>
          scope.metrics.some((metric) => metric.dataPoints.length > 0),
        );
        const body = hasData
          ? ProtobufMetricsSerializer.serializeRequest(resourceMetrics)
          : undefined;
        if (body !== undefined && body.byteLength > 0) {
          await postSerialized(body, target, transport, "metric", postOtlp, signal);
        }
      }, resultCallback);
    },
    selectAggregationTemporality: () => AggregationTemporality.DELTA,
    forceFlush: lifecycle.forceFlush,
    shutdown: lifecycle.shutdown,
  };
}

export function otlpHeaders(apiKey: string, sdkName = "@telemetry-dev/otel") {
  return {
    "content-type": "application/x-protobuf",
    authorization: `Bearer ${apiKey}`,
    "x-telemetry-dev-sdk": sdkName,
  };
}
