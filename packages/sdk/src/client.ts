import { context as apiContext, propagation, trace, type Tracer } from "@opentelemetry/api";
import type { Logger } from "@opentelemetry/api-logs";
import { W3CTraceContextPropagator } from "@opentelemetry/core";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  BatchLogRecordProcessor,
  LoggerProvider,
  type LogRecordExporter,
  SimpleLogRecordProcessor,
} from "@opentelemetry/sdk-logs";
import type { PushMetricExporter } from "@opentelemetry/sdk-metrics";
import {
  BasicTracerProvider,
  type ReadableSpan,
  type SpanExporter,
} from "@opentelemetry/sdk-trace-base";

import {
  als,
  AlsContextManager,
  BATCHED_METRIC_INTERVAL_MS,
  createLogExporter,
  createMetricExporter,
  createMetricsPipeline,
  createTraceExporter,
  diag,
  DORMANT_INTERVAL_MS,
  otlpHeaders,
  reportError,
  SCOPE_NAME,
  SCOPE_VERSION,
  sessionRootTracerProvider,
  sessionSampler,
  setLogLevel,
  StampingSpanProcessor,
  type Transport,
} from "@telemetry-dev/otel";

import { type ResolvedConfig, resolveConfig, type TelemetryOptions } from "./config.ts";

/** Test seam: inject in-memory exporters instead of the OTLP fetch transport. */
export interface ClientOverrides {
  spanExporter?: SpanExporter;
  logRecordExporter?: LogRecordExporter;
  metricExporter?: PushMetricExporter;
}

export interface TelemetryClient {
  readonly enabled: boolean;
  /** Never rejects; hands the export off to waitUntil when configured. */
  flush(): Promise<void>;
  /** Flush + teardown + unregister globals; the client becomes a no-op afterwards. */
  shutdown(): Promise<void>;
}

export interface ClientCore {
  config: ResolvedConfig;
  tracer: Tracer;
  ensureLogger(): Logger | undefined;
}

export interface ClientHandle extends TelemetryClient {
  core?: ClientCore;
}

const NOOP_CLIENT: ClientHandle = {
  enabled: false,
  flush: () => Promise.resolve(),
  shutdown: () => Promise.resolve(),
};

let activeClient: ClientHandle | undefined;

export function currentClient(): ClientHandle {
  return activeClient ?? NOOP_CLIENT;
}

export function flush(): Promise<void> {
  return currentClient().flush();
}

export function shutdown(): Promise<void> {
  return currentClient().shutdown();
}

export function init(options?: TelemetryOptions, overrides?: ClientOverrides): TelemetryClient {
  try {
    return initInner(options, overrides);
  } catch (error) {
    reportError(options?.onError, error instanceof Error ? error : new Error(String(error)));
    activeClient = NOOP_CLIENT;

    return NOOP_CLIENT;
  }
}

function initInner(options?: TelemetryOptions, overrides?: ClientOverrides): TelemetryClient {
  const config = resolveConfig(options);
  setLogLevel(config.logLevel);

  if (activeClient && activeClient !== NOOP_CLIENT) {
    diag.warn("init() called again; replacing the previous client");
    // shutdown() synchronously releases global registrations before this init claims them.
    void activeClient.shutdown().catch((error: Error) => reportError(config.onError, error));
  }

  const enabled = config.enabled && Boolean(config.apiKey ?? overrides?.spanExporter);

  if (!enabled) {
    if (config.enabled && !config.apiKey) {
      diag.debug("no api key (apiKey option or TELEMETRY_DEV_API_KEY); telemetry disabled");
    }

    activeClient = NOOP_CLIENT;

    return NOOP_CLIENT;
  }

  if (config.sessionMode === "process") config.processSessionId = globalThis.crypto.randomUUID();

  const resource = resourceFromAttributes({
    "service.name": config.serviceName,
    "deployment.environment.name": config.environment,
    ...config.resourceAttributes,
  });

  const transport: Transport = { fetchImpl: config.fetchImpl, onError: config.onError };
  const headers = config.apiKey ? otlpHeaders(config.apiKey, config.sdkName) : undefined;

  const spanExporter =
    overrides?.spanExporter ??
    createTraceExporter({ url: `${config.baseUrl}/v1/traces`, headers: headers! }, transport);

  const metricExporter =
    overrides?.metricExporter ??
    (headers
      ? createMetricExporter({ url: `${config.baseUrl}/v1/metrics`, headers }, transport)
      : undefined);

  const metrics = metricExporter
    ? createMetricsPipeline({
        resource,
        exporter: metricExporter,
        exportIntervalMillis:
          config.exportMode === "batched" ? BATCHED_METRIC_INTERVAL_MS : DORMANT_INTERVAL_MS,
      })
    : undefined;

  const spanFilter =
    config.spanFilter ??
    (config.registerGlobal
      ? (span: ReadableSpan) => span.instrumentationScope.name === SCOPE_NAME
      : undefined);

  const processor = new StampingSpanProcessor({
    exporter: spanExporter,
    exportMode: config.exportMode,
    batch: config.batch,
    spanFilter,
    recordMetrics: metrics ? (span) => metrics.record(span) : undefined,
    onError: config.onError,
  });

  const provider = new BasicTracerProvider({
    resource,
    sampler: sessionSampler(config.sampler),
    spanProcessors: [processor],
    spanLimits: { attributeValueLengthLimit: config.maxAttributeLength },
  });

  const tracer = provider.getTracer(SCOPE_NAME, SCOPE_VERSION);

  let registeredTrace = false;
  let registeredContext = false;
  let registeredPropagation = false;

  if (config.registerGlobal) {
    registeredTrace = trace.setGlobalTracerProvider(
      config.sessionRootOf
        ? sessionRootTracerProvider(provider, config.apiKey, config.sessionRootOf, config.onError)
        : provider,
    );

    if (als) {
      registeredContext = apiContext.setGlobalContextManager(new AlsContextManager(als));
    }

    registeredPropagation = propagation.setGlobalPropagator(new W3CTraceContextPropagator());

    if (!registeredTrace) {
      diag.warn(
        "a global TracerProvider is already registered; attach the TelemetrySpanProcessor from '@telemetry-dev/otel' to your own provider instead",
      );
    }
  }

  const logExporter =
    overrides?.logRecordExporter ??
    (headers
      ? createLogExporter({ url: `${config.baseUrl}/v1/logs`, headers }, transport)
      : undefined);

  // Lazy: the log pipeline only exists once log() is first called.
  let logs: { logger: Logger; forceFlush(): Promise<void>; shutdown(): Promise<void> } | undefined;

  const ensureLogger = (): Logger | undefined => {
    if (!logExporter) return undefined;

    if (!logs) {
      const logProcessor =
        config.exportMode === "immediate"
          ? new SimpleLogRecordProcessor(logExporter)
          : new BatchLogRecordProcessor(logExporter);

      const loggerProvider = new LoggerProvider({
        resource,
        logRecordLimits: { attributeValueLengthLimit: config.maxAttributeLength },
        processors: [logProcessor],
      });

      logs = {
        logger: loggerProvider.getLogger(SCOPE_NAME, SCOPE_VERSION),
        forceFlush: () => loggerProvider.forceFlush(),
        shutdown: () => loggerProvider.shutdown(),
      };
    }

    return logs.logger;
  };

  let torn = false;

  const collect = (op: "forceFlush" | "shutdown"): Promise<void> => {
    const parts: Promise<void>[] = [provider[op]()];

    if (logs) parts.push(logs[op]());

    if (metrics) parts.push(metrics[op]());

    return Promise.all(parts)
      .then(() => undefined)
      .catch((error: Error) => reportError(config.onError, error));
  };

  const handoff = (p: Promise<void>): Promise<void> => {
    if (config.waitUntil) {
      config.waitUntil(p);

      return Promise.resolve();
    }

    return p;
  };

  const handle: ClientHandle = {
    get enabled() {
      return !torn;
    },
    flush: () => (torn ? Promise.resolve() : handoff(collect("forceFlush"))),
    shutdown: () => {
      if (torn) return Promise.resolve();
      torn = true;

      if (registeredTrace) trace.disable();

      if (registeredContext) apiContext.disable();

      if (registeredPropagation) propagation.disable();
      handle.core = undefined;

      return handoff(collect("shutdown"));
    },
  };

  handle.core = { config, tracer, ensureLogger };
  activeClient = handle;

  return handle;
}
