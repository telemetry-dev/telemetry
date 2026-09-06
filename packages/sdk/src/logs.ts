import type { AttributeValue } from "@opentelemetry/api";
import { SeverityNumber } from "@opentelemetry/api-logs";

import {
  activeContext,
  PROPAGATED_KEY,
  propagatedFromContext,
  reportError,
} from "@telemetry-dev/otel";

import { prepareCaptureValue, truncate } from "./capture.ts";
import { currentClient } from "./client.ts";
import type { LogLevel } from "./config.ts";

// Severity numbers match the ingest's severityFromNumber buckets exactly.
const SEVERITIES = {
  debug: { number: SeverityNumber.DEBUG, text: "DEBUG" },
  info: { number: SeverityNumber.INFO, text: "INFO" },
  warn: { number: SeverityNumber.WARN, text: "WARN" },
  error: { number: SeverityNumber.ERROR, text: "ERROR" },
} satisfies Record<LogLevel, { number: SeverityNumber; text: string }>;

export interface LogOptions {
  /** Default "info". */
  level?: LogLevel;
  attributes?: Record<string, Parameters<import("./config.ts").MaskFn>[0]>;
  eventName?: string;
  timestamp?: Date | number;
}

/**
 * Emit a log record to /v1/logs, correlated to the active span (when any) and stamped with the
 * propagated correlation attributes. Works inside and outside spans.
 */
export function log(message: string, options?: LogOptions): void {
  const core = currentClient().core;
  if (!core) return;
  try {
    const logger = core.ensureLogger();
    if (!logger) return;
    const cfg = core.config;
    const captureCfg = {
      mask: cfg.mask,
      maxAttributeLength: cfg.maxAttributeLength,
      onError: cfg.onError,
    };
    const severity = SEVERITIES[options?.level ?? "info"];
    const active = activeContext();
    const explicitSession = options?.attributes?.["gen_ai.conversation.id"];
    const propagatedSession = propagatedFromContext(active)?.["gen_ai.conversation.id"];
    const sessionId =
      explicitSession !== undefined
        ? undefined
        : typeof propagatedSession === "string"
          ? propagatedSession
          : cfg.processSessionId;
    const ctx =
      sessionId === undefined
        ? active
        : active.setValue(PROPAGATED_KEY, {
            ...propagatedFromContext(active),
            "gen_ai.conversation.id": sessionId,
          });

    const attributes: Record<string, AttributeValue> = {};
    for (const [key, value] of Object.entries(options?.attributes ?? {})) {
      if (value === undefined) continue;
      if (typeof value === "number" || typeof value === "boolean") {
        attributes[key] = value;
        continue;
      }
      const prepared = prepareCaptureValue(key, value, captureCfg);
      if (prepared !== undefined) attributes[key] = prepared;
    }
    const propagated = propagatedFromContext(ctx);
    if (propagated) {
      for (const [key, value] of Object.entries(propagated)) {
        if (value !== undefined) {
          attributes[key] =
            value.constructor === String ? truncate(String(value), cfg.maxAttributeLength) : value;
        }
      }
    }

    logger.emit({
      body: prepareCaptureValue("log.message", message, captureCfg) ?? "",
      severityNumber: severity.number,
      severityText: severity.text,
      eventName: options?.eventName,
      timestamp: options?.timestamp,
      context: ctx,
      attributes,
    });
  } catch (error) {
    reportError(core.config.onError, error instanceof Error ? error : new Error(String(error)));
  }
}
