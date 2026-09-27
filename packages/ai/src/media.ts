import { type Attributes, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import {
  createGenerationEmitter,
  type GenerationEmitterOverrides,
  omitUndefined,
} from "@telemetry-dev/otel";

import { resolveConfig, type TelemetryDevOptions } from "./config.ts";
import { providerLabel, unknownErrorMessage } from "./shared.ts";

export type MediaWrapperOverrides = GenerationEmitterOverrides;

type Callable = (...args: never[]) => unknown;

type MediaKind = "image" | "speech" | "text" | "video";

const finite = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

const text = (value: unknown): string | undefined =>
  typeof value === "string" && value.length > 0 ? value : undefined;

function dataProperty(value: unknown, key: PropertyKey): unknown {
  if ((typeof value !== "object" && typeof value !== "function") || value === null)
    return undefined;

  try {
    let current: object | null = value;

    while (current) {
      const descriptor = Object.getOwnPropertyDescriptor(current, key);

      if (descriptor) return "value" in descriptor ? descriptor.value : undefined;
      current = Object.getPrototypeOf(current);
    }
  } catch {
    return undefined;
  }

  return undefined;
}

function modelAttributes(args: readonly unknown[]): Attributes {
  const input = args[0];
  const modelValue = dataProperty(input, "model");

  const provider =
    text(dataProperty(modelValue, "provider")) ?? text(dataProperty(input, "provider"));

  const modelId =
    text(modelValue) ??
    text(dataProperty(modelValue, "modelId")) ??
    text(dataProperty(modelValue, "model")) ??
    text(dataProperty(input, "modelId"));

  return omitUndefined({
    "gen_ai.provider.name": provider ? providerLabel(provider) : undefined,
    "gen_ai.request.model": modelId,
  });
}

function resultAttributes(kind: MediaKind, args: readonly unknown[], value: unknown): Attributes {
  const input = args[0];
  const result = value;
  const usage = dataProperty(result, "usage");
  const attrs: Attributes = {};

  if (kind === "image") {
    attrs["gen_ai.usage.input_tokens"] = finite(dataProperty(usage, "inputTokens"));
    attrs["gen_ai.usage.output_tokens"] = finite(dataProperty(usage, "outputTokens"));
    attrs["gen_ai.usage.image.output_tokens"] = finite(dataProperty(usage, "outputTokens"));
    attrs["gen_ai.usage.total_tokens"] = finite(dataProperty(usage, "totalTokens"));
    const images = dataProperty(result, "images");
    attrs["td.ai.output.image_count"] = Array.isArray(images)
      ? images.length
      : dataProperty(result, "image") == null
        ? undefined
        : 1;
  } else if (kind === "video") {
    const videos = dataProperty(result, "videos");
    attrs["td.ai.output.video_count"] = Array.isArray(videos)
      ? videos.length
      : dataProperty(result, "video") == null
        ? undefined
        : 1;
  } else if (kind === "speech") {
    attrs["td.ai.speech.input_character_count"] = text(dataProperty(input, "text"))?.length;
  } else {
    attrs["td.ai.transcription.duration_seconds"] =
      finite(dataProperty(result, "durationInSeconds")) ?? finite(dataProperty(result, "duration"));
  }

  return omitUndefined(attrs);
}

function wrapMedia<F extends Callable>(
  fn: F,
  kind: MediaKind,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F {
  let config: ReturnType<typeof resolveConfig>;

  try {
    config = resolveConfig(options);
  } catch (error) {
    reportTelemetryError(dataProperty(options, "onError") as TelemetryDevOptions["onError"], error);

    return fn;
  }

  if (!config.apiKey) return fn;

  let emitter: ReturnType<typeof createGenerationEmitter>;

  try {
    emitter = createGenerationEmitter({ ...config, sdkName: "@telemetry-dev/ai-sdk" }, overrides);
  } catch (error) {
    reportTelemetryError(config.onError, error);

    return fn;
  }

  return ((...args: Parameters<F>) => {
    let span:
      | ReturnType<ReturnType<typeof createGenerationEmitter>["tracer"]["startSpan"]>
      | undefined;

    let startedAt: number | undefined;
    let attrs: Attributes | undefined;

    try {
      startedAt = performance.now();
      attrs = {
        "gen_ai.operation.name": "generate_content",
        "gen_ai.output.type": kind,
        ...modelAttributes(args),
      };
      span = emitter.tracer.startSpan(`generate_content ${kind}`, {
        kind: SpanKind.CLIENT,
        attributes: attrs,
      });
    } catch (error) {
      reportTelemetryError(config.onError, error);
    }

    const finish = (value: unknown, failed: boolean, error?: unknown) => {
      if (!span || !attrs || startedAt === undefined) return;

      const attempt = (operation: () => void) => {
        try {
          operation();
        } catch (cause) {
          reportTelemetryError(config.onError, cause);
        }
      };

      if (failed) {
        attempt(() =>
          span.setStatus({ code: SpanStatusCode.ERROR, message: unknownErrorMessage(error) }),
        );
        attempt(() =>
          span.recordException(error instanceof Error ? error : unknownErrorMessage(error)),
        );
      } else {
        let resultAttrs: Attributes = {};
        attempt(() => {
          resultAttrs = resultAttributes(kind, args, value);
          span.setAttributes(resultAttrs);
        });
        const inputTokens = resultAttrs["gen_ai.usage.input_tokens"];
        const outputTokens = resultAttrs["gen_ai.usage.output_tokens"];

        if (typeof inputTokens === "number")
          attempt(() => emitter.recordTokens("input", inputTokens, attrs));

        if (typeof outputTokens === "number")
          attempt(() => emitter.recordTokens("output", outputTokens, attrs));
      }

      attempt(() => span.end());
      attempt(() => emitter.recordDuration((performance.now() - startedAt) / 1000, attrs));
      attempt(() => {
        void emitter.flush([span]).catch((cause) => reportTelemetryError(config.onError, cause));
      });
    };

    let value: ReturnType<F>;

    try {
      value = fn(...args) as ReturnType<F>;
    } catch (error) {
      finish(undefined, true, error);
      throw error;
    }

    if (typeof dataProperty(value, "then") === "function") {
      return Promise.resolve(value).then(
        (result) => {
          finish(result, false);

          return result;
        },
        (error) => {
          finish(undefined, true, error);
          throw error;
        },
      ) as ReturnType<F>;
    }

    finish(value, false);

    return value;
  }) as F;
}

function reportTelemetryError(onError: TelemetryDevOptions["onError"], error: unknown): void {
  try {
    onError?.(error instanceof Error ? error : String(error));
  } catch {}
}

export const wrapGenerateImage = <F extends Callable>(
  fn: F,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F => wrapMedia(fn, "image", options, overrides);

export const wrapGenerateSpeech = <F extends Callable>(
  fn: F,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F => wrapMedia(fn, "speech", options, overrides);

export const wrapTranscribe = <F extends Callable>(
  fn: F,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F => wrapMedia(fn, "text", options, overrides);

export const wrapGenerateVideo = <F extends Callable>(
  fn: F,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F => wrapMedia(fn, "video", options, overrides);
