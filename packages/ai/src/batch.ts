import { type Attributes, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { createGenerationEmitter, type GenerationEmitterOverrides } from "@telemetry-dev/otel";

import { resolveConfig, type TelemetryDevOptions } from "./config.ts";
import { unknownErrorMessage } from "./shared.ts";

type Callable = (...args: never[]) => unknown;

type BatchOperation = "batch.submit" | "batch.status" | "batch.cancel" | "batch.results";

const scalar = (value: unknown): string | number | boolean | undefined =>
  typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    ? value
    : undefined;

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

function resolvedConfig(options: TelemetryDevOptions | undefined) {
  try {
    return resolveConfig(options);
  } catch (error) {
    reportTelemetryError(dataProperty(options, "onError") as TelemetryDevOptions["onError"], error);

    return undefined;
  }
}

function wrapBatch<F extends Callable>(
  fn: F,
  operation: BatchOperation,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F {
  const config = resolvedConfig(options);

  if (!config) return fn;

  if (!config.apiKey) return fn;

  let emitter: ReturnType<typeof createGenerationEmitter>;

  try {
    emitter = createGenerationEmitter({ ...config, sdkName: "@telemetry-dev/ai-sdk" }, overrides);
  } catch (error) {
    reportTelemetryError(config.onError, error);

    return fn;
  }

  return ((...args: Parameters<F>) => {
    let batch: unknown;
    let attrs: Attributes;
    let span: ReturnType<ReturnType<typeof createGenerationEmitter>["tracer"]["startSpan"]>;
    let startedAt: number;

    try {
      startedAt = performance.now();
      const input = args[0];
      batch = dataProperty(input, "batch");
      const requests = dataProperty(input, "requests");
      attrs = {
        "gen_ai.operation.name": operation,
        "td.ai.batch.id": scalar(dataProperty(batch, "id")),
        "td.ai.batch.item_count": Array.isArray(requests) ? requests.length : undefined,
      };
      span = emitter.tracer.startSpan(operation, { kind: SpanKind.CLIENT, attributes: attrs });
    } catch (error) {
      reportTelemetryError(config.onError, error);

      return fn(...args) as ReturnType<F>;
    }

    const finish = (value: unknown, failed: boolean, error?: unknown) => {
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
        const result = value;
        attempt(() => {
          const batches = dataProperty(result, "batches");
          span.setAttributes({
            "td.ai.batch.id":
              scalar(dataProperty(result, "id")) ?? scalar(dataProperty(batch, "id")),
            "td.ai.batch.status": scalar(dataProperty(result, "status")),
            "td.ai.batch.item_count": Array.isArray(batches)
              ? batches.length
              : attrs["td.ai.batch.item_count"],
          });
        });
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

export const wrapStartBatch = <F extends Callable>(
  fn: F,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F => wrapBatch(fn, "batch.submit", options, overrides);

export const wrapGetBatchStatus = <F extends Callable>(
  fn: F,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F => wrapBatch(fn, "batch.status", options, overrides);

export const wrapCancelBatch = <F extends Callable>(
  fn: F,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F => wrapBatch(fn, "batch.cancel", options, overrides);

export const wrapGetBatchResults = <F extends Callable>(
  fn: F,
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): F => {
  const config = resolvedConfig(options);

  if (!config) return fn;

  if (!config.apiKey) return fn;

  let emitter: ReturnType<typeof createGenerationEmitter>;

  try {
    emitter = createGenerationEmitter({ ...config, sdkName: "@telemetry-dev/ai-sdk" }, overrides);
  } catch (error) {
    reportTelemetryError(config.onError, error);

    return fn;
  }

  return ((...args: Parameters<F>) => {
    let attrs: Attributes;
    let span: ReturnType<ReturnType<typeof createGenerationEmitter>["tracer"]["startSpan"]>;
    let startedAt: number;

    try {
      startedAt = performance.now();
      const input = args[0];
      const batch = dataProperty(input, "batch");
      attrs = {
        "gen_ai.operation.name": "batch.results",
        "td.ai.batch.id": scalar(dataProperty(batch, "id")),
      };
      span = emitter.tracer.startSpan("batch.results", {
        kind: SpanKind.CLIENT,
        attributes: attrs,
      });
    } catch (error) {
      reportTelemetryError(config.onError, error);

      return fn(...args) as ReturnType<F>;
    }

    let count = 0;
    let finished = false;

    const finish = (failed: boolean, completed: boolean, error?: unknown) => {
      if (finished) return;
      finished = true;

      const attempt = (operation: () => void) => {
        try {
          operation();
        } catch (cause) {
          reportTelemetryError(config.onError, cause);
        }
      };

      attempt(() => span.setAttribute("td.ai.batch.item_count", count));
      attempt(() => span.setAttribute("td.ai.batch.completed", completed));

      if (failed) {
        attempt(() =>
          span.setStatus({ code: SpanStatusCode.ERROR, message: unknownErrorMessage(error) }),
        );
        attempt(() =>
          span.recordException(error instanceof Error ? error : unknownErrorMessage(error)),
        );
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
      finish(true, false, error);
      throw error;
    }

    try {
      if (value instanceof ReadableStream) {
        const reader = value.getReader();
        let cancelling = false;
        let released = false;

        const release = () => {
          if (released) return;
          released = true;
          reader.releaseLock();
        };

        const stream = new ReadableStream({
          async pull(controller) {
            try {
              const next = await reader.read();

              if (cancelling) return;

              if (next.done) {
                controller.close();
                finish(false, true);
                release();
              } else {
                count += 1;
                controller.enqueue(next.value);
              }
            } catch (error) {
              if (cancelling) return;
              controller.error(error);
              finish(true, false, error);
              release();
            }
          },
          async cancel(reason) {
            cancelling = true;

            try {
              await reader.cancel(reason);
              finish(false, false);
            } catch (error) {
              finish(true, false, error);
              throw error;
            } finally {
              release();
            }
          },
        });

        Object.defineProperty(stream, Symbol.asyncIterator, {
          configurable: true,
          value: () => {
            const streamReader = stream.getReader();
            let released = false;
            let terminal = false;

            const release = () => {
              if (released) return;
              released = true;
              streamReader.releaseLock();
            };

            return {
              async next() {
                if (terminal) return { done: true as const, value: undefined };

                try {
                  const next = await streamReader.read();

                  if (next.done) {
                    terminal = true;
                    release();
                  }

                  return next;
                } catch (error) {
                  terminal = true;
                  release();
                  throw error;
                }
              },
              async return(reason?: unknown) {
                if (terminal) return { done: true as const, value: reason };
                terminal = true;

                try {
                  await streamReader.cancel(reason);
                } finally {
                  release();
                }

                return { done: true as const, value: reason };
              },
              async throw(error?: unknown) {
                if (terminal) throw error;
                terminal = true;

                try {
                  await streamReader.cancel(error);
                } finally {
                  release();
                }

                throw error;
              },
              [Symbol.asyncIterator]() {
                return this;
              },
            };
          },
        });

        return stream as ReturnType<F>;
      }

      const iterable = dataProperty(value, Symbol.asyncIterator);

      if (typeof iterable === "function") {
        let claimed = false;

        const facade = {
          [Symbol.asyncIterator]() {
            return iterable.call(value) as AsyncIterator<unknown>;
          },
        };

        const wrapped = {
          async *[Symbol.asyncIterator]() {
            if (claimed) throw new Error("Batch results can only be iterated once");
            claimed = true;
            let failed = false;
            let completed = false;
            let error: unknown;

            try {
              for await (const item of facade) {
                count += 1;
                yield item;
              }

              completed = true;
            } catch (cause) {
              failed = true;
              error = cause;
              throw cause;
            } finally {
              finish(failed, completed, error);
            }
          },
        };

        return wrapped as ReturnType<F>;
      }

      finish(false, true);

      return value;
    } catch (error) {
      reportTelemetryError(config.onError, error);
      finish(false, false);

      return value;
    }
  }) as F;
};

function reportTelemetryError(onError: TelemetryDevOptions["onError"], error: unknown): void {
  try {
    onError?.(error instanceof Error ? error : String(error));
  } catch {}
}
