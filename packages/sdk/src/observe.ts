import { reportError, withContext } from "@telemetry-dev/otel";

import { currentClient } from "./client.ts";
import { createSpanHandle, isThenable, type SpanHandle, type StartSpanOptions } from "./span.ts";

export interface ObserveOptions extends StartSpanOptions {
  /** Span name; defaults to fn.name or "anonymous". */
  name?: string;
}

/**
 * Wrap a function (sync or async) so every call becomes a span: args → input, return value →
 * output, thrown/rejected errors captured and rethrown. Activates context so nested SDK calls
 * parent to it. Resolves the client at call time, so wrapping at module load before init() works.
 */
export function observe<TThis, TArgs extends unknown[], TReturn>(
  fn: (this: TThis, ...args: TArgs) => TReturn,
  options?: ObserveOptions,
): (...args: TArgs) => TReturn {
  const wrapped = function (this: TThis, ...args: TArgs): TReturn {
    const core = currentClient().core;

    if (!core) return fn.apply(this, args);

    let handle: SpanHandle;

    try {
      const name = options?.name ?? (fn.name || "anonymous");

      const input =
        options?.input !== undefined
          ? options.input
          : args.length === 0
            ? undefined
            : args.length === 1
              ? args[0]
              : args;

      handle = createSpanHandle(core, name, { ...options, input });
    } catch (error) {
      reportError(core.config.onError, error instanceof Error ? error : new Error(String(error)));

      return fn.apply(this, args);
    }

    try {
      const result = withContext(handle.context, () => fn.apply(this, args));

      if (isThenable(result)) {
        return result.then(
          (value) => {
            handle.end({ output: value });

            return value;
          },
          (error: Error) => {
            handle.end({ error: error instanceof Error ? error : new Error(String(error)) });
            throw error;
          },
        ) as TReturn;
      }

      handle.end({ output: result });

      return result;
    } catch (error) {
      handle.end({ error: error instanceof Error ? error : new Error(String(error)) });
      throw error;
    }
  };

  Object.defineProperty(wrapped, "name", { value: fn.name || "anonymous", configurable: true });

  return wrapped;
}
