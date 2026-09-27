import { type Attributes, SpanStatusCode } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { expect, test } from "vitest";

import {
  wrapGenerateImage,
  wrapGenerateSpeech,
  wrapGetBatchResults,
  wrapStartBatch,
  wrapTranscribe,
} from "../src/index.ts";

function capture() {
  const spans: ReadableSpan[] = [];
  const durations: Attributes[] = [];

  return {
    spans,
    durations,
    overrides: {
      sendSpans: async (batch: ReadableSpan[]) => {
        spans.push(...batch);
      },
      recordDuration: (_seconds: number, attributes: Attributes) => durations.push(attributes),
    },
  };
}

const config = { apiKey: "td_live_test", environment: "test", serviceName: "test" };

test("media wrappers preserve structural function types and record metadata without bytes", async () => {
  const { spans, overrides } = capture();
  const bytes = new Uint8Array([1, 2, 3]);

  const generate = async (_options: {
    model: { provider: string; modelId: string };
    prompt: string;
  }) => ({ images: [{ uint8Array: bytes }], usage: { inputTokens: 4, outputTokens: 2 } });

  const wrapped: typeof generate = wrapGenerateImage(generate, config, overrides);

  expect(
    (await wrapped({ model: { provider: "openai", modelId: "gpt-image-1" }, prompt: "x" }))
      .images[0]!.uint8Array,
  ).toBe(bytes);
  expect(spans).toHaveLength(1);
  expect(spans[0]!.attributes).toMatchObject({
    "gen_ai.operation.name": "generate_content",
    "gen_ai.output.type": "image",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "gpt-image-1",
    "gen_ai.usage.input_tokens": 4,
    "gen_ai.usage.output_tokens": 2,
    "gen_ai.usage.image.output_tokens": 2,
    "td.ai.output.image_count": 1,
  });
  expect(JSON.stringify(spans[0]!.attributes)).not.toContain("1,2,3");
});

test("media wrappers preserve rejection and synchronous throw identity", async () => {
  const rejected = new Error("rejected");
  const rejectedCapture = capture();
  await expect(
    wrapTranscribe(() => Promise.reject(rejected), config, rejectedCapture.overrides)(),
  ).rejects.toBe(rejected);
  expect(rejectedCapture.spans[0]!.status.code).toBe(SpanStatusCode.ERROR);

  const synchronous = new Error("synchronous");
  const synchronousCapture = capture();

  const wrapped = wrapTranscribe(
    () => {
      throw synchronous;
    },
    config,
    synchronousCapture.overrides,
  );

  expect(() => wrapped()).toThrow(synchronous);
  expect(synchronousCapture.spans[0]!.status.code).toBe(SpanStatusCode.ERROR);
});

test("media and batch wrappers keep rejecting thenables asynchronous", async () => {
  const rejection = new Error("thenable rejected");
  const thenKey = ["t", "h", "e", "n"].join("");

  const thenable = Object.defineProperty({}, thenKey, {
    value(_resolve: (value: never) => void, reject: (error: Error) => void) {
      queueMicrotask(() => reject(rejection));
    },
  });

  for (const wrap of [wrapTranscribe, wrapStartBatch]) {
    const { spans, overrides } = capture();
    const result = wrap(() => thenable, config, overrides)();
    expect(spans).toHaveLength(0);
    await expect(result).rejects.toBe(rejection);
    expect(spans).toHaveLength(1);
    expect(spans[0]!.status.code).toBe(SpanStatusCode.ERROR);
  }
});

test("media and batch wrappers record undefined failures without changing them", async () => {
  const mediaCapture = capture();
  await expect(
    wrapTranscribe(() => Promise.reject(undefined), config, mediaCapture.overrides)(),
  ).rejects.toBeUndefined();
  expect(mediaCapture.spans[0]!.status.code).toBe(SpanStatusCode.ERROR);

  const batchCapture = capture();

  const batch = wrapStartBatch(
    () => {
      throw undefined;
    },
    config,
    batchCapture.overrides,
  );

  expect(batch).toThrow(undefined);
  expect(batchCapture.spans[0]!.status.code).toBe(SpanStatusCode.ERROR);
});

test("throwing telemetry overrides do not replace provider results or errors", async () => {
  const telemetryErrors: unknown[] = [];
  const options = { ...config, onError: (error: unknown) => telemetryErrors.push(error) };
  const { spans, overrides: capturedOverrides } = capture();
  let durationAttempts = 0;

  const overrides = {
    ...capturedOverrides,
    recordDuration: () => {
      durationAttempts += 1;
      throw new Error("duration failed");
    },
    recordTokens: () => {
      throw new Error("tokens failed");
    },
  };

  const value = { images: [{}], usage: { inputTokens: 1 } };
  expect(wrapGenerateImage(() => value, options, overrides)()).toBe(value);
  await Promise.resolve();
  expect(durationAttempts).toBe(1);
  expect(spans).toHaveLength(1);

  const providerError = new Error("provider failed");
  expect(() =>
    wrapStartBatch(
      () => {
        throw providerError;
      },
      options,
      overrides,
    )(),
  ).toThrow(providerError);
  expect(telemetryErrors.length).toBeGreaterThan(0);
});

test("throwing telemetry setup and one-shot input getters preserve exact provider outcomes", () => {
  const telemetryErrors: unknown[] = [];
  const setupError = new Error("sampling failed");

  const options = {
    ...config,
    sampler: {
      shouldSample() {
        throw setupError;
      },
      toString: () => "throwing sampler",
    },
    onError: (error: unknown) => telemetryErrors.push(error),
  };

  const mediaValue = { image: {} };
  const mediaTelemetryErrors: unknown[] = [];
  const modelError = new Error("model getter failed");
  let modelReads = 0;

  const mediaInput = Object.defineProperty({ model: undefined as unknown }, "model", {
    get() {
      modelReads += 1;

      if (modelReads === 1) throw modelError;

      return "unexpected-second-read";
    },
  });

  expect(() =>
    wrapGenerateImage(
      (input: { model: unknown }) => {
        void input.model;

        return mediaValue;
      },
      { ...config, onError: (error) => mediaTelemetryErrors.push(error) },
      capture().overrides,
    )(mediaInput),
  ).toThrow(modelError);
  expect(modelReads).toBe(1);
  expect(mediaTelemetryErrors).not.toContain(modelError);

  const batchValue = { id: "batch-1" };
  let batchCalls = 0;
  expect(
    wrapStartBatch((_input: { batch?: { id?: string } }) => {
      batchCalls += 1;

      return batchValue;
    }, options)({}),
  ).toBe(batchValue);
  expect(batchCalls).toBe(1);

  const providerError = new Error("provider failed");
  expect(() =>
    wrapStartBatch((_input: { batch?: { id?: string } }) => {
      throw providerError;
    }, options)({}),
  ).toThrow(providerError);
  expect(telemetryErrors).toContain(setupError);
});

test("throwing async iterable probes preserve the provider value", () => {
  const telemetryErrors: unknown[] = [];
  const probeError = new Error("iterator getter failed");
  let reads = 0;

  const value = Object.defineProperty({}, Symbol.asyncIterator, {
    get() {
      reads += 1;

      if (reads === 1) throw probeError;

      return async function* () {};
    },
  });

  const result = wrapGetBatchResults(
    () => value,
    { ...config, onError: (error) => telemetryErrors.push(error) },
    capture().overrides,
  )();

  expect(result).toBe(value);
  expect(() => (result as { [Symbol.asyncIterator]: unknown })[Symbol.asyncIterator]).toThrow(
    probeError,
  );
  expect(reads).toBe(1);
  expect(telemetryErrors).not.toContain(probeError);
});

test("speech character count is custom metadata, not invented token usage", async () => {
  const { spans, overrides } = capture();
  await wrapGenerateSpeech(
    async (_input: { model: { provider: string; modelId: string }; text: string }) => ({
      audio: new Uint8Array([7]),
    }),
    config,
    overrides,
  )({ model: { provider: "x", modelId: "speech" }, text: "hello" });
  expect(spans[0]!.attributes["td.ai.speech.input_character_count"]).toBe(5);
  expect(spans[0]!.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
});

test("media wrappers record string model IDs", () => {
  const { spans, overrides } = capture();
  wrapGenerateImage(
    (_input: { model: string }) => ({ image: {} }),
    config,
    overrides,
  )({
    model: "provider/model",
  });
  expect(spans[0]!.attributes["gen_ai.request.model"]).toBe("provider/model");
});

test("batch calls are separate lifecycle spans and results count a wrapped iterable", async () => {
  const { spans, overrides } = capture();
  await wrapStartBatch(
    async ({ requests }: { requests: unknown[] }) => ({
      id: "batch-1",
      status: "submitted",
      requests,
    }),
    config,
    overrides,
  )({ requests: [{}, {}] });

  const results = wrapGetBatchResults(
    (_input: { batch: { id: string } }) =>
      (async function* () {
        yield { id: "a" };
        yield { id: "b" };
      })(),
    config,
    overrides,
  )({ batch: { id: "batch-1" } });

  const items = [];

  for await (const item of results) items.push(item);
  expect(items).toHaveLength(2);

  expect(spans.map((span) => span.attributes["gen_ai.operation.name"])).toEqual([
    "batch.submit",
    "batch.results",
  ]);
  expect(spans[0]!.attributes["td.ai.batch.item_count"]).toBe(2);
  expect(spans[1]!.attributes["td.ai.batch.item_count"]).toBe(2);
  expect(spans[1]!.attributes["td.ai.batch.completed"]).toBe(true);
});

test("batch result iterable finishes once without error on early break", async () => {
  const { spans, overrides } = capture();
  let cleanups = 0;

  const results = wrapGetBatchResults(
    () =>
      (async function* () {
        try {
          yield "first";
          yield "second";
        } finally {
          cleanups += 1;
        }
      })(),
    config,
    overrides,
  )();

  for await (const _item of results) break;
  expect(cleanups).toBe(1);
  expect(spans).toHaveLength(1);
  expect(spans[0]!.attributes["td.ai.batch.item_count"]).toBe(1);
  expect(spans[0]!.attributes["td.ai.batch.completed"]).toBe(false);
  expect(spans[0]!.status.code).not.toBe(SpanStatusCode.ERROR);
});

test("batch result iterable does not close after exhaustion or a rejected next", async () => {
  for (const rejectNext of [false, true]) {
    const { spans, overrides } = capture();
    let returns = 0;
    let nextCalls = 0;

    const source = {
      [Symbol.asyncIterator]() {
        return {
          next() {
            nextCalls += 1;

            if (nextCalls === 1) return Promise.resolve({ done: false as const, value: "item" });

            if (rejectNext) return Promise.reject(undefined);

            return Promise.resolve({ done: true as const, value: undefined });
          },
          return(): Promise<IteratorResult<string>> {
            returns += 1;

            return Promise.reject(new Error("return must not be called"));
          },
        };
      },
    };

    const results = wrapGetBatchResults(() => source, config, overrides)();

    const consume = async () => {
      const items = [];

      for await (const item of results) items.push(item);

      return items;
    };

    if (rejectNext) await expect(consume()).rejects.toBeUndefined();
    else await expect(consume()).resolves.toEqual(["item"]);
    expect(returns).toBe(0);
    expect(spans[0]!.status.code === SpanStatusCode.ERROR).toBe(rejectNext);
  }
});

test("batch result iterable has one owner across concurrent and repeated iteration", async () => {
  const { spans, overrides } = capture();
  let finalized = 0;

  const source = (async function* () {
    try {
      yield "one";
      await Promise.resolve();
      yield "two";
    } finally {
      finalized += 1;
    }
  })();

  const results = wrapGetBatchResults(() => source, config, overrides)();
  const first = results[Symbol.asyncIterator]();
  const second = results[Symbol.asyncIterator]();

  expect(await first.next()).toEqual({ done: false, value: "one" });
  await expect(second.next()).rejects.toThrow("Batch results can only be iterated once");
  expect(await first.next()).toEqual({ done: false, value: "two" });
  expect(await first.next()).toEqual({ done: true, value: undefined });
  await expect(results[Symbol.asyncIterator]().next()).rejects.toThrow(
    "Batch results can only be iterated once",
  );
  expect(finalized).toBe(1);
  expect(spans).toHaveLength(1);
  expect(spans[0]!.attributes["td.ai.batch.item_count"]).toBe(2);
  expect(spans[0]!.attributes["td.ai.batch.completed"]).toBe(true);
  expect(spans[0]!.status.code).not.toBe(SpanStatusCode.ERROR);
});

test("batch result streams remain async iterable and finish when cancellation rejects", async () => {
  const { spans, overrides } = capture();
  const cancellationError = new Error("cancel failed");

  const source = new ReadableStream<string>({
    pull(controller) {
      controller.enqueue("first");
    },
    cancel() {
      throw cancellationError;
    },
  });

  const results = wrapGetBatchResults(() => source, config, overrides)();
  expect(Object.hasOwn(results, Symbol.asyncIterator)).toBe(true);
  expect(typeof results[Symbol.asyncIterator]).toBe("function");
  const reader = results.getReader();
  await reader.read();
  await expect(reader.cancel()).rejects.toBe(cancellationError);
  expect(spans).toHaveLength(1);
  expect(spans[0]!.status.code).toBe(SpanStatusCode.ERROR);
});

test("batch result stream cancellation stays successful while pull is pending", async () => {
  const { spans, overrides } = capture();
  let resolvePull: (() => void) | undefined;

  const source = new ReadableStream<string>({
    async pull(controller) {
      await new Promise<void>((resolve) => {
        resolvePull = resolve;
      });
      controller.enqueue("late");
    },
    cancel() {
      resolvePull?.();
    },
  });

  const results = wrapGetBatchResults(() => source, config, overrides)();
  const reader = results.getReader();
  const pending = reader.read();
  await Promise.resolve();
  await reader.cancel();
  await pending;

  expect(spans).toHaveLength(1);
  expect(spans[0]!.status.code).not.toBe(SpanStatusCode.ERROR);
  expect(spans[0]!.attributes["td.ai.batch.completed"]).toBe(false);
});

test("batch result stream iterator stays terminal after completion", async () => {
  const source = new ReadableStream<string>({
    start(controller) {
      controller.enqueue("only");
      controller.close();
    },
  });

  const results = wrapGetBatchResults(() => source, config, capture().overrides)();
  const iterator = results[Symbol.asyncIterator]();
  expect(await iterator.next()).toEqual({ done: false, value: "only" });
  expect(await iterator.next()).toEqual({ done: true, value: undefined });
  expect(await iterator.next()).toEqual({ done: true, value: undefined });
  await expect((iterator as AsyncIterator<string, unknown>).return?.("late")).resolves.toEqual({
    done: true,
    value: "late",
  });
});

test("batch result stream own iterator cancels once on early for-await exit", async () => {
  let cancellations = 0;

  const source = new ReadableStream<string>({
    pull(controller) {
      controller.enqueue("first");
    },
    cancel() {
      cancellations += 1;
    },
  });

  const results = wrapGetBatchResults(() => source, config, capture().overrides)();

  for await (const _item of results) break;
  expect(cancellations).toBe(1);
});
