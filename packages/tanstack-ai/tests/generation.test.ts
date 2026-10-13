import { SpanKind, SpanStatusCode, type Attributes } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  embed,
  generateAudio,
  generateImage,
  generateSpeech,
  generateTranscription,
  generateVideo,
  getVideoJobStatus,
  rerank,
  summarize,
  type EmbeddingAdapter,
  type ImageAdapter,
  type SummarizeAdapter,
  type TokenUsage,
  type TranscriptionAdapter,
  type TTSAdapter,
  type VideoAdapter,
} from "@tanstack/ai";
import { expect, test } from "vitest";

import { telemetryDev } from "../src/index.ts";

function capture() {
  const spans: ReadableSpan[] = [];
  const metrics: Array<{ type: string; value: number; attributes: Attributes }> = [];
  const errors: unknown[] = [];

  const middleware = telemetryDev(
    { apiKey: "td_live_test", onError: (error) => errors.push(error) },
    {
      sendSpans: async (batch) => {
        spans.push(...batch);
      },
      recordDuration: (value, attributes) => metrics.push({ type: "duration", value, attributes }),
      recordTokens: (type, value, attributes) => metrics.push({ type, value, attributes }),
    },
  );

  return { middleware, spans, metrics, errors };
}

const usage = (promptTokens: number, completionTokens: number, extra?: Partial<TokenUsage>) => ({
  promptTokens,
  completionTokens,
  totalTokens: promptTokens + completionTokens,
  ...extra,
});

// Minimal adapters: the activities only read kind, name, model, and the one provider method.
const adapter = <A = never>(kind: string, name: string, methods: Record<string, unknown>): A =>
  ({ kind, name, model: `${name}-model`, "~types": {}, ...methods }) as unknown as A;

const tokenMetrics = (metrics: ReturnType<typeof capture>["metrics"]) =>
  metrics.filter((metric) => metric.type === "input" || metric.type === "output");

test.each([
  {
    activity: "embedding",
    run: (middleware: never) =>
      embed({
        adapter: adapter<EmbeddingAdapter>("embedding", "openai", {
          createEmbeddings: async () => ({
            id: "e1",
            model: "text-embedding-3-small",
            embeddings: [{ index: 0, vector: [0.1, 0.2] }],
            usage: usage(8, 0),
          }),
        }),
        input: "hello",
        middleware: [middleware],
      }),
    name: "embeddings",
    operation: "embeddings",
    outputType: undefined,
    responseModel: undefined,
    input: 8,
    output: 0,
  },
  {
    activity: "image",
    run: (middleware: never) =>
      generateImage({
        adapter: adapter<ImageAdapter>("image", "openai", {
          generateImages: async () => ({
            id: "i1",
            model: "gpt-image-1-2025",
            images: [{ b64Json: "aW1hZ2UtYnl0ZXM=" }],
            usage: usage(12, 40, {
              promptTokensDetails: { textTokens: 10, imageTokens: 2 },
              completionTokensDetails: { imageTokens: 40 },
            }),
          }),
        }),
        prompt: "a lighthouse",
        middleware: [middleware],
      }),
    name: "generate_content image",
    operation: "generate_content",
    outputType: "image",
    responseModel: "gpt-image-1-2025",
    input: 12,
    output: 40,
  },
  {
    activity: "tts",
    run: (middleware: never) =>
      generateSpeech({
        adapter: adapter<TTSAdapter>("tts", "elevenlabs", {
          generateSpeech: async () => ({
            id: "s1",
            model: "eleven-v3",
            audio: "YXVkaW8tYnl0ZXM=",
            format: "mp3",
            usage: usage(5, 30, { completionTokensDetails: { audioTokens: 30 } }),
          }),
        }),
        text: "read this aloud",
        middleware: [middleware],
      }),
    name: "generate_content speech",
    operation: "generate_content",
    outputType: "speech",
    responseModel: "eleven-v3",
    input: 5,
    output: 30,
  },
  {
    activity: "transcription",
    run: (middleware: never) =>
      generateTranscription({
        adapter: adapter<TranscriptionAdapter>("transcription", "openai", {
          transcribe: async () => ({
            id: "t1",
            model: "whisper-1",
            text: "secret transcript",
            usage: usage(20, 4, { promptTokensDetails: { audioTokens: 20 } }),
          }),
        }),
        audio: "YXVkaW8tYnl0ZXM=",
        middleware: [middleware],
      }),
    name: "generate_content text",
    operation: "generate_content",
    outputType: "text",
    responseModel: "whisper-1",
    input: 20,
    output: 4,
  },
  {
    activity: "summarize",
    run: (middleware: never) =>
      summarize({
        adapter: adapter<SummarizeAdapter<string, object>>("summarize", "anthropic", {
          summarize: async () => ({
            id: "m1",
            model: "claude-resolved",
            summary: "secret summary",
            usage: usage(100, 10, { promptTokensDetails: { cachedTokens: 30 } }),
          }),
        }),
        text: "long secret article",
        middleware: [middleware],
      }),
    name: "summarize",
    operation: "chat",
    outputType: "text",
    responseModel: "claude-resolved",
    // Anthropic reports promptTokens without cache reads; the span total includes them.
    input: 130,
    output: 10,
  },
])(
  "$activity emits one $operation span with provider, model, and usage",
  async ({ run, name, operation, outputType, responseModel, input, output }) => {
    const { middleware, spans, metrics, errors } = capture();

    await run(middleware as never);

    expect(errors).toEqual([]);
    expect(spans).toHaveLength(1);
    const [span] = spans;
    expect(span!.name).toBe(name);
    expect(span!.kind).toBe(SpanKind.CLIENT);
    expect(span!.ended).toBe(true);
    expect(span!.status.code).toBe(SpanStatusCode.UNSET);
    expect(span!.attributes["gen_ai.operation.name"]).toBe(operation);
    expect(span!.attributes["gen_ai.output.type"]).toBe(outputType);
    expect(span!.attributes["gen_ai.provider.name"]).toBeTypeOf("string");
    expect(span!.attributes["gen_ai.request.model"]).toBe(
      `${String(span!.attributes["gen_ai.provider.name"])}-model`,
    );
    expect(span!.attributes["gen_ai.response.model"]).toBe(responseModel);
    expect(span!.attributes["gen_ai.usage.input_tokens"]).toBe(input);
    expect(span!.attributes["gen_ai.usage.output_tokens"]).toBe(output);

    // Inputs and outputs (prompts, text, audio, image bytes) are never captured.
    const values = JSON.stringify(Object.values(span!.attributes));
    expect(span!.attributes["gen_ai.input.messages"]).toBeUndefined();
    expect(span!.attributes["gen_ai.output.messages"]).toBeUndefined();
    expect(values).not.toMatch(/secret|lighthouse|aloud|hello|YXVkaW8|aW1hZ2U/);

    const durations = metrics.filter((metric) => metric.type === "duration");
    expect(durations).toHaveLength(1);
    expect(durations[0]!.attributes["gen_ai.operation.name"]).toBe(operation);
    expect(tokenMetrics(metrics).map((metric) => [metric.type, metric.value])).toEqual([
      ["input", input],
      ["output", output],
    ]);
  },
);

test("media usage records provider-reported modality token breakdowns", async () => {
  const { middleware, spans } = capture();

  await generateImage({
    adapter: adapter<ImageAdapter>("image", "openai", {
      generateImages: async () => ({
        id: "i1",
        model: "gpt-image-1",
        images: [{ url: "https://example.com/i.png" }],
        usage: usage(12, 40, {
          promptTokensDetails: { textTokens: 10, imageTokens: 2 },
          completionTokensDetails: { imageTokens: 40 },
        }),
      }),
    }),
    prompt: "p",
    middleware: [middleware],
  });

  expect(spans[0]!.attributes).toMatchObject({
    "gen_ai.usage.text.input_tokens": 10,
    "gen_ai.usage.image.input_tokens": 2,
    "gen_ai.usage.image.output_tokens": 40,
  });
  expect(spans[0]!.attributes["gen_ai.usage.audio.input_tokens"]).toBeUndefined();
});

test("a generation without reported usage omits token attributes and metrics", async () => {
  const { middleware, spans, metrics } = capture();

  await generateImage({
    adapter: adapter<ImageAdapter>("image", "fal", {
      generateImages: async () => ({ id: "i1", model: "flux", images: [] }),
    }),
    prompt: "p",
    middleware: [middleware],
  });

  expect(spans).toHaveLength(1);
  expect(spans[0]!.attributes["gen_ai.usage.input_tokens"]).toBeUndefined();
  expect(spans[0]!.attributes["gen_ai.usage.output_tokens"]).toBeUndefined();
  expect(tokenMetrics(metrics)).toEqual([]);
});

test.each([
  {
    activity: "embedding",
    run: (middleware: never, error: unknown) =>
      embed({
        adapter: adapter<EmbeddingAdapter>("embedding", "openai", {
          createEmbeddings: async () => {
            throw error;
          },
        }),
        input: "x",
        middleware: [middleware],
      }),
  },
  {
    activity: "tts",
    run: (middleware: never, error: unknown) =>
      generateSpeech({
        adapter: adapter<TTSAdapter>("tts", "openai", {
          generateSpeech: async () => {
            throw error;
          },
        }),
        text: "x",
        middleware: [middleware],
      }),
  },
  {
    activity: "video creation",
    run: (middleware: never, error: unknown) =>
      generateVideo({
        adapter: adapter<VideoAdapter>("video", "openai", {
          createVideoJob: async () => {
            throw error;
          },
        }),
        prompt: "x",
        middleware: [middleware],
      }),
  },
])("$activity failure rejects unchanged and flushes one error span", async ({ run }) => {
  const { middleware, spans, metrics } = capture();
  const error = new TypeError("quota exceeded");

  await expect(run(middleware as never, error)).rejects.toBe(error);

  expect(spans).toHaveLength(1);
  expect(spans[0]!.ended).toBe(true);
  expect(spans[0]!.status).toEqual({ code: SpanStatusCode.ERROR, message: "quota exceeded" });
  expect(spans[0]!.attributes["error.type"]).toBe("TypeError");
  expect(spans[0]!.events.map((event) => event.name)).toEqual(["exception"]);
  expect(metrics.filter((metric) => metric.type === "duration")).toHaveLength(1);
  expect(tokenMetrics(metrics)).toEqual([]);
});

test("an aborted image generation flushes one cancelled span", async () => {
  const { middleware, spans } = capture();
  const controller = new AbortController();

  await expect(
    generateImage({
      adapter: adapter<ImageAdapter>("image", "openai", {
        generateImages: () => {
          controller.abort();

          return new Promise(() => {});
        },
      }),
      prompt: "p",
      abortSignal: controller.signal,
      middleware: [middleware],
    }),
  ).rejects.toBeDefined();

  expect(spans).toHaveLength(1);
  expect(spans[0]!.attributes["error.type"]).toBe("cancelled");
});

test("untraced activities emit nothing and do not fail", async () => {
  const { middleware, spans, metrics, errors } = capture();

  const ranked = await rerank({
    adapter: adapter("rerank", "cohere", {
      rerank: async () => ({
        id: "r1",
        model: "rerank-3",
        ranking: [{ index: 0, score: 0.9, document: "a" }],
        usage: usage(4, 0),
      }),
    }),
    query: "q",
    documents: ["a"],
    middleware: [middleware],
  } as never);

  const audio = await generateAudio({
    adapter: adapter("audio", "elevenlabs", {
      generateAudio: async () => ({
        id: "a1",
        model: "music",
        audio: { b64Json: "eA==", contentType: "audio/mpeg" },
        usage: usage(1, 1),
      }),
    }),
    prompt: "jingle",
    middleware: [middleware],
  } as never);

  await expect(
    generateAudio({
      adapter: adapter("audio", "elevenlabs", {
        generateAudio: async () => {
          throw new Error("audio failed");
        },
      }),
      prompt: "jingle",
      middleware: [middleware],
    } as never),
  ).rejects.toThrow("audio failed");

  expect(ranked).toMatchObject({ rerankedDocuments: ["a"] });
  expect(audio).toMatchObject({ model: "music" });
  expect(spans).toEqual([]);
  expect(metrics).toEqual([]);
  expect(errors).toEqual([]);
});

const videoAdapter = (statuses: string[], over?: Record<string, unknown>) => {
  const queue = [...statuses];

  return adapter<VideoAdapter>("video", "openai", {
    createVideoJob: async () => ({ jobId: "job/42", model: "sora-2-2025" }),
    getVideoStatus: async (jobId: string) => ({ jobId, status: queue.shift() ?? "processing" }),
    getVideoUrl: async (jobId: string) => ({
      jobId,
      url: "https://cdn.example.com/v.mp4",
      usage: usage(0, 0, { durationSeconds: 8, cost: 0.4 }),
    }),
    ...over,
  });
};

const traceId = (span: ReadableSpan) => span.spanContext().traceId;

test("video job creation and the terminal poll are correlated by job id; earlier polls emit nothing", async () => {
  const { middleware, spans, metrics, errors } = capture();
  const video = videoAdapter(["processing", "completed"]);

  const job = await generateVideo({
    adapter: video,
    prompt: "secret prompt",
    middleware: [middleware],
  });

  expect(job).toEqual({ jobId: "job/42", model: "sora-2-2025" });
  expect(spans).toHaveLength(1);
  const [created] = spans;
  expect(created!.ended).toBe(true);
  expect(created!.attributes).toMatchObject({
    "gen_ai.operation.name": "generate_content",
    "gen_ai.output.type": "video",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "openai-model",
    "gen_ai.response.model": "sora-2-2025",
    "gen_ai.response.id": "job/42",
  });
  expect(JSON.stringify(created!.attributes)).not.toContain("secret");
  // Job acceptance is a point in time; its duration is not a provider latency sample.
  expect(metrics).toEqual([]);

  const pending = await getVideoJobStatus({
    adapter: video,
    jobId: job.jobId,
    middleware: [middleware],
  });

  expect(pending.status).toBe("processing");
  expect(spans).toHaveLength(1);
  expect(metrics).toEqual([]);

  const done = await getVideoJobStatus({
    adapter: video,
    jobId: job.jobId,
    middleware: [middleware],
  });

  expect(done).toMatchObject({ status: "completed", url: "https://cdn.example.com/v.mp4" });
  expect(spans).toHaveLength(2);
  const terminal = spans[1]!;
  expect(terminal.ended).toBe(true);
  expect(terminal.status.code).toBe(SpanStatusCode.UNSET);
  expect(terminal.attributes).toMatchObject({
    "gen_ai.operation.name": "generate_content",
    "gen_ai.output.type": "video",
    "gen_ai.response.id": "job/42",
    "gen_ai.usage.cost": 0.4,
  });
  expect(traceId(terminal)).toBe(traceId(created!));
  expect(metrics.filter((metric) => metric.type === "duration")).toHaveLength(1);
  expect(errors).toEqual([]);
});

test("different video jobs land in different traces", async () => {
  const { middleware, spans } = capture();
  let next = 0;

  const video = videoAdapter([], {
    createVideoJob: async () => ({ jobId: `job-${++next}`, model: "sora-2" }),
  });

  await generateVideo({ adapter: video, prompt: "a", middleware: [middleware] });
  await generateVideo({ adapter: video, prompt: "b", middleware: [middleware] });

  expect(spans.map((span) => span.attributes["gen_ai.response.id"])).toEqual(["job-1", "job-2"]);
  expect(traceId(spans[0]!)).not.toBe(traceId(spans[1]!));
});

test("a failed video job poll flushes an error span correlated to the job", async () => {
  const { middleware, spans, metrics } = capture();

  const video = videoAdapter([], {
    getVideoStatus: async (jobId: string) => ({ jobId, status: "failed", error: "content policy" }),
  });

  const job = await generateVideo({ adapter: video, prompt: "x", middleware: [middleware] });
  const status = await getVideoJobStatus({
    adapter: video,
    jobId: job.jobId,
    middleware: [middleware],
  });

  expect(status.status).toBe("failed");
  expect(spans).toHaveLength(2);
  const failed = spans[1]!;
  expect(failed.ended).toBe(true);
  expect(failed.status).toEqual({ code: SpanStatusCode.ERROR, message: "content policy" });
  expect(failed.attributes["gen_ai.response.id"]).toBe("job/42");
  expect(traceId(failed)).toBe(traceId(spans[0]!));
  expect(metrics.filter((metric) => metric.type === "duration")).toHaveLength(1);
});

test("streaming video traces one span from job creation to completion", async () => {
  const { middleware, spans } = capture();
  const video = videoAdapter(["processing", "completed"]);

  const chunks = [];

  for await (const chunk of generateVideo({
    adapter: video,
    prompt: "x",
    stream: true,
    pollingInterval: 0,
    middleware: [middleware],
  })) {
    chunks.push(chunk);

    // Non-terminal status chunks must not end the trace.
    if (chunk.type === "CUSTOM" && chunk.name === "video:status") {
      const value = chunk.value as { status: string };

      if (value.status !== "completed") expect(spans).toEqual([]);
    }
  }

  expect(chunks.at(-1)?.type).toBe("RUN_FINISHED");
  expect(spans).toHaveLength(1);
  expect(spans[0]!.attributes).toMatchObject({
    "gen_ai.response.id": "job/42",
    "gen_ai.output.type": "video",
    "gen_ai.usage.cost": 0.4,
  });
});
