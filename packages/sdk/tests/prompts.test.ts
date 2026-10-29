import { createServer, type RequestListener } from "node:http";
import type { AddressInfo } from "node:net";

import { afterEach, beforeEach, expect, test, vi } from "vitest";

import {
  flush,
  getPrompt,
  propagateAttributes,
  PromptError,
  shutdown,
  startActiveSpan,
  startSpan,
  withPrompt,
} from "../src/index.ts";
import { setup } from "./helpers.ts";

interface Call {
  url: URL;
  init: RequestInit | undefined;
}

type Reply = Record<string, unknown> | Response | Promise<Response> | Error;

let baseCount = 0;

// The prompt cache lives for the process; a new base URL gives each test its own entries.
function newBaseUrl(): string {
  baseCount += 1;

  return `https://ingest-${baseCount}.test`;
}

function fakeFetch(...replies: Reply[]) {
  const calls: Call[] = [];

  const fetch = vi.fn(async (input: string | URL, init?: RequestInit) => {
    calls.push({ url: new URL(input), init });
    const reply = replies.length > 1 ? replies.shift()! : replies[0]!;

    if (reply instanceof Error) throw reply;

    if (reply instanceof Response) return reply.clone();

    if (reply instanceof Promise) return (await reply).clone();

    return Response.json(reply);
  });

  return { fetch: fetch as unknown as typeof globalThis.fetch, calls };
}

function reply(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    name: "support/reply",
    type: "chat",
    label: "production",
    version: 12,
    messages: [
      { role: "system", content: "Use a {{tone}} tone." },
      { role: "placeholder", name: "history" },
      { role: "user", content: "{{ customer_message }}" },
    ],
    text: null,
    model: { provider: "openai", name: "gpt-4o" },
    parameters: { temperature: 0.3 },
    tools: [],
    responseFormat: { type: "text" },
    rollout: null,
    ...overrides,
  };
}

function apiError(status: number, code: string, message: string): Response {
  return Response.json({ error: { code, message } }, { status });
}

function deferred(): { promise: Promise<Response>; resolve: (response: Response) => void } {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((done) => {
    resolve = done;
  });

  return { promise, resolve };
}

async function serve(handler: RequestListener): Promise<{ url: string; close: () => void }> {
  const server = createServer(handler);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;

  return {
    url: `http://127.0.0.1:${port}`,
    close: () => {
      server.closeAllConnections();
      server.close();
    },
  };
}

beforeEach(() => {
  vi.stubEnv("TELEMETRY_DEV_API_KEY", undefined);
  vi.stubEnv("TELEMETRY_DEV_BASE_URL", undefined);
});

afterEach(async () => {
  await shutdown();
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

test.each([
  [{}, { label: "production" }],
  [{ label: "staging" }, { label: "staging" }],
  [{ version: 13 }, { version: "13" }],
])("getPrompt(%j) sends query %j with the key and the SDK header", async (options, query) => {
  const { fetch, calls } = fakeFetch(reply());
  const baseUrl = newBaseUrl();

  await getPrompt("support/reply", { ...options, apiKey: "td_live_a", baseUrl, fetch });

  expect(calls).toHaveLength(1);
  const { url, init } = calls[0]!;
  expect(`${url.origin}${url.pathname}`).toBe(`${baseUrl}/v1/prompts`);
  expect(Object.fromEntries(url.searchParams)).toEqual({ name: "support/reply", ...query });
  expect(init?.method).toBe("GET");
  expect(init?.headers).toMatchObject({
    authorization: "Bearer td_live_a",
    "x-telemetry-dev-sdk": "@telemetry-dev/sdk",
  });
});

test("getPrompt reads the key, base URL, and fetch of the initialized client", async () => {
  const { fetch, calls } = fakeFetch(reply());
  const baseUrl = newBaseUrl();
  setup({ apiKey: "td_live_client", baseUrl, fetch });

  const prompt = await getPrompt("support/reply");

  expect(prompt.version).toBe(12);
  expect(calls[0]!.url.origin).toBe(baseUrl);
  expect(calls[0]!.init?.headers).toMatchObject({ authorization: "Bearer td_live_client" });
});

test.each([false, true])(
  "getPrompt reads the key and base URL from the environment (initialized: %s)",
  async (initialized) => {
    const { fetch, calls } = fakeFetch(reply());
    const baseUrl = newBaseUrl();
    vi.stubEnv("TELEMETRY_DEV_API_KEY", "td_live_env");
    vi.stubEnv("TELEMETRY_DEV_BASE_URL", `${baseUrl}/`);
    if (initialized) setup({ apiKey: undefined });

    await getPrompt("support/reply", { fetch });

    const { url, init } = calls[0]!;
    expect(`${url.origin}${url.pathname}`).toBe(`${baseUrl}/v1/prompts`);
    expect(init?.headers).toMatchObject({ authorization: "Bearer td_live_env" });
  },
);

test("getPrompt sends the request to the base URL that telemetry exports use", async () => {
  const exported: string[] = [];
  const { fetch: prompts, calls } = fakeFetch(reply());

  const fetch = (async (input: string | URL, init?: RequestInit) => {
    const url = new URL(input);

    if (url.pathname === "/v1/prompts") return prompts(input, init);
    exported.push(url.href);

    return new Response(null, { status: 200 });
  }) as typeof globalThis.fetch;

  const baseUrl = newBaseUrl();
  setup({ baseUrl: `${baseUrl}/`, fetch }, { spanExporter: undefined });

  startSpan("work").end();
  await flush();
  await getPrompt("support/reply");

  expect(exported).toEqual([`${baseUrl}/v1/traces`]);
  expect(calls[0]!.url.href).toBe(`${baseUrl}/v1/prompts?name=support%2Freply&label=production`);
});

test.each([
  { initialized: false, envBaseUrl: undefined },
  { initialized: true, envBaseUrl: undefined },
  { initialized: false, envBaseUrl: "" },
  { initialized: true, envBaseUrl: "" },
])(
  "without a base URL, getPrompt reads https://ingest.telemetry.dev (initialized: $initialized, TELEMETRY_DEV_BASE_URL: $envBaseUrl)",
  async ({ initialized, envBaseUrl }) => {
    const { fetch, calls } = fakeFetch(reply());
    vi.stubEnv("TELEMETRY_DEV_BASE_URL", envBaseUrl);
    if (initialized) setup();

    // The cases use one base URL, thus each uses its own key to get its own cache entry.
    await getPrompt("support/reply", {
      apiKey: `td_live_${initialized}_${envBaseUrl === undefined ? "unset" : "empty"}`,
      fetch,
    });

    expect(calls[0]!.url.href).toBe(
      "https://ingest.telemetry.dev/v1/prompts?name=support%2Freply&label=production",
    );
  },
);

test("getPrompt keeps a prompt for the TTL, then refreshes it in the background", async () => {
  vi.useFakeTimers({ toFake: ["Date"] });
  const refresh = deferred();
  const { fetch, calls } = fakeFetch(reply(), refresh.promise);
  const options = { apiKey: "td_live_a", baseUrl: newBaseUrl(), fetch };

  expect((await getPrompt("support/reply", options)).version).toBe(12);
  vi.advanceTimersByTime(59_000);
  expect((await getPrompt("support/reply", options)).version).toBe(12);
  expect(calls).toHaveLength(1);

  // After the TTL, calls get the copy at once while one refresh runs.
  vi.advanceTimersByTime(1_000);
  const versions = await Promise.all(
    [1, 2, 3].map(async () => (await getPrompt("support/reply", options)).version),
  );
  expect(versions).toEqual([12, 12, 12]);
  expect(calls).toHaveLength(2);

  refresh.resolve(Response.json(reply({ version: 13 })));
  await vi.waitFor(async () => {
    expect((await getPrompt("support/reply", options)).version).toBe(13);
  });
  expect(calls).toHaveLength(2);

  // A different key does not share the copy.
  await getPrompt("support/reply", { ...options, apiKey: "td_live_b" });
  expect(calls).toHaveLength(3);
});

test("concurrent calls share one fetch", async () => {
  const { fetch, calls } = fakeFetch(reply());
  const options = { apiKey: "td_live_a", baseUrl: newBaseUrl(), fetch };

  const prompts = await Promise.all([
    getPrompt("support/reply", options),
    getPrompt("support/reply", options),
    getPrompt("support/reply", options),
  ]);

  expect(calls).toHaveLength(1);
  expect(prompts.map((prompt) => prompt.version)).toEqual([12, 12, 12]);
});

test("a failed refresh returns the last copy and reports the error", async () => {
  vi.useFakeTimers({ toFake: ["Date"] });
  const errors: Error[] = [];
  vi.spyOn(console, "error").mockImplementation(() => {});

  const { fetch, calls } = fakeFetch(
    reply(),
    apiError(503, "prompts_unavailable", "Prompts are unavailable"),
    new TypeError("fetch failed"),
  );

  const baseUrl = newBaseUrl();
  setup({ baseUrl, fetch, onError: (error) => errors.push(error) });

  expect((await getPrompt("support/reply")).version).toBe(12);
  vi.advanceTimersByTime(60_000);
  expect((await getPrompt("support/reply")).version).toBe(12);
  await vi.waitFor(() => {
    expect(errors[0]).toMatchObject({ code: "prompts_unavailable", status: 503 });
  });

  // The stale copy counts as fresh for another TTL, so calls do not each start a fetch.
  await getPrompt("support/reply");
  expect(calls).toHaveLength(2);

  vi.advanceTimersByTime(60_000);
  expect((await getPrompt("support/reply")).version).toBe(12);
  await vi.waitFor(() => expect(errors[1]).toMatchObject({ code: "network_error" }));
});

test("with no copy, a failed fetch returns the fallback, and so do calls for 10 seconds", async () => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.spyOn(console, "error").mockImplementation(() => {});
  const { fetch, calls } = fakeFetch(
    apiError(404, "label_not_found", "support/reply has no production label"),
  );

  const options = {
    apiKey: "td_live_a",
    baseUrl: newBaseUrl(),
    fetch,
    fallback: [{ role: "user" as const, content: "Answer {{customer_message}}" }],
  };

  const prompt = await getPrompt("support/reply", options);

  expect(prompt).toMatchObject({
    name: "support/reply",
    type: "chat",
    version: null,
    label: "production",
    model: null,
    isFallback: true,
  });
  expect(prompt.compile({ customer_message: "hi" })).toEqual([
    { role: "user", content: "Answer hi" },
  ]);

  vi.advanceTimersByTime(9_999);
  expect((await getPrompt("support/reply", options)).isFallback).toBe(true);
  expect(calls).toHaveLength(1);

  vi.advanceTimersByTime(1);
  expect((await getPrompt("support/reply", options)).isFallback).toBe(true);
  expect(calls).toHaveLength(2);

  // A TTL under 10 seconds bounds the wait too.
  await getPrompt("support/reply", { ...options, cacheTtlSeconds: 0 });
  expect(calls).toHaveLength(3);
});

test("with no copy and no fallback, getPrompt rejects with a PromptError", async () => {
  const { fetch } = fakeFetch(
    apiError(404, "label_not_found", "support/reply has no production label"),
  );
  const options = { apiKey: "td_live_a", baseUrl: newBaseUrl(), fetch };

  const error = await getPrompt("support/reply", options).catch((cause: unknown) => cause);

  expect(error).toBeInstanceOf(PromptError);
  expect(error).toMatchObject({
    code: "label_not_found",
    status: 404,
    message: "support/reply has no production label",
  });

  for (const apiKey of [undefined, ""]) {
    await expect(getPrompt("support/reply", { apiKey, fetch })).rejects.toMatchObject({
      code: "missing_api_key",
    });
  }

  const { fetch: offline } = fakeFetch(new TypeError("fetch failed"));
  await expect(
    getPrompt("support/reply", { ...options, baseUrl: newBaseUrl(), fetch: offline }),
  ).rejects.toMatchObject({ code: "network_error" });

  const { fetch: invalid } = fakeFetch({ name: "support/reply", type: "chat" });
  await expect(
    getPrompt("support/reply", { ...options, baseUrl: newBaseUrl(), fetch: invalid }),
  ).rejects.toMatchObject({ code: "invalid_response" });
});

test("a redirect fails with http_302 and does not send the key to the target", async () => {
  const targetRequests: string[] = [];

  const target = await serve((request, response) => {
    targetRequests.push(request.headers.authorization ?? "");
    response.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(reply()));
  });

  const origin = await serve((_request, response) => {
    response.writeHead(302, { location: `${target.url}/v1/prompts?name=support%2Freply` }).end();
  });

  try {
    await expect(
      getPrompt("support/reply", { apiKey: "td_live_a", baseUrl: origin.url }),
    ).rejects.toMatchObject({ code: "http_302", status: 302 });
    expect(targetRequests).toEqual([]);
  } finally {
    target.close();
    origin.close();
  }
});

test("a fetch that takes more than 10 seconds fails with timeout", async () => {
  vi.useFakeTimers();

  const fetch = ((_input: string | URL | Request, init?: RequestInit) =>
    new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
    })) as typeof globalThis.fetch;

  const pending = getPrompt("support/reply", { apiKey: "td_live_a", baseUrl: newBaseUrl(), fetch });
  const result = pending.catch((cause: unknown) => cause);
  await vi.advanceTimersByTimeAsync(10_000);

  expect(await result).toMatchObject({ code: "timeout" });
});

test("getPrompt rejects a prompt of the other type and bad options", async () => {
  const { fetch } = fakeFetch(reply({ type: "text", messages: [], text: "Hi" }));
  const options = { apiKey: "td_live_a", baseUrl: newBaseUrl(), fetch };

  await expect(getPrompt("support/reply", options)).rejects.toMatchObject({
    code: "type_mismatch",
  });

  const text = await getPrompt("support/reply", { ...options, type: "text" });
  expect(text.compile()).toBe("Hi");

  await expect(getPrompt("support/reply", { ...options, label: "a", version: 2 })).rejects.toThrow(
    TypeError,
  );
  await expect(getPrompt("support/reply", { ...options, version: 0 })).rejects.toThrow(RangeError);
});

test("one session keeps one version of a gradual rollout", async () => {
  const rollout = (percent: number) =>
    reply({
      rollout: {
        percent,
        version: 13,
        messages: [{ role: "user", content: "v13" }],
        text: null,
        model: { provider: "openai", name: "gpt-4o-mini" },
        parameters: {},
        tools: [],
        responseFormat: { type: "text" },
      },
    });

  const fetchWith = async (percent: number, sessionId?: string) => {
    const options = {
      apiKey: "td_live_a",
      baseUrl: newBaseUrl(),
      fetch: fakeFetch(rollout(percent)).fetch,
    };

    return sessionId === undefined
      ? getPrompt("support/reply", options)
      : propagateAttributes({ sessionId }, () => getPrompt("support/reply", options));
  };

  // FNV-1a("support/reply:production:session-123") = 777154315, so the bucket is 15. The Python
  // SDK test asserts the same vector.
  expect((await fetchWith(15, "session-123")).version).toBe(12);
  const chosen = await fetchWith(16, "session-123");
  expect(chosen.version).toBe(13);
  expect(chosen.model).toEqual({ provider: "openai", name: "gpt-4o-mini" });
  expect(chosen.compile()).toEqual([{ role: "user", content: "v13" }]);

  // Without a session, each call draws a bucket.
  vi.spyOn(Math, "random").mockReturnValueOnce(0.159).mockReturnValueOnce(0.16);
  expect((await fetchWith(16)).version).toBe(13);
  expect((await fetchWith(16)).version).toBe(12);
});

test("compile fills variables and expands placeholders", async () => {
  const { fetch } = fakeFetch(reply());
  const prompt = await getPrompt("support/reply", {
    apiKey: "td_live_a",
    baseUrl: newBaseUrl(),
    fetch,
  });

  const history = [
    { role: "user" as const, content: "Hi {{tone}}" },
    { role: "assistant" as const, content: "Hello" },
  ];

  expect(prompt.compile({ tone: "friendly", customer_message: "Where is #4?", history })).toEqual([
    { role: "system", content: "Use a friendly tone." },
    ...history,
    { role: "user", content: "Where is #4?" },
  ]);

  // Missing values become empty text, other values JSON; prototype keys are not variables.
  expect(prompt.compile({ customer_message: 42 })).toEqual([
    { role: "system", content: "Use a  tone." },
    { role: "user", content: "42" },
  ]);
  expect(prompt.compile({ tone: null, customer_message: true })[1]).toEqual({
    role: "user",
    content: "true",
  });

  const { fetch: prototypeFetch } = fakeFetch(
    reply({ messages: [{ role: "user", content: "{{__proto__}}{{constructor}}" }] }),
  );
  const tricky = await getPrompt("support/reply", {
    apiKey: "td_live_a",
    baseUrl: newBaseUrl(),
    fetch: prototypeFetch,
  });
  expect(tricky.compile()).toEqual([{ role: "user", content: "" }]);

  expect(() => prompt.compile({ history: "not a list" as never })).toThrow(
    "The history variable must be a list of messages",
  );

  // The cached copy does not change when an app edits a prompt.
  expect(Object.isFrozen(prompt.messages[0])).toBe(true);
});

test("withPrompt stamps model-call spans in scope and no others", async () => {
  const { spans } = setup();
  const { fetch } = fakeFetch(reply());
  const prompt = await getPrompt("support/reply", {
    apiKey: "td_live_a",
    baseUrl: newBaseUrl(),
    fetch,
  });

  startSpan("before", { type: "generation" }).end();
  const outside = startSpan("outside");

  await withPrompt(prompt, async () => {
    prompt.compile({ customer_message: "Where is #4?", tone: "friendly" });
    await Promise.resolve();
    startSpan("chat gpt-4o", { type: "generation" }).end();
    startSpan("image", {
      type: "generation",
      attributes: { "gen_ai.operation.name": "generate_content" },
    }).end();
    startActiveSpan("tool", { type: "tool" }, () => {});
    startSpan("plain").end();
    startSpan("embed", { type: "embedding" }).end();
    startSpan("explicit parent", { type: "generation", parent: outside }).end();
  });

  outside.end();

  startSpan("after", { type: "generation" }).end();
  await flush();

  const byName = Object.fromEntries(spans.getFinishedSpans().map((s) => [s.name, s.attributes]));

  for (const name of ["chat gpt-4o", "image", "explicit parent"]) {
    expect(byName[name]).toMatchObject({
      "td.prompt.name": "support/reply",
      "td.prompt.version": 12,
      "td.prompt.label": "production",
      "td.prompt.variables": JSON.stringify({ customer_message: "Where is #4?", tone: "friendly" }),
    });
  }

  for (const name of ["before", "outside", "tool", "plain", "embed", "after"]) {
    expect(byName[name]?.["td.prompt.name"]).toBeUndefined();
  }
});

test.each(["before", "inside"])(
  "each task records its own variables when it compiles a shared prompt %s withPrompt",
  async (where) => {
    const { spans } = setup();
    const { fetch } = fakeFetch(reply());
    const prompt = await getPrompt("support/reply", {
      apiKey: "td_live_a",
      baseUrl: newBaseUrl(),
      fetch,
    });

    const task = async (name: string, customerMessage: string, wait: Promise<void>) => {
      if (where === "before") prompt.compile({ customer_message: customerMessage });

      await withPrompt(prompt, async () => {
        if (where === "inside") prompt.compile({ customer_message: customerMessage });
        await wait;
        startSpan(name, { type: "generation" }).end();
      });
    };

    // A compiles and waits in withPrompt. Then B compiles and calls the model, and A calls it last.
    let releaseA!: () => void;
    const a = task("a", "from A", new Promise((resolve) => (releaseA = resolve)));
    await task("b", "from B", Promise.resolve());
    releaseA();
    await a;
    await flush();

    const variables = Object.fromEntries(
      spans.getFinishedSpans().map((span) => [span.name, span.attributes["td.prompt.variables"]]),
    );
    expect(variables).toEqual({
      a: '{"customer_message":"from A"}',
      b: '{"customer_message":"from B"}',
    });
  },
);

test("variables are omitted when input capture is off; a version fetch has no label", async () => {
  const masked: string[] = [];
  const { spans } = setup({
    mask: (value, { key }) => {
      masked.push(key);

      return value;
    },
  });
  const { fetch } = fakeFetch(reply({ label: null, version: 7 }));

  const prompt = await getPrompt("support/reply", {
    version: 7,
    apiKey: "td_live_a",
    baseUrl: newBaseUrl(),
    fetch,
  });

  prompt.compile({ tone: "SECRET" });

  withPrompt(prompt, () => {
    startSpan("captured", { type: "generation" }).end();
    startSpan("private", { type: "generation", captureInput: false }).end();
  });

  await flush();
  const [captured, privateSpan] = spans.getFinishedSpans();

  expect(captured!.attributes).toMatchObject({
    "td.prompt.name": "support/reply",
    "td.prompt.version": 7,
    "td.prompt.variables": '{"tone":"SECRET"}',
  });
  expect(captured!.attributes["td.prompt.label"]).toBeUndefined();
  expect(masked).toContain("td.prompt.variables");

  expect(privateSpan!.attributes["td.prompt.name"]).toBe("support/reply");
  expect(privateSpan!.attributes["td.prompt.variables"]).toBeUndefined();

  await shutdown();
  const off = setup({ captureInput: false });
  withPrompt(prompt, () => startSpan("off", { type: "generation" }).end());
  await flush();
  expect(off.spans.getFinishedSpans()[0]!.attributes).toMatchObject({ "td.prompt.version": 7 });
  expect(off.spans.getFinishedSpans()[0]!.attributes["td.prompt.variables"]).toBeUndefined();
});

test("a fallback prompt records its name and label but no version", async () => {
  vi.spyOn(console, "error").mockImplementation(() => {});
  const { spans } = setup();
  const { fetch } = fakeFetch(new TypeError("fetch failed"));

  const prompt = await getPrompt("support/reply", {
    apiKey: "td_live_a",
    baseUrl: newBaseUrl(),
    fetch,
    fallback: [{ role: "user", content: "Hi" }],
  });

  withPrompt(prompt, () => startSpan("chat", { type: "generation" }).end());
  await flush();

  const attributes = spans.getFinishedSpans()[0]!.attributes;
  expect(attributes["td.prompt.name"]).toBe("support/reply");
  expect(attributes["td.prompt.label"]).toBe("production");
  expect(attributes["td.prompt.version"]).toBeUndefined();
});
