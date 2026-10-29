import { createServer, type RequestListener } from "node:http";
import type { AddressInfo } from "node:net";
import { setFlagsFromString } from "node:v8";
import { runInNewContext } from "node:vm";

import { afterEach, beforeEach, expect, test, vi } from "vitest";

import {
  type ChatMessage,
  flush,
  getPrompt,
  type Prompt,
  promptAttributesFor,
  type PromptVariables,
  propagateAttributes,
  PromptError,
  shutdown,
  startActiveSpan,
  startSpan,
  updateActiveSpan,
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
  "without a base URL, getPrompt reads https://api.telemetry.dev (initialized: $initialized, TELEMETRY_DEV_BASE_URL: $envBaseUrl)",
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
      "https://api.telemetry.dev/v1/prompts?name=support%2Freply&label=production",
    );
  },
);

test("getPrompt keeps a prompt for the TTL, then refreshes it in the background", async () => {
  vi.useFakeTimers({ toFake: ["Date", "performance"] });
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

test("getPrompt measures the TTL in elapsed time, not by the wall clock", async () => {
  vi.useFakeTimers({ toFake: ["Date", "performance"], now: Date.parse("2026-10-29T12:00:00Z") });
  const { fetch, calls } = fakeFetch(reply());
  const options = { apiKey: "td_live_a", baseUrl: newBaseUrl(), fetch };

  await getPrompt("support/reply", options);
  // The system clock moves back an hour; 60 seconds still pass.
  vi.setSystemTime(Date.parse("2026-10-29T11:00:00Z"));
  vi.advanceTimersByTime(60_000);
  await getPrompt("support/reply", options);
  expect(calls).toHaveLength(2);
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
  vi.useFakeTimers({ toFake: ["Date", "performance"] });
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
  vi.useFakeTimers({ toFake: ["Date", "performance"] });
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

  const { label: _label, ...unlabeled } = reply();
  const { fetch: noLabel } = fakeFetch(unlabeled);
  await expect(
    getPrompt("support/reply", { ...options, baseUrl: newBaseUrl(), fetch: noLabel }),
  ).rejects.toMatchObject({ code: "invalid_response" });
});

test.each([
  { status: 429, retryAfter: "120", wait: 120 },
  { status: 503, retryAfter: "120", wait: 120 },
  // An HTTP date 120 seconds after the fake clock starts.
  { status: 429, retryAfter: "date", wait: 120 },
  { status: 429, retryAfter: "86400", wait: 3600 },
  // Not a Retry-After that getPrompt reads: the 10 second error wait applies.
  { status: 429, retryAfter: "soon", wait: 10 },
  { status: 429, retryAfter: "Thu, 01 Jan 2099 00:00:00", wait: 10 },
  { status: 429, retryAfter: "01 Jan 2099 00:00:00 GMT", wait: 10 },
  { status: 500, retryAfter: "120", wait: 10 },
])(
  "with no copy, getPrompt does not fetch before the Retry-After ($status, $retryAfter)",
  async ({ status, retryAfter, wait }) => {
    vi.useFakeTimers({ toFake: ["Date", "performance"], now: Date.parse("2026-10-29T12:00:00Z") });
    vi.spyOn(console, "error").mockImplementation(() => {});
    const header =
      retryAfter === "date" ? new Date(Date.now() + 120_000).toUTCString() : retryAfter;

    const { fetch, calls } = fakeFetch(
      Response.json(
        { error: { code: "rate_limited", message: "Too many prompt requests" } },
        { status, headers: { "retry-after": header } },
      ),
    );

    const options = {
      apiKey: "td_live_a",
      baseUrl: newBaseUrl(),
      fetch,
      fallback: [{ role: "user" as const, content: "Hi" }],
    };

    expect((await getPrompt("support/reply", options)).isFallback).toBe(true);
    vi.advanceTimersByTime(wait * 1000 - 1);
    expect((await getPrompt("support/reply", options)).isFallback).toBe(true);
    expect(calls).toHaveLength(1);

    vi.advanceTimersByTime(1);
    await getPrompt("support/reply", options);
    expect(calls).toHaveLength(2);
  },
);

test("a refresh that gets a Retry-After keeps the copy until then", async () => {
  vi.useFakeTimers({ toFake: ["Date", "performance"] });
  vi.spyOn(console, "error").mockImplementation(() => {});

  const { fetch, calls } = fakeFetch(
    reply(),
    Response.json(
      { error: { code: "rate_limited", message: "Too many prompt requests" } },
      { status: 429, headers: { "retry-after": "300" } },
    ),
    reply({ version: 13 }),
  );

  const options = { apiKey: "td_live_a", baseUrl: newBaseUrl(), fetch };

  expect((await getPrompt("support/reply", options)).version).toBe(12);
  vi.advanceTimersByTime(60_000);
  expect((await getPrompt("support/reply", options)).version).toBe(12);
  await vi.waitFor(() => expect(calls).toHaveLength(2));
  await new Promise((resolve) => setTimeout(resolve, 0));

  // The TTL ends after 60 seconds, but the server asked for 300.
  vi.advanceTimersByTime(299_999);
  expect((await getPrompt("support/reply", options)).version).toBe(12);
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(calls).toHaveLength(2);

  vi.advanceTimersByTime(1);
  await getPrompt("support/reply", options);
  await vi.waitFor(async () => {
    expect((await getPrompt("support/reply", options)).version).toBe(13);
  });
  expect(calls).toHaveLength(3);
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

async function promptOf(overrides: Record<string, unknown>): Promise<Prompt> {
  const { fetch } = fakeFetch(reply(overrides));

  return getPrompt("support/reply", {
    apiKey: "td_live_a",
    baseUrl: newBaseUrl(),
    fetch,
    type: overrides.type as "chat",
  });
}

const DESK_PROMPT = {
  messages: [
    { role: "system", content: "Answer as the {{team}} desk." },
    { role: "user", content: "Order {{order}}: where is it?" },
  ],
};

test("promptAttributesFor names the compile of a message object; a copy or a string names nothing", async () => {
  setup();
  const prompt = await promptOf(DESK_PROMPT);
  const compiled = prompt.compile({ team: "billing", order: "#4" }) as ChatMessage[];
  const attributes = {
    "td.prompt.name": "support/reply",
    "td.prompt.version": 12,
    "td.prompt.label": "production",
    "td.prompt.variables": '{"team":"billing","order":"#4"}',
  };

  expect(promptAttributesFor(compiled)).toEqual(attributes);
  expect(promptAttributesFor(compiled[0])).toEqual(attributes);
  expect(JSON.stringify(compiled[0])).toBe(
    '{"role":"system","content":"Answer as the billing desk."}',
  );

  for (const copy of [
    compiled.map((message) => ({ ...message })),
    JSON.parse(JSON.stringify(compiled)) as unknown,
    structuredClone(compiled),
    compiled.map((message) => message.content),
  ]) {
    expect(promptAttributesFor(copy)).toBeUndefined();
  }

  const text = await promptOf({ type: "text", messages: [], text: "Summarize {{doc}}." });
  expect(promptAttributesFor([text.compile({ doc: "the policy" })])).toBeUndefined();
});

test("an unrelated message with the text of a compiled message is not tagged", async () => {
  setup();
  const prompt = await promptOf({
    messages: [{ role: "system", content: "You are a helpful assistant." }],
  });
  prompt.compile();

  expect(
    promptAttributesFor([
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is 2 + 2?" },
    ]),
  ).toBeUndefined();
});

test("two tasks with one text and different other variables each record their own", async () => {
  const { spans } = setup();
  const prompt = await promptOf(DESK_PROMPT);

  // Both compiles make the system text "Answer as the billing desk.", and each task sends only it.
  const [systemA] = prompt.compile({ team: "billing", order: "#A" }) as ChatMessage[];
  const [systemB] = prompt.compile({ team: "billing", order: "#B" }) as ChatMessage[];
  startSpan("a", { type: "generation", attributes: promptAttributesFor([systemA]) }).end();
  startSpan("b", { type: "generation", attributes: promptAttributesFor([systemB]) }).end();
  await flush();

  expect(
    Object.fromEntries(
      spans.getFinishedSpans().map((span) => [span.name, span.attributes["td.prompt.variables"]]),
    ),
  ).toEqual({
    a: '{"team":"billing","order":"#A"}',
    b: '{"team":"billing","order":"#B"}',
  });
});

test("a rollout with two versions of the same messages records the version of each call", async () => {
  setup();
  const { fetch } = fakeFetch(
    reply({
      ...DESK_PROMPT,
      rollout: {
        percent: 16,
        version: 13,
        ...DESK_PROMPT,
        text: null,
        model: { provider: "openai", name: "gpt-4o-mini" },
        parameters: {},
        tools: [],
        responseFormat: { type: "text" },
      },
    }),
  );
  const options = { apiKey: "td_live_a", baseUrl: newBaseUrl(), fetch };
  vi.spyOn(Math, "random").mockReturnValueOnce(0.159).mockReturnValueOnce(0.16);

  const rollout = await getPrompt("support/reply", options);
  const production = await getPrompt("support/reply", options);
  const fromRollout = rollout.compile({ team: "billing", order: "#4" });
  const fromProduction = production.compile({ team: "billing", order: "#4" });

  expect(promptAttributesFor(fromRollout)?.["td.prompt.version"]).toBe(13);
  expect(promptAttributesFor(fromProduction)?.["td.prompt.version"]).toBe(12);
});

test("the newest compiled message wins, also after 170 messages of history", async () => {
  setup();
  const prompt = await promptOf({
    messages: [
      { role: "system", content: "Answer as the {{team}} desk." },
      { role: "placeholder", name: "history" },
      { role: "user", content: "{{question}}" },
    ],
  });
  const history = Array.from({ length: 170 }, (_, index) => ({
    role: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
    content: `message ${index}`,
  }));

  const firstTurn = prompt.compile({
    team: "billing",
    question: "Where is #4?",
    history,
  }) as ChatMessage[];
  const answer = { role: "assistant", content: "On its way." };
  const secondTurn = prompt.compile({
    team: "returns",
    question: "Can I return it?",
  }) as ChatMessage[];
  const chat = [...firstTurn, answer, ...secondTurn.slice(1)];

  expect(promptAttributesFor(firstTurn)?.["td.prompt.variables"]).toContain('"team":"billing"');
  expect(promptAttributesFor([firstTurn[0], ...history])?.["td.prompt.variables"]).toContain(
    '"team":"billing"',
  );
  expect(promptAttributesFor(chat)?.["td.prompt.variables"]).toBe(
    '{"team":"returns","question":"Can I return it?"}',
  );
});

test("a message that a placeholder brings from an older compile does not win", async () => {
  setup();
  const router = await promptOf({
    name: "router",
    version: 1,
    messages: [{ role: "user", content: "{{question}}" }],
  });
  const answer = await promptOf({
    name: "answer",
    version: 7,
    messages: [
      { role: "system", content: "Answer for {{team}}." },
      { role: "placeholder", name: "history" },
    ],
  });
  const chat = await promptOf({
    name: "chat",
    messages: [{ role: "user", content: "{{question}}" }],
  });
  const summary = await promptOf({
    name: "summary",
    messages: [
      { role: "system", content: "Summarize the chat." },
      { role: "placeholder", name: "history" },
    ],
  });

  const question = router.compile({ question: "Where is #4?" }) as ChatMessage[];
  const routed = answer.compile({ team: "billing", history: question });
  // The last 1000 messages are all from chat; the summary message is the first.
  const conversation = Array.from(
    { length: 1500 },
    (_, index) => chat.compile({ question: `question ${index}` }) as ChatMessage[],
  ).flat();
  const summarized = summary.compile({ history: conversation });

  expect(promptAttributesFor(routed)).toEqual({
    "td.prompt.name": "answer",
    "td.prompt.version": 7,
    "td.prompt.label": "production",
    "td.prompt.variables":
      '{"team":"billing","history":[{"role":"user","content":"Where is #4?"}]}',
  });
  expect(promptAttributesFor(summarized)?.["td.prompt.name"]).toBe("summary");
});

test("promptAttributesFor looks at the first and the last 1000 messages", async () => {
  setup();
  const prompt = await promptOf(DESK_PROMPT);
  const [system] = prompt.compile({ team: "billing", order: "#4" }) as ChatMessage[];
  const filler = (count: number) => Array.from({ length: count }, () => ({ role: "user" }));

  expect(promptAttributesFor([...filler(1000), system, ...filler(999)])).toBeDefined();
  expect(promptAttributesFor([...filler(999), system, ...filler(1001)])).toBeDefined();
  expect(promptAttributesFor([...filler(1001), system, ...filler(999)])).toBeDefined();
  expect(promptAttributesFor([...filler(1000), system, ...filler(1000)])).toBeUndefined();
});

test("compile records its variables, not later changes of the object or of its lists", async () => {
  const { spans } = setup();
  const prompt = await promptOf({
    messages: [
      { role: "placeholder", name: "history" },
      { role: "user", content: "Order {{order}}" },
    ],
  });
  const history: ChatMessage[] = [{ role: "user", content: "Hi" }];
  const variables = { order: "#A", history };

  const first = prompt.compile(variables);
  variables.order = "#B";
  history.push({ role: "assistant", content: "Hello" });
  const second = prompt.compile(variables);
  variables.order = "#C";
  history.push({ role: "user", content: "Added after compile" });
  withPrompt(prompt, () => startSpan("scope", { type: "generation" }).end());
  await flush();

  const secondVariables =
    '{"order":"#B","history":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}';
  expect(promptAttributesFor(first)?.["td.prompt.variables"]).toBe(
    '{"order":"#A","history":[{"role":"user","content":"Hi"}]}',
  );
  expect(promptAttributesFor(second)?.["td.prompt.variables"]).toBe(secondVariables);
  expect(spans.getFinishedSpans()[0]!.attributes["td.prompt.variables"]).toBe(secondVariables);
});

// A WeakRef keeps its target until the current job ends, so collect in a new one.
async function collect(): Promise<void> {
  setFlagsFromString("--expose-gc");
  const gc = runInNewContext("gc") as () => void;
  await new Promise((resolve) => setTimeout(resolve, 0));
  gc();
}

const RAG_PROMPT = {
  messages: [
    { role: "system", content: "Answer from {{documents}}." },
    { role: "user", content: "{{question}}" },
  ],
};

test("the variables of a compile are freed with its messages", async () => {
  setup();
  const prompt = await promptOf(RAG_PROMPT);

  let document: ChatMessage | undefined = { role: "user", content: "The refund policy" };
  const freed = new WeakRef(document);
  let messages = prompt.compile({ documents: [document], question: "q" }) as
    | ChatMessage[]
    | undefined;
  document = undefined;
  // withPrompt keeps the variables of the last compile of a prompt object.
  prompt.compile({ documents: [], question: "other" });
  await collect();
  expect(freed.deref()).toBeDefined();

  expect(messages).toHaveLength(2);
  messages = undefined;
  await collect();
  expect(freed.deref()).toBeUndefined();
});

test("with input capture off, compiled messages keep no variables", async () => {
  setup({ captureInput: false });
  const prompt = await promptOf(RAG_PROMPT);
  const documents: WeakRef<ChatMessage>[] = [];
  const questions: ChatMessage[] = [];

  for (let turn = 0; turn < 200; turn += 1) {
    const document: ChatMessage = { role: "user", content: `document ${turn}` };
    documents.push(new WeakRef(document));
    const compiled = prompt.compile({ documents: [document], question: `question ${turn}` });
    questions.push(compiled.at(-1) as ChatMessage);
  }
  await collect();

  // withPrompt keeps the variables of the last compile of a prompt object.
  expect(documents.filter((document) => document.deref()).length).toBe(1);

  await shutdown();
  setup();
  expect(promptAttributesFor(questions)).toEqual({
    "td.prompt.name": "support/reply",
    "td.prompt.version": 12,
    "td.prompt.label": "production",
  });
});

// Each row keeps the compiled question and the answer of a turn for the next turn: the variable
// that gets them, and the value of that variable after the turn.
test.each<[string, string, (kept: unknown, question: object, answer: object) => unknown]>([
  ["a new list each turn", "history", (kept, ...turn) => [...(kept as object[]), ...turn]],
  [
    "one list trimmed in place to 20 messages",
    "history",
    (kept, ...turn) => {
      const history = kept as object[];
      history.push(...turn);
      history.splice(0, history.length - 20);

      return history;
    },
  ],
  [
    "a new list of the last 20 messages",
    "history",
    (kept, ...turn) => [...(kept as object[]), ...turn].slice(-20),
  ],
  ["the last question, not in a list", "previous", (_kept, question) => question],
])(
  "a chat history in %s keeps the variables of its last two turns",
  async (_shape, variable, next) => {
    setup();
    const prompt = await promptOf({
      messages: [
        { role: "system", content: "Answer as the {{team}} desk." },
        { role: "placeholder", name: "history" },
        { role: "user", content: "{{question}}" },
      ],
    });
    const turns: WeakRef<ChatMessage>[] = [];
    let kept: unknown = variable === "history" ? [] : undefined;
    let last: ChatMessage[] = [];
    let older: ChatMessage | undefined;

    // An unused variable marks the variables of each turn.
    for (let turn = 0; turn < 300; turn += 1) {
      const marker: ChatMessage = { role: "user", content: `turn ${turn}` };
      turns.push(new WeakRef(marker));
      last = prompt.compile({
        team: "billing",
        question: `question ${turn}`,
        marker: [marker],
        [variable]: kept,
      } as PromptVariables) as ChatMessage[];
      const question = last.at(-1)!;

      if (turn === 297) older = question;
      kept = next(kept, question, { role: "assistant", content: `answer ${turn}` });
    }
    await collect();

    expect(turns.filter((turn) => turn.deref()).length).toBe(2);
    expect(promptAttributesFor(last)?.["td.prompt.variables"]).toContain(
      '"question":"question 299"',
    );
    expect(promptAttributesFor(older)).toEqual({
      "td.prompt.name": "support/reply",
      "td.prompt.version": 12,
      "td.prompt.label": "production",
    });
  },
);

test("a guard prompt keeps the variables of its request, also after other compiles", async () => {
  setup();
  const answer = await promptOf(RAG_PROMPT);
  const guard = await promptOf({
    name: "guard",
    messages: [
      { role: "system", content: "Is this safe?" },
      { role: "placeholder", name: "conversation" },
    ],
  });
  const other = await promptOf({ name: "other", messages: [{ role: "user", content: "{{q}}" }] });
  const variables = '{"documents":"The refund policy","question":"Where is #4?"}';

  const messages = answer.compile({
    documents: "The refund policy",
    question: "Where is #4?",
  }) as ChatMessage[];
  // Other requests of the server compile before this request runs its guard.
  for (let index = 0; index < 150; index += 1) other.compile({ q: `${index}` });
  guard.compile({ conversation: messages });
  expect(promptAttributesFor(messages)?.["td.prompt.variables"]).toBe(variables);

  // An agent loop runs the guard before each step.
  const conversation = [...messages];
  for (let step = 0; step < 150; step += 1) {
    guard.compile({ conversation });
    conversation.push({ role: "assistant", content: `step ${step}` });
  }
  expect(promptAttributesFor(conversation)?.["td.prompt.variables"]).toBe(variables);
});

test("withPrompt wins over a compiled message of another prompt, through a parent and in updates", async () => {
  const { spans } = setup();
  const scoped = await promptOf({ name: "scoped/prompt" });
  // Fetched by version, so it has no label that the label of the scope could hide.
  const { fetch } = fakeFetch(
    reply({
      name: "other/prompt",
      label: null,
      version: 3,
      messages: [{ role: "user", content: "Cancel order {{order}}." }],
    }),
  );
  const other = await getPrompt("other/prompt", {
    apiKey: "td_live_a",
    baseUrl: newBaseUrl(),
    version: 3,
    fetch,
  });
  const compiled = other.compile({ order: "#9" });

  const parent = withPrompt(scoped, () => {
    startSpan("in scope", { type: "generation", attributes: promptAttributesFor(compiled) }).end();

    return startSpan("parent");
  });
  const attributes = promptAttributesFor(compiled);
  startSpan("with parent", { type: "generation", parent, attributes }).end();
  parent.end();
  withPrompt(scoped, () => {
    startSpan("updated", { type: "generation" }).update({ attributes }).end();
    startSpan("ended", { type: "generation" }).end({ attributes });
    startActiveSpan("active", { type: "generation" }, () => updateActiveSpan({ attributes }));
  });
  await flush();

  expect(attributes?.["td.prompt.name"]).toBe("other/prompt");
  for (const name of ["in scope", "with parent", "updated", "ended", "active"]) {
    const span = spans.getFinishedSpans().find((finished) => finished.name === name)!;
    expect(
      Object.fromEntries(
        Object.entries(span.attributes).filter(([key]) => key.startsWith("td.prompt.")),
      ),
    ).toEqual({
      "td.prompt.name": "scoped/prompt",
      "td.prompt.version": 12,
      "td.prompt.label": "production",
    });
  }
});

test("td.prompt.variables follows the input capture of the client and of the span", async () => {
  const prompt = await promptOf({
    messages: [{ role: "user", content: "My card is {{card}}." }],
  });
  const compiled = prompt.compile({ card: "4111 1111" });

  setup({ captureInput: false });
  expect(promptAttributesFor(compiled)).toEqual({
    "td.prompt.name": "support/reply",
    "td.prompt.version": 12,
    "td.prompt.label": "production",
  });

  await shutdown();
  const { spans } = setup({
    mask: (value, { key }) => (key === "td.prompt.variables" ? { card: "[masked]" } : value),
  });
  const attributes = promptAttributesFor(compiled);
  expect(attributes?.["td.prompt.variables"]).toBe('{"card":"[masked]"}');

  startSpan("private", { type: "generation", captureInput: false, attributes }).end();
  startSpan("updated", { type: "generation", captureInput: false }).update({ attributes }).end();
  await flush();

  for (const span of spans.getFinishedSpans()) {
    expect(span.attributes["td.prompt.name"]).toBe("support/reply");
    expect(span.attributes["td.prompt.variables"]).toBeUndefined();
  }
});

test("promptAttributesFor reports an error and gives undefined", async () => {
  const errors: Error[] = [];
  setup({ onError: (error) => errors.push(error) });

  const trap = new Proxy([], {
    get() {
      throw new Error("trap");
    },
  });

  expect(promptAttributesFor(trap)).toBeUndefined();
  expect(errors.map((error) => error.message)).toEqual(["trap"]);
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
