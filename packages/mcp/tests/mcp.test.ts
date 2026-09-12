import { propagation, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";
import {
  Client,
  InMemoryTransport,
  type JSONRPCMessage,
  StreamableHTTPClientTransport,
} from "@modelcontextprotocol/client";
import { createMcpHandler, McpServer } from "@modelcontextprotocol/server";
import {
  activeContext,
  flush,
  init,
  shutdown,
  startActiveSpan,
  startSpan,
  withContext,
} from "@telemetry-dev/sdk";
import { afterEach, expect, test, vi } from "vitest";

import { instrumentMcpTransport } from "../src/index.ts";

interface TestTransport {
  send(message: unknown, options?: unknown): Promise<void>;
  onmessage?: (message: unknown, extra?: unknown) => void;
  onclose?: () => void;
  sessionId?: string;
  readonly protocolVersion?: string;
  setProtocolVersion?: (version: string) => void;
}

type JsonValue = string | number | boolean | null | JsonValue[] | JsonRecord;

interface JsonRecord {
  [key: string]: JsonValue | undefined;
}

function setup(): InMemorySpanExporter {
  const spans = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "mcp-tests",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter: spans },
  );

  return spans;
}

async function exported(spans: InMemorySpanExporter, count: number): Promise<ReadableSpan[]> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    await flush();

    if (spans.getFinishedSpans().length === count) return spans.getFinishedSpans();
    await Promise.resolve();
  }

  expect(spans.getFinishedSpans()).toHaveLength(count);

  return spans.getFinishedSpans();
}

afterEach(async () => {
  await shutdown();
});

test("instruments both v2 transport directions with propagation and opt-in payloads", async () => {
  const spans = setup();
  const [clientRaw, serverRaw] = InMemoryTransport.createLinkedPair();
  const client = instrumentMcpTransport(clientRaw, { capturePayloads: true });
  const server = instrumentMcpTransport(serverRaw, { capturePayloads: true });
  expect(instrumentMcpTransport(client)).toBe(clientRaw);

  client.onmessage = () => undefined;
  let receivedMeta: JsonRecord | undefined;
  server.onmessage = (message) => {
    receivedMeta = (message as { params?: { _meta?: JsonRecord } }).params?._meta;
    startSpan("handler-work").end();
    void server.send({
      jsonrpc: "2.0",
      id: "call-1",
      result: { content: [{ type: "text", text: "sunny" }] },
    });
    void message;
  };

  await client.start();
  await server.start();

  const request = {
    jsonrpc: "2.0",
    id: "call-1",
    method: "tools/call",
    params: {
      name: "weather",
      arguments: { city: "Paris" },
      _meta: {
        existing: "kept",
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      },
    },
  } satisfies JSONRPCMessage;

  const baggage = propagation.createBaggage({ tenant: { value: "acme" } });
  await withContext(propagation.setBaggage(activeContext(), baggage), () => client.send(request));

  const finished = await exported(spans, 3);
  const clientSpan = finished.find((span) => span.kind === SpanKind.CLIENT)!;
  const serverSpan = finished.find((span) => span.kind === SpanKind.SERVER)!;
  const handlerSpan = finished.find((span) => span.name === "handler-work")!;
  expect(clientSpan.name).toBe("tools/call weather");
  expect(clientSpan.attributes["mcp.method.name"]).toBe("tools/call");
  expect(clientSpan.attributes["jsonrpc.request.id"]).toBe("call-1");
  expect(clientSpan.attributes["gen_ai.operation.name"]).toBe("execute_tool");
  expect(clientSpan.attributes["gen_ai.tool.name"]).toBe("weather");
  expect(clientSpan.attributes["gen_ai.tool.call.arguments"]).toBe('{"city":"Paris"}');
  expect(clientSpan.attributes["gen_ai.tool.call.result"]).toBe(
    '{"content":[{"type":"text","text":"sunny"}]}',
  );
  expect(serverSpan.parentSpanContext?.spanId).toBe(clientSpan.spanContext().spanId);
  expect(handlerSpan.parentSpanContext?.spanId).toBe(serverSpan.spanContext().spanId);
  expect(clientSpan.attributes["mcp.protocol.version"]).toBe("2026-07-28");
  expect(serverSpan.attributes["mcp.protocol.version"]).toBe("2026-07-28");
  expect(receivedMeta?.traceparent).toBeTypeOf("string");
  expect(receivedMeta?.baggage).toBeUndefined();
  expect(request.params._meta).toEqual({
    existing: "kept",
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  });
});

test("propagates baggage only when enabled", async () => {
  const spans = setup();
  let sent: unknown;

  const transport = instrumentMcpTransport<TestTransport>(
    {
      send: async (message: unknown) => {
        sent = message;
      },
    },
    { propagateBaggage: true },
  );

  transport.onmessage = () => undefined;

  const baggage = propagation.createBaggage({ tenant: { value: "acme" } });
  await withContext(propagation.setBaggage(activeContext(), baggage), () =>
    transport.send({ jsonrpc: "2.0", id: 2, method: "ping" } as never),
  );

  const meta = (sent as { params?: { _meta?: JsonRecord } }).params?._meta;
  expect(meta?.traceparent).toBeTypeOf("string");
  expect(meta?.baggage).toBe("tenant=acme");
  transport.onmessage?.({ jsonrpc: "2.0", id: 2, result: {} } as never);
  await exported(spans, 1);
});

test("works through the official v2 client and server connection lifecycle", async () => {
  const spans = setup();
  const [clientRaw, serverRaw] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test-client", version: "1.0.0" });
  const server = new McpServer({ name: "test-server", version: "1.0.0" });
  let preexistingCalls = 0;
  serverRaw.onmessage = function (message) {
    expect(this).toBe(serverRaw);

    if ("method" in message && message.method === "tools/call") preexistingCalls += 1;
  };

  server.registerTool("weather", { description: "Get the weather" }, async () => ({
    content: [{ type: "text", text: "sunny" }],
  }));

  await server.connect(instrumentMcpTransport(serverRaw));
  await client.connect(instrumentMcpTransport(clientRaw));
  await client.callTool({ name: "weather", arguments: {} });
  await flush();

  const toolSpans = spans
    .getFinishedSpans()
    .filter((span) => span.attributes["mcp.method.name"] === "tools/call");

  expect(toolSpans).toHaveLength(2);
  expect(toolSpans.map((span) => span.kind)).toContain(SpanKind.CLIENT);
  expect(toolSpans.map((span) => span.kind)).toContain(SpanKind.SERVER);
  expect(
    toolSpans.every((span) => typeof span.attributes["mcp.protocol.version"] === "string"),
  ).toBe(true);
  expect(preexistingCalls).toBe(1);

  await client.close();
  await server.close();
});

test("completes responses before a per-request HTTP transport closes", async () => {
  const spans = setup();

  const handler = createMcpHandler(
    () => {
      const server = new McpServer({ name: "test-server", version: "1.0.0" });
      server.registerTool("weather", { description: "Get the weather" }, async () => ({
        content: [{ type: "text", text: "sunny" }],
      }));
      const connect = server.connect.bind(server);
      server.connect = (transport) => connect(instrumentMcpTransport(transport));

      return server;
    },
    { legacy: "reject" },
  );

  const response = await handler.fetch(
    new Request("https://mcp.example.test", {
      method: "POST",
      headers: {
        accept: "application/json, text/event-stream",
        "content-type": "application/json",
        "mcp-method": "tools/list",
        "mcp-protocol-version": "2026-07-28",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 36,
        method: "tools/list",
        params: {
          _meta: {
            "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            "io.modelcontextprotocol/clientCapabilities": {},
          },
        },
      }),
    }),
  );

  expect(response.status).toBe(200);
  expect(await response.json()).toMatchObject({
    jsonrpc: "2.0",
    id: 36,
    result: { tools: [{ name: "weather" }] },
  });
  const [span] = await exported(spans, 1);
  expect(span?.attributes["mcp.method.name"]).toBe("tools/list");
  expect(span?.attributes["error.type"]).toBeUndefined();
  expect(span?.status.code).toBe(SpanStatusCode.UNSET);
  await handler.close();
});

test("applies asymmetric JSON-RPC error status rules", async () => {
  const spans = setup();

  const [client, server] = InMemoryTransport.createLinkedPair().map((transport) =>
    instrumentMcpTransport(transport),
  );

  client.onmessage = () => undefined;
  server.onmessage = () => {
    void server.send({
      jsonrpc: "2.0",
      id: 7,
      error: { code: -32602, message: "Invalid params" },
    });
  };

  await client.start();
  await server.start();
  await client.send({ jsonrpc: "2.0", id: 7, method: "prompts/get", params: { name: "review" } });

  const finished = await exported(spans, 2);
  const clientSpan = finished.find((span) => span.kind === SpanKind.CLIENT)!;
  const serverSpan = finished.find((span) => span.kind === SpanKind.SERVER)!;
  expect(clientSpan.attributes["rpc.response.status_code"]).toBe("-32602");
  expect(clientSpan.attributes["error.type"]).toBe("-32602");
  expect(clientSpan.status.code).toBe(SpanStatusCode.ERROR);
  expect(serverSpan.attributes["rpc.response.status_code"]).toBe("-32602");
  expect(serverSpan.attributes["error.type"]).toBeUndefined();
  expect(serverSpan.status.code).toBe(SpanStatusCode.UNSET);
});

test.each([-32021, -32022])(
  "treats MCP v2 caller error %i as a client failure only",
  async (code) => {
    const spans = setup();

    const [client, server] = InMemoryTransport.createLinkedPair().map((transport) =>
      instrumentMcpTransport(transport),
    );

    client.onmessage = () => undefined;
    server.onmessage = () => {
      void server.send({
        jsonrpc: "2.0",
        id: 8,
        error: { code, message: "Client compatibility error" },
      });
    };

    await client.start();
    await server.start();
    await client.send({ jsonrpc: "2.0", id: 8, method: "ping" });

    const finished = await exported(spans, 2);
    const clientSpan = finished.find((span) => span.kind === SpanKind.CLIENT)!;
    const serverSpan = finished.find((span) => span.kind === SpanKind.SERVER)!;
    expect(clientSpan.attributes["error.type"]).toBe(String(code));
    expect(clientSpan.status.code).toBe(SpanStatusCode.ERROR);
    expect(serverSpan.attributes["error.type"]).toBeUndefined();
    expect(serverSpan.status.code).toBe(SpanStatusCode.UNSET);
  },
);

test("marks CallToolResult errors and closes pending requests", async () => {
  const spans = setup();

  const [client, server] = InMemoryTransport.createLinkedPair().map((transport) =>
    instrumentMcpTransport(transport),
  );

  client.onmessage = () => undefined;
  server.onmessage = () => {
    void server.send({
      jsonrpc: "2.0",
      id: 9,
      result: { content: [{ type: "text", text: "failed" }], isError: true },
    });
  };

  await client.start();
  await server.start();
  await client.send({ jsonrpc: "2.0", id: 9, method: "tools/call", params: { name: "broken" } });

  const toolSpans = await exported(spans, 2);
  expect(toolSpans.every((span) => span.attributes["error.type"] === "tool_error")).toBe(true);
  expect(toolSpans.every((span) => span.status.code === SpanStatusCode.ERROR)).toBe(true);

  await shutdown();
  const closeSpans = setup();

  const rawTransport: TestTransport = {
    send: async (_message: unknown) => undefined,
    onmessage: undefined,
    onclose: undefined,
  };

  const transport = instrumentMcpTransport(rawTransport);
  transport.onclose = () => undefined;
  await transport.send({ jsonrpc: "2.0", id: 10, method: "ping" } as never);
  transport.onclose?.();
  const [closed] = await exported(closeSpans, 1);
  expect(closed?.attributes["error.type"]).toBe("connection_error");
  expect(closed?.status.code).toBe(SpanStatusCode.ERROR);
});

test("ends an outgoing request span when send fails", async () => {
  const spans = setup();

  const transport = instrumentMcpTransport({
    send: async (_message: unknown) => {
      throw new TypeError("disconnected");
    },
  });

  await expect(
    transport.send({ jsonrpc: "2.0", id: 11, method: "resources/list" } as never),
  ).rejects.toThrow("disconnected");
  const [span] = await exported(spans, 1);
  expect(span?.attributes["error.type"]).toBe("TypeError");
  expect(span?.status.code).toBe(SpanStatusCode.ERROR);
});

test("records response send failures that race with transport close", async () => {
  const spans = setup();

  const rawTransport: TestTransport = {
    send: async (_message: unknown) => {
      rawTransport.onclose?.();
      throw new TypeError("response disconnected");
    },
  };

  const transport = instrumentMcpTransport(rawTransport);
  transport.onmessage = () => undefined;
  transport.onmessage({ jsonrpc: "2.0", id: 14, method: "resources/list" } as never);

  await expect(
    transport.send({ jsonrpc: "2.0", id: 14, result: { resources: [] } } as never),
  ).rejects.toThrow("response disconnected");
  const [span] = await exported(spans, 1);
  expect(span?.attributes["error.type"]).toBe("TypeError");
  expect(span?.status.code).toBe(SpanStatusCode.ERROR);
});

test("ends an incoming request span when its handler throws", async () => {
  const spans = setup();

  const transport = instrumentMcpTransport<TestTransport>({
    send: async (_message: unknown) => undefined,
  });

  transport.onmessage = () => {
    throw new TypeError("handler failed");
  };

  expect(() =>
    transport.onmessage?.({ jsonrpc: "2.0", id: 12, method: "resources/list" } as never),
  ).toThrow("handler failed");
  const [span] = await exported(spans, 1);
  expect(span?.attributes["error.type"]).toBe("TypeError");
  expect(span?.status.code).toBe(SpanStatusCode.ERROR);
});

test("ends an incoming request span when its async handler rejects", async () => {
  const spans = setup();

  const transport = instrumentMcpTransport<TestTransport>({
    send: async (_message: unknown) => undefined,
  });

  transport.onmessage = async () => {
    await Promise.resolve();
    throw new TypeError("async handler failed");
  };

  const result = transport.onmessage?.({
    jsonrpc: "2.0",
    id: 13,
    method: "resources/list",
  } as never) as unknown;

  await expect(result).rejects.toThrow("async handler failed");
  const [span] = await exported(spans, 1);
  expect(span?.attributes["error.type"]).toBe("TypeError");
  expect(span?.status.code).toBe(SpanStatusCode.ERROR);
});

test("observes protocol versions without owning transport negotiation", async () => {
  const spans = setup();
  const setProtocolVersion = vi.fn();

  const rawTransport: TestTransport = {
    send: async (_message: unknown) => undefined,
    setProtocolVersion,
  };

  const transport = instrumentMcpTransport(rawTransport);
  transport.onmessage = () => undefined;

  await transport.send({
    jsonrpc: "2.0",
    id: 20,
    method: "initialize",
    params: { protocolVersion: "2025-11-25" },
  } as never);
  transport.onmessage?.({
    jsonrpc: "2.0",
    id: 20,
    result: { protocolVersion: "2025-06-18" },
  } as never);

  const [span] = await exported(spans, 1);
  expect(span?.attributes["mcp.protocol.version"]).toBe("2025-06-18");
  expect(setProtocolVersion).not.toHaveBeenCalled();

  transport.setProtocolVersion?.("2025-03-26");
  Object.defineProperty(transport, "protocolVersion", { value: "stale", configurable: true });
  await transport.send({ jsonrpc: "2.0", id: 21, method: "ping" } as never);
  transport.onmessage?.({ jsonrpc: "2.0", id: 21, result: {} } as never);

  const versions = (await exported(spans, 2)).map(
    (finished) => finished.attributes["mcp.protocol.version"],
  );

  expect(versions).toContain("2025-03-26");
  expect(setProtocolVersion).toHaveBeenCalledOnce();
});

test("completes pending spans exactly once on abort, cancellation, and stream end", async () => {
  const spans = setup();

  const sentOptions = new Map<
    number,
    { requestSignal?: AbortSignal; onRequestStreamEnd?: () => void }
  >();

  const originalStreamEnd = vi.fn();

  const rawTransport: TestTransport = {
    send: async (message: unknown, options?: unknown) => {
      const id = (message as { id?: unknown }).id;

      if (typeof id === "number" && options !== undefined) {
        sentOptions.set(
          id,
          options as { requestSignal?: AbortSignal; onRequestStreamEnd?: () => void },
        );
      }
    },
  };

  const transport = instrumentMcpTransport(rawTransport);
  transport.onmessage = () => undefined;

  const controller = new AbortController();
  const abortOptions = { requestSignal: controller.signal };
  await transport.send({ jsonrpc: "2.0", id: 30, method: "ping" } as never, abortOptions as never);
  controller.abort();

  await transport.send({ jsonrpc: "2.0", id: 31, method: "ping" } as never);
  await transport.send({
    jsonrpc: "2.0",
    method: "notifications/cancelled",
    params: { requestId: 31 },
  } as never);

  const streamOptions = { onRequestStreamEnd: originalStreamEnd };
  await transport.send({ jsonrpc: "2.0", id: 32, method: "ping" } as never, streamOptions as never);
  const decoratedStreamEnd = sentOptions.get(32)?.onRequestStreamEnd;
  expect(decoratedStreamEnd).toBeTypeOf("function");
  (decoratedStreamEnd as () => void)();

  transport.onmessage?.({ jsonrpc: "2.0", id: 33, method: "ping" } as never);
  transport.onmessage?.({
    jsonrpc: "2.0",
    method: "notifications/cancelled",
    params: { requestId: 33 },
  } as never);

  transport.onmessage?.({ jsonrpc: "2.0", id: 30, result: {} } as never);
  transport.onmessage?.({ jsonrpc: "2.0", id: 31, result: {} } as never);
  transport.onmessage?.({ jsonrpc: "2.0", id: 32, result: {} } as never);
  await transport.send({ jsonrpc: "2.0", id: 33, result: {} } as never);

  const finished = await exported(spans, 4);
  expect(finished.filter((span) => span.attributes["error.type"] === "cancelled")).toHaveLength(3);
  expect(
    finished.filter((span) => span.attributes["error.type"] === "connection_error"),
  ).toHaveLength(1);
  expect(originalStreamEnd).toHaveBeenCalledOnce();
  expect(streamOptions.onRequestStreamEnd).toBe(originalStreamEnd);
  expect(abortOptions).toEqual({ requestSignal: controller.signal });
});

test("stale stream callbacks cannot complete a newer request with a reused id", async () => {
  const spans = setup();
  let firstStreamEnd: (() => void) | undefined;

  const rawTransport: TestTransport = {
    send: async (message: unknown, options?: unknown) => {
      const id = (message as { id?: unknown }).id;

      if (id === 34 && firstStreamEnd === undefined) {
        firstStreamEnd = (options as { onRequestStreamEnd?: () => void }).onRequestStreamEnd;
      }
    },
  };

  const transport = instrumentMcpTransport(rawTransport);
  transport.onmessage = () => undefined;

  await transport.send({ jsonrpc: "2.0", id: 34, method: "ping" } as never, {} as never);
  transport.onmessage?.({ jsonrpc: "2.0", id: 34, result: {} } as never);
  await transport.send({ jsonrpc: "2.0", id: 34, method: "tools/list" } as never);
  firstStreamEnd?.();
  expect(spans.getFinishedSpans()).toHaveLength(1);
  transport.onmessage?.({ jsonrpc: "2.0", id: 34, result: {} } as never);

  const finished = await exported(spans, 2);
  expect(finished.every((span) => span.attributes["error.type"] === undefined)).toBe(true);
});

test("instruments mixed batched requests without mutating untouched messages", async () => {
  const spans = setup();
  let sent: unknown;

  const transport = instrumentMcpTransport<TestTransport>(
    {
      send: async (message: unknown) => {
        sent = message;
      },
    },
    { capturePayloads: true },
  );

  transport.onmessage = () => undefined;

  const first = {
    jsonrpc: "2.0",
    id: 60,
    method: "tools/call",
    params: { name: "weather", arguments: { city: "Paris" }, _meta: { existing: "kept" } },
  };

  const notification = {
    jsonrpc: "2.0",
    method: "notifications/progress",
    params: { progressToken: "one", progress: 1 },
  };

  const second = {
    jsonrpc: "2.0",
    id: "61",
    method: "prompts/get",
    params: { name: "review" },
  };

  const batch = [first, notification, second];

  await transport.send(batch as never);

  const sentBatch = sent as JsonRecord[];
  const firstParams = sentBatch[0]?.params as { _meta?: JsonRecord } | undefined;
  const secondParams = sentBatch[2]?.params as { _meta?: JsonRecord } | undefined;
  const firstMeta = firstParams?._meta;
  const secondMeta = secondParams?._meta;
  expect(sentBatch).not.toBe(batch);
  expect(sentBatch[0]).not.toBe(first);
  expect(sentBatch[1]).toBe(notification);
  expect(sentBatch[2]).not.toBe(second);
  expect(firstMeta?.existing).toBe("kept");
  expect(firstMeta?.traceparent).toBeTypeOf("string");
  expect(secondMeta?.traceparent).toBeTypeOf("string");
  expect(firstMeta?.traceparent).not.toBe(secondMeta?.traceparent);
  expect(first.params._meta).toEqual({ existing: "kept" });
  expect(second.params).toEqual({ name: "review" });

  transport.onmessage?.({ jsonrpc: "2.0", id: "61", result: { description: "Review" } });
  transport.onmessage?.({
    jsonrpc: "2.0",
    id: 60,
    result: { content: [{ type: "text", text: "sunny" }] },
  });

  const finished = await exported(spans, 2);
  expect(finished.map((span) => span.name)).toEqual(
    expect.arrayContaining(["tools/call weather", "prompts/get review"]),
  );
  expect(
    finished.find((span) => span.name === "tools/call weather")?.attributes[
      "gen_ai.tool.call.result"
    ],
  ).toBe('{"content":[{"type":"text","text":"sunny"}]}');
});

test("instruments both directions when a transport delivers batches intact", async () => {
  const spans = setup();
  let clientRaw: TestTransport;
  let serverRaw: TestTransport;
  clientRaw = {
    send: async (message: unknown) => {
      serverRaw.onmessage?.(message);
    },
  };
  serverRaw = {
    send: async (message: unknown) => {
      clientRaw.onmessage?.(message);
    },
  };
  const client = instrumentMcpTransport(clientRaw);
  const server = instrumentMcpTransport(serverRaw);
  client.onmessage = () => undefined;

  const serverHandler = vi.fn((message: unknown) => {
    const requests = message as Array<{ id: string | number }>;
    void server.send(requests.map((request) => ({ jsonrpc: "2.0", id: request.id, result: {} })));
  });

  server.onmessage = serverHandler;

  const batch = [
    { jsonrpc: "2.0", id: 71, method: "ping" },
    { jsonrpc: "2.0", id: "72", method: "tools/list" },
  ];

  await client.send(batch as never);

  const finished = await exported(spans, 4);
  expect(serverHandler).toHaveBeenCalledOnce();
  expect(serverHandler.mock.calls[0]?.[0]).toBeInstanceOf(Array);

  for (const id of ["71", "72"]) {
    const matching = finished.filter((span) => span.attributes["jsonrpc.request.id"] === id);
    const clientSpan = matching.find((span) => span.kind === SpanKind.CLIENT);
    const serverSpan = matching.find((span) => span.kind === SpanKind.SERVER);
    expect(matching).toHaveLength(2);
    expect(serverSpan?.parentSpanContext?.spanId).toBe(clientSpan?.spanContext().spanId);
  }
});

test("skips duplicate batch ids while distinguishing numeric and string ids", async () => {
  const spans = setup();
  let sent: unknown;

  const transport = instrumentMcpTransport<TestTransport>({
    send: async (message: unknown) => {
      sent = message;
    },
  });

  transport.onmessage = () => undefined;
  const first = { jsonrpc: "2.0", id: 62, method: "ping" };
  const duplicate = { jsonrpc: "2.0", id: 62, method: "tools/list" };
  const distinct = { jsonrpc: "2.0", id: "62", method: "prompts/list" };
  const batch = [first, duplicate, distinct];

  await transport.send(batch as never);

  const sentBatch = sent as JsonRecord[];
  expect(sentBatch[0]).toBe(first);
  expect(sentBatch[1]).toBe(duplicate);
  expect(sentBatch[2]).not.toBe(distinct);
  const distinctParams = sentBatch[2]?.params as { _meta?: JsonRecord } | undefined;
  expect(distinctParams?._meta?.traceparent).toBeTypeOf("string");
  transport.onmessage?.({ jsonrpc: "2.0", id: 62, result: {} });
  expect(spans.getFinishedSpans()).toHaveLength(0);
  transport.onmessage?.({ jsonrpc: "2.0", id: "62", result: {} });

  const [finished] = await exported(spans, 1);
  expect(finished?.attributes["mcp.method.name"]).toBe("prompts/list");
});

test("applies batched cancellations in message order", async () => {
  const spans = setup();
  let sent: unknown;

  const transport = instrumentMcpTransport<TestTransport>({
    send: async (message: unknown) => {
      sent = message;
    },
  });

  transport.onmessage = () => undefined;
  await transport.send({ jsonrpc: "2.0", id: 63, method: "ping" } as never);

  const cancellation = {
    jsonrpc: "2.0",
    method: "notifications/cancelled",
    params: { requestId: 63 },
  };

  const replacement = { jsonrpc: "2.0", id: 63, method: "tools/list" };

  await transport.send([cancellation, replacement] as never);

  const sentBatch = sent as JsonRecord[];
  expect(sentBatch[0]).toBe(cancellation);
  expect(sentBatch[1]).not.toBe(replacement);
  transport.onmessage?.({ jsonrpc: "2.0", id: 63, result: {} });
  const finished = await exported(spans, 2);
  expect(
    finished.find((span) => span.attributes["mcp.method.name"] === "ping")?.attributes[
      "error.type"
    ],
  ).toBe("cancelled");
  expect(
    finished.find((span) => span.attributes["mcp.method.name"] === "tools/list")?.attributes[
      "error.type"
    ],
  ).toBeUndefined();
});

test.each([73, 74])("does not trace id-bearing cancellation controls with id %i", async (id) => {
  const spans = setup();
  const originalSend = vi.fn(async (_message: unknown) => undefined);
  const transport = instrumentMcpTransport<TestTransport>({ send: originalSend });
  transport.onmessage = () => undefined;
  await transport.send({ jsonrpc: "2.0", id: 73, method: "ping" } as never);

  const cancellation = {
    jsonrpc: "2.0",
    id,
    method: "notifications/cancelled",
    params: { requestId: 73 },
  };

  await transport.send(cancellation as never);
  transport.onclose?.();

  const [finished] = await exported(spans, 1);
  expect(finished?.attributes["mcp.method.name"]).toBe("ping");
  expect(finished?.attributes["error.type"]).toBe("cancelled");
  expect(originalSend.mock.calls[1]?.[0]).toBe(cancellation);
});

test.each([75, 76])(
  "does not trace received id-bearing cancellation controls with id %i",
  async (id) => {
    const spans = setup();

    const transport = instrumentMcpTransport<TestTransport>({
      send: async (_message: unknown) => undefined,
    });

    transport.onmessage = () => undefined;
    transport.onmessage?.({ jsonrpc: "2.0", id: 75, method: "ping" });
    transport.onmessage?.({
      jsonrpc: "2.0",
      id,
      method: "notifications/cancelled",
      params: { requestId: 75 },
    });
    transport.onclose?.();

    const [finished] = await exported(spans, 1);
    expect(finished?.attributes["mcp.method.name"]).toBe("ping");
    expect(finished?.attributes["error.type"]).toBe("cancelled");
  },
);

test("completes every batched request when send fails", async () => {
  const spans = setup();

  const transport = instrumentMcpTransport<TestTransport>({
    send: async (_message: unknown) => {
      throw new TypeError("batch disconnected");
    },
  });

  await expect(
    transport.send([
      { jsonrpc: "2.0", id: 64, method: "ping" },
      { jsonrpc: "2.0", id: 65, method: "resources/list" },
    ] as never),
  ).rejects.toThrow("batch disconnected");

  const finished = await exported(spans, 2);
  expect(finished.every((span) => span.attributes["error.type"] === "TypeError")).toBe(true);
});

test("passes uninstrumented batches through exactly", async () => {
  const spans = setup();
  const originalSend = vi.fn(async (_message: unknown, _options?: unknown) => undefined);
  const transport = instrumentMcpTransport<TestTransport>({ send: originalSend });
  const resumeBatch = [{ jsonrpc: "2.0", id: 66, method: "tools/list" }];
  const resumeOptions = { resumptionToken: "event-2" };

  await transport.send(resumeBatch as never, resumeOptions as never);

  const first = { jsonrpc: "2.0", id: 67, method: "ping" };
  const failing = { jsonrpc: "2.0", id: 68 };
  Object.defineProperty(failing, "method", {
    get: () => {
      throw new Error("batch metadata failed");
    },
  });
  const failingBatch = [first, failing];
  const failingOptions = { headers: { "x-test": "one" } };
  await transport.send(failingBatch as never, failingOptions as never);

  expect(originalSend.mock.calls[0]?.[0]).toBe(resumeBatch);
  expect(originalSend.mock.calls[0]?.[1]).toBe(resumeOptions);
  expect(originalSend.mock.calls[1]?.[0]).toBe(failingBatch);
  expect(originalSend.mock.calls[1]?.[1]).toBe(failingOptions);
  await flush();
  expect(spans.getFinishedSpans()).toHaveLength(0);
});

test("completes incoming requests from a batched response send", async () => {
  const spans = setup();
  const originalSend = vi.fn(async (_message: unknown) => undefined);
  const transport = instrumentMcpTransport<TestTransport>({ send: originalSend });
  transport.onmessage = () => undefined;
  transport.onmessage?.({ jsonrpc: "2.0", id: 69, method: "ping" });
  transport.onmessage?.({ jsonrpc: "2.0", id: 70, method: "tools/list" });

  const responses = [
    { jsonrpc: "2.0", id: 70, result: { tools: [] } },
    { jsonrpc: "2.0", id: 69, result: {} },
  ];

  await transport.send(responses as never);

  expect(originalSend.mock.calls[0]?.[0]).toBe(responses);
  const finished = await exported(spans, 2);
  expect(finished.every((span) => span.kind === SpanKind.SERVER)).toBe(true);
  expect(finished.every((span) => span.attributes["error.type"] === undefined)).toBe(true);
});

test("does not trace resume-only Streamable HTTP sends", async () => {
  const spans = setup();

  const fetch = vi.fn(
    async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(null, { status: 405, statusText: "Method Not Allowed" }),
  );

  const rawTransport = new StreamableHTTPClientTransport(new URL("https://mcp.example.test"), {
    fetch,
  });

  const transport = instrumentMcpTransport(rawTransport);
  transport.onerror = () => undefined;
  await transport.start();

  await transport.send({ jsonrpc: "2.0", id: 35, method: "tools/list" } as never, {
    resumptionToken: "event-1",
  });
  await vi.waitFor(() => expect(fetch).toHaveBeenCalledOnce());
  await transport.close();
  await flush();

  expect(fetch.mock.calls[0]?.[1]?.method).toBe("GET");
  expect(spans.getFinishedSpans()).toHaveLength(0);
});

test("parents inbound work to remote context and links a distinct ambient span", async () => {
  const spans = setup();
  const rawTransport: TestTransport = { send: async (_message: unknown) => undefined };
  const transport = instrumentMcpTransport(rawTransport);
  transport.onmessage = (message: unknown) => {
    void transport.send({
      jsonrpc: "2.0",
      id: (message as { id: number }).id,
      result: {},
    } as never);
  };

  const remoteTraceId = "0af7651916cd43dd8448eb211c80319c";
  const remoteSpanId = "b7ad6b7169203331";

  startActiveSpan("http-server", (ambient) => {
    transport.onmessage?.({
      jsonrpc: "2.0",
      id: 40,
      method: "tools/list",
      params: { _meta: { traceparent: `00-${remoteTraceId}-${remoteSpanId}-01` } },
    } as never);
    void ambient;
  });

  const finished = await exported(spans, 2);
  const ambient = finished.find((span) => span.name === "http-server")!;
  const server = finished.find((span) => span.kind === SpanKind.SERVER)!;
  expect(server.spanContext().traceId).toBe(remoteTraceId);
  expect(server.parentSpanContext?.spanId).toBe(remoteSpanId);
  expect(server.links).toEqual([{ context: ambient.spanContext() }]);
});

test("parents unpropagated inbound work to the ambient context", async () => {
  const spans = setup();

  const transport = instrumentMcpTransport<TestTransport>({
    send: async (_message: unknown) => undefined,
  });

  transport.onmessage = (message: unknown) => {
    void transport.send({
      jsonrpc: "2.0",
      id: (message as { id: number }).id,
      result: {},
    } as never);
  };

  startActiveSpan("http-server", () => {
    transport.onmessage?.({
      jsonrpc: "2.0",
      id: 41,
      method: "tools/list",
    } as never);
  });

  const finished = await exported(spans, 2);
  const ambient = finished.find((span) => span.name === "http-server")!;
  const server = finished.find((span) => span.kind === SpanKind.SERVER)!;
  expect(server.parentSpanContext?.spanId).toBe(ambient.spanContext().spanId);
});

test("fails open when observable message metadata throws", async () => {
  setup();
  const originalSend = vi.fn(async (_message: unknown) => undefined);
  const transport = instrumentMcpTransport({ send: originalSend });
  const message = { jsonrpc: "2.0", id: 50, params: {} };
  Object.defineProperty(message, "method", {
    get: () => {
      throw new Error("observable metadata failed");
    },
  });

  await expect(transport.send(message as never)).resolves.toBeUndefined();
  expect(originalSend).toHaveBeenCalledOnce();
  expect(originalSend.mock.calls[0]?.[0]).toBe(message);
});

test("does not instrument null ids outside the MCP v2 RequestId contract", async () => {
  const spans = setup();
  const originalSend = vi.fn(async (_message: unknown) => undefined);
  const transport = instrumentMcpTransport<TestTransport>({ send: originalSend });
  transport.onmessage = () => undefined;
  const request = { jsonrpc: "2.0", id: null, method: "ping" };

  await transport.send(request as never);
  transport.onmessage?.({ jsonrpc: "2.0", id: null, result: {} });
  await flush();

  expect(originalSend.mock.calls[0]?.[0]).toBe(request);
  expect(spans.getFinishedSpans()).toHaveLength(0);
});

test("closes pending spans even when no onclose handler was assigned", async () => {
  const spans = setup();

  const transport = instrumentMcpTransport<TestTransport>({
    send: async (_message: unknown) => undefined,
  });

  await transport.send({ jsonrpc: "2.0", id: 51, method: "ping" } as never);
  transport.onclose?.();

  const [span] = await exported(spans, 1);
  expect(span?.attributes["error.type"]).toBe("connection_error");
});
