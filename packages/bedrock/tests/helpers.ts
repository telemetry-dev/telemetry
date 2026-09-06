import { InMemoryLogRecordExporter } from "@opentelemetry/sdk-logs";
import { InMemorySpanExporter, type ReadableSpan } from "@opentelemetry/sdk-trace-base";

import { init, shutdown } from "@telemetry-dev/sdk";

export function setup(): InMemorySpanExporter {
  const spans = new InMemorySpanExporter();
  init(
    {
      apiKey: "td_live_test",
      serviceName: "svc",
      environment: "test",
      exportMode: "immediate",
      logLevel: "silent",
      fetch: async () => new Response(null, { status: 200 }),
    },
    { spanExporter: spans, logRecordExporter: new InMemoryLogRecordExporter() },
  );
  return spans;
}

export async function teardown(): Promise<void> {
  await shutdown();
}

export function jsonAttr(span: ReadableSpan, key: string): any {
  const value = span.attributes[key];
  if (String(value) !== value) throw new Error(`${key} is not a string attribute`);
  return JSON.parse(value);
}

export class FakeClient {
  readonly calls: any[] = [];
  constructor(private readonly responses: any[]) {}

  async send(command: any, ..._rest: any[]): Promise<any> {
    const input =
      command && Object(command) === command && "input" in command ? command.input : undefined;
    this.calls.push(structuredClone(input));
    const response = this.responses.shift();
    if (response instanceof Error) throw response;
    return response;
  }
}

export async function collect<T>(iterable: AsyncIterable<T>, limit?: number): Promise<T[]> {
  const out: T[] = [];
  for await (const item of iterable) {
    out.push(item);
    if (limit !== undefined && out.length >= limit) break;
  }
  return out;
}

export function bytes(value: any): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(value));
}

export async function* streamOf(
  events: any[],
  throwAt?: number,
  error: Error = Object.assign(new Error("stream boom"), { name: "ModelStreamError" }),
): AsyncGenerator<any> {
  for (let i = 0; i < events.length; i += 1) {
    if (throwAt === i) throw error;
    yield events[i];
  }
  if (throwAt === events.length) {
    throw error;
  }
}
