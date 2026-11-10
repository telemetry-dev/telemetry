# @telemetry-dev/mcp

Model Context Protocol instrumentation for telemetry.dev. It instruments official TypeScript MCP SDK v2 transports and emits request spans using the current OpenTelemetry MCP attribute vocabulary through `@telemetry-dev/sdk`.

Docs: https://telemetry.dev/docs/integrations/mcp

## Install

For a client:

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/mcp @modelcontextprotocol/client
```

For a server:

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/mcp @modelcontextprotocol/server
```

Requires `@modelcontextprotocol/client` or `@modelcontextprotocol/server` 2.x, `@opentelemetry/api` 1.9 or a later 1.x release, and `@telemetry-dev/sdk`, which needs Node.js 20.19.0 or later.

## Client

Set `TELEMETRY_DEV_API_KEY`, then instrument the transport before connecting it:

```ts
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { instrumentMcpTransport } from "@telemetry-dev/mcp";
import { init } from "@telemetry-dev/sdk";

init({ serviceName: "mcp-client" });

const client = new Client({ name: "example-client", version: "1.0.0" });
const transport = instrumentMcpTransport(
  new StdioClientTransport({ command: "example-mcp-server" }),
);
await client.connect(transport);
```

Versions of `@telemetry-dev/sdk` before 0.2.0 sent data to `https://ingest.telemetry.dev` by
default. That hostname reaches the same service and stays supported: if a firewall allows only it,
set `TELEMETRY_DEV_BASE_URL` to it.

## Server

```ts
import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { instrumentMcpTransport } from "@telemetry-dev/mcp";
import { init } from "@telemetry-dev/sdk";

init({ serviceName: "mcp-server" });

const server = new McpServer({ name: "example-server", version: "1.0.0" });
const transport = instrumentMcpTransport(new StdioServerTransport());
await server.connect(transport);
```

`instrumentMcpTransport(transport)` mutates and returns the same transport and is idempotent.

Request spans also carry `mcp.client.name` and `mcp.client.version`, from the `clientInfo` the client reports, each cut to 128 characters. Since protocol 2026-07-28, every request carries it. On older protocol versions, only `initialize` does, and later requests on the same transport reuse it. The OpenTelemetry MCP conventions don't define these two attributes yet.

## Payload capture

`tools/call` spans record the tool arguments and the tool result, including a result with `isError: true`. Core `@telemetry-dev/sdk` settings still apply: `captureInput: false` drops the arguments, `captureOutput: false` drops the result, and `mask` runs on both. To leave tool payloads out while the core SDK captures other content, pass `instrumentMcpTransport(transport, { capturePayloads: false })`. Error codes and the JSON-RPC error message are recorded either way; `captureOutput: false` and `mask` do not apply to them.

Trace context propagates by default. OpenTelemetry baggage remains local unless you enable it with `instrumentMcpTransport(transport, { propagateBaggage: true })`.

## Limitations

- Only the stable official TypeScript MCP SDK v2 transport contract is supported. MCP v1 is not supported.
- Metrics and notification processing spans are not emitted.
- OpenTelemetry MCP semantic conventions are currently in Development status, so attribute requirements may change in future releases.

## License

MIT. See [LICENSE](https://github.com/telemetry-dev/telemetry/blob/main/packages/mcp/LICENSE).
