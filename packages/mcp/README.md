# @telemetry-dev/mcp

Model Context Protocol instrumentation for telemetry.dev. It instruments official TypeScript MCP SDK v2 transports and emits request spans using the current OpenTelemetry MCP attribute vocabulary through `@telemetry-dev/sdk`.

## Install

For a client:

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/mcp @modelcontextprotocol/client
```

For a server:

```sh
pnpm add @telemetry-dev/sdk @telemetry-dev/mcp @modelcontextprotocol/server
```

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

## Payload capture

Tool arguments and successful tool results are not captured by default. Enable them with `instrumentMcpTransport(transport, { capturePayloads: true })`. Core `@telemetry-dev/sdk` capture and masking settings still apply.

Trace context propagates by default. OpenTelemetry baggage remains local unless you enable it with `instrumentMcpTransport(transport, { propagateBaggage: true })`.

## Limitations

- Only the stable official TypeScript MCP SDK v2 transport contract is supported. MCP v1 is not supported.
- Metrics and notification processing spans are not emitted.
- OpenTelemetry MCP semantic conventions are currently in Development status, so attribute requirements may change in future releases.
