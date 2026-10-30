import type { Client } from "@modelcontextprotocol/client";

import { SessionAuthProvider } from "./auth.ts";
import { CliError, notSignedIn, toolError, translateError } from "./errors.ts";
import { mcpUrl, type Session } from "./session.ts";
import { withSpinner } from "./terminal.ts";

import packageJson from "../package.json" with { type: "json" };

export const VERSION = packageJson.version;

// The server gives each query 25 seconds, so this only catches a stalled connection.
const REQUEST_TIMEOUT_MS = 40_000;

/**
 * The telemetry.dev MCP server, called as an API. The CLI reads through the same
 * tools and permissions agents get, so a JSON result has the shape the tool documents.
 */
export class Api {
  #client: Promise<Client> | undefined;

  readonly session: Session;

  constructor(session: Session) {
    if (!session.tokens) throw notSignedIn();

    this.session = session;
  }

  #connect() {
    this.#client ??= (async () => {
      // Loaded on first use: the client is most of the CLI's code, and help, status, and
      // usage errors should answer without parsing it.
      const { Client, StreamableHTTPClientTransport } =
        await import("@modelcontextprotocol/client");

      const transport = new StreamableHTTPClientTransport(mcpUrl(this.session.server), {
        authProvider: new SessionAuthProvider(this.session),
        // MCP requests carry the client's own signal and its timeout; this bounds the
        // token refresh and discovery requests, which carry none.
        fetch: (url, init) =>
          fetch(url, { ...init, signal: init?.signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS) }),
      });

      // The modern protocol connects with one request instead of initialize plus its
      // notification, which halves a command's time on a distant connection. Servers
      // without it get the classic handshake.
      const client = new Client(
        { name: "telemetry-cli", version: VERSION },
        { versionNegotiation: { mode: "auto" } },
      );
      await client.connect(transport, { timeout: REQUEST_TIMEOUT_MS });

      return client;
    })();

    return this.#client;
  }

  /** Calls one tool and returns its parsed JSON result. A tool error becomes a CliError. */
  async call<T>(tool: string, args: Record<string, unknown> = {}, label = "Loading"): Promise<T> {
    const result = await withSpinner(
      label,
      (async () => {
        const client = await this.#connect();

        return client.callTool({ name: tool, arguments: args }, { timeout: REQUEST_TIMEOUT_MS });
      })(),
    ).catch(async (error: unknown) => {
      const { OAuthError, UnauthorizedError } = await import("@modelcontextprotocol/client");

      return translateError(error, [OAuthError, UnauthorizedError]);
    });

    const text = result.content.find((block) => block.type === "text")?.text ?? "";

    if (result.isError) throw toolError(text, result._meta);

    try {
      return JSON.parse(text) as T;
    } catch {
      // The server cuts results over its size cap, which leaves the JSON unterminated.
      throw new CliError(
        "The result was too large to read. Narrow it with filters or a smaller --limit.",
        { code: "result_too_large" },
      );
    }
  }

  async close() {
    const client = await this.#client?.catch(() => undefined);
    await client?.close().catch(() => {});
  }
}
