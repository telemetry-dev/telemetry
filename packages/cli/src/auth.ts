import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createServer, type Server } from "node:http";
import { createInterface } from "node:readline/promises";
import { setTimeout as sleep } from "node:timers/promises";

import type {
  OAuthClientMetadata,
  OAuthDiscoveryState,
  OAuthClientProvider,
  StoredOAuthClientInformation,
  StoredOAuthTokens,
} from "@modelcontextprotocol/client";

import { CliError, EXIT } from "./errors.ts";
import { loadSession, mcpUrl, type Session, updateSession } from "./session.ts";
import { oneLine } from "./terminal.ts";

export const PAYLOADS_SCOPE = "telemetry:payloads";

// telemetry:read is the MCP server's required scope. The consent page leaves payloads
// unchecked unless the person ticks it, openid and email name the account in
// `telemetry status`, and offline_access keeps the CLI signed in past the hour an
// access token lasts.
const SCOPES = ["openid", "email", "offline_access", "telemetry:read", PAYLOADS_SCOPE].join(" ");

const CALLBACK_PATH = "/callback";
const LOGIN_TIMEOUT_MS = 10 * 60 * 1000;
const OAUTH_TIMEOUT_MS = 15_000;

/**
 * For OAuth requests the CLI makes, so a stalled server can't hold `login` or `logout`
 * open. The SDK follows only same-origin redirects with it; the CLI's own requests carry
 * a token and follow none.
 */
const oauthFetch = (url: string | URL, init?: RequestInit) =>
  fetch(url, { ...init, signal: init?.signal ?? AbortSignal.timeout(OAUTH_TIMEOUT_MS) });

const clientMetadata = (redirectUrl: string): OAuthClientMetadata => ({
  client_name: "telemetry.dev CLI",
  client_uri: "https://telemetry.dev",
  redirect_uris: [redirectUrl],
  grant_types: ["authorization_code", "refresh_token"],
  response_types: ["code"],
  token_endpoint_auth_method: "none",
  scope: SCOPES,
});

type LoginHooks = {
  readonly verifier?: string;
  readonly redirectUrl: string;
  readonly state: string;
  readonly open: (url: URL) => void;
};

// The token that names a sign-in: the refresh token when there is one.
const tokenKey = (tokens: StoredOAuthTokens | undefined) =>
  tokens?.refresh_token ?? tokens?.access_token;

// How long a refused refresh waits for the command that won the rotation to save.
const REFRESH_RACE_WAIT_MS = 3000;
const REFRESH_RACE_POLL_MS = 100;

/**
 * Keeps the session's OAuth state on disk for the MCP client. Outside `telemetry login`
 * it can only refresh: when the server wants a new authorization, the command stops and
 * asks the person to sign in again. During a sign-in nothing is saved until it finishes.
 */
export class SessionAuthProvider implements OAuthClientProvider {
  #verifier = "";

  session: Session;
  readonly #login: LoginHooks | undefined;

  constructor(session: Session, login?: LoginHooks) {
    this.session = session;
    this.#login = login;
  }

  async #save(change: (saved: Session) => Session) {
    if (!this.#login) await updateSession(this.session.server, change);
  }

  // Always set: the SDK treats a provider without one as a machine client and never
  // refreshes. Outside a sign-in nothing is redirected to it.
  get redirectUrl() {
    return this.#login?.redirectUrl ?? `http://127.0.0.1${CALLBACK_PATH}`;
  }

  get clientMetadata() {
    return clientMetadata(this.redirectUrl);
  }

  state() {
    return this.#login?.state ?? randomBytes(16).toString("base64url");
  }

  clientInformation() {
    return this.session.client;
  }

  async saveClientInformation(client: StoredOAuthClientInformation) {
    this.session = { ...this.session, client };
    await this.#save((saved) => ({ ...saved, client }));
  }

  /** Tokens saved by another command since this one loaded the session, if any. */
  async #newerTokens() {
    const { tokens } = await loadSession(this.session.server);

    return tokens && tokenKey(tokens) !== tokenKey(this.session.tokens) ? tokens : undefined;
  }

  // Read again before each use: the server revokes every token of a sign-in when it sees
  // a rotated refresh token again, so a long command must not refresh with the one it
  // started with after another command replaced it.
  async tokens() {
    if (!this.#login) {
      const newer = await this.#newerTokens();

      if (newer) this.session = { ...this.session, tokens: newer };
    }

    return this.session.tokens;
  }

  async saveTokens(tokens: StoredOAuthTokens) {
    const previous = this.session.tokens;
    // A refresh response can omit the refresh token, which means keep the old one.
    const refresh = tokens.refresh_token ?? previous?.refresh_token;
    const next = refresh ? { ...tokens, refresh_token: refresh } : tokens;

    this.session = { ...this.session, tokens: next };
    // When another command refreshed, signed in, or signed out after this one loaded the
    // session, the file already holds the current sign-in; this one only finishes its run.
    await this.#save((saved) =>
      tokenKey(saved.tokens) === tokenKey(previous) ? { ...saved, tokens: next } : saved,
    );
  }

  redirectToAuthorization(url: URL) {
    if (!this.#login) throw sessionExpired();

    this.#login.open(url);
  }

  saveCodeVerifier(verifier: string) {
    this.#verifier = verifier;
  }

  codeVerifier() {
    return this.#verifier || (this.#login?.verifier ?? "");
  }

  // Kept with the session, so a refresh skips discovery and a sign-in finished with
  // --finish checks the code came back from the same authorization server.
  async saveDiscoveryState(discovery: OAuthDiscoveryState) {
    this.session = { ...this.session, discovery };
    await this.#save((saved) => ({ ...saved, discovery }));
  }

  discoveryState() {
    return this.session.discovery;
  }

  async invalidateCredentials(scope: "all" | "client" | "tokens" | "verifier" | "discovery") {
    if (scope === "verifier") return;

    const drops = (field: "client" | "tokens" | "discovery") => scope === "all" || scope === field;
    const forget = (session: Session): Session => {
      const next = { ...session };

      if (drops("client")) delete next.client;

      if (drops("tokens")) delete next.tokens;

      if (drops("discovery")) delete next.discovery;

      return next;
    };

    if (this.#login) {
      this.session = forget(this.session);

      return;
    }

    // Tokens are only dropped in memory. A refused refresh most often means another
    // command refreshed with the same token and is about to save the new one; deleting
    // them here would make that command think the person signed out and discard its
    // tokens. When the new tokens arrive, the SDK retries with them.
    let newer: StoredOAuthTokens | undefined;

    for (let waited = 0; drops("tokens") && waited < REFRESH_RACE_WAIT_MS; ) {
      newer = await this.#newerTokens();

      if (newer) break;

      await sleep(REFRESH_RACE_POLL_MS);
      waited += REFRESH_RACE_POLL_MS;
    }

    this.session = { ...forget(this.session), ...(newer ? { tokens: newer } : {}) };

    if (drops("client") || drops("discovery")) {
      await this.#save((saved) => ({
        ...forget(saved),
        ...(saved.tokens ? { tokens: saved.tokens } : {}),
      }));
    }
  }
}

const sessionExpired = () =>
  new CliError(
    "Your sign-in has expired or was disconnected. Run `telemetry login` again.",
    EXIT.auth,
  );

export const grantedScopes = (session: Session) =>
  new Set((session.tokens?.scope ?? "").split(" ").filter(Boolean));

/**
 * The account's email from the OpenID userinfo endpoint, for `telemetry status`; the
 * telemetry.dev ID token names only the subject. Shown, never trusted for access, and a
 * failed lookup only leaves the email out.
 */
const fetchEmail = async (session: Session) => {
  const metadata = session.discovery?.authorizationServerMetadata as
    | { userinfo_endpoint?: unknown }
    | undefined;

  const endpoint = metadata?.userinfo_endpoint;
  const token = session.tokens?.access_token;

  if (typeof endpoint !== "string" || !token) return undefined;

  try {
    const response = await oauthFetch(endpoint, {
      headers: { Authorization: `Bearer ${token}` },
      redirect: "error",
    });

    if (!response.ok) return undefined;

    const { email } = (await response.json()) as { email?: unknown };

    return typeof email === "string" ? oneLine(email) : undefined;
  } catch {
    return undefined;
  }
};

type Callback = { readonly params: URLSearchParams };

const CALLBACK_PAGE = (title: string, detail: string) => `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title} · telemetry.dev</title>
<style>
:root{color-scheme:light dark;font-family:ui-sans-serif,system-ui,-apple-system,sans-serif}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:Canvas;color:CanvasText}
main{display:flex;flex-direction:column;align-items:center;gap:12px;text-align:center;padding:24px}
svg{width:40px;height:40px}h1{font-size:20px;font-weight:600;margin:8px 0 0}p{margin:0;opacity:.65;font-size:15px}
</style></head><body><main>
<svg viewBox="0 0 114.6 115" fill="currentColor" aria-hidden="true"><path d="M28.3 17.8C26.8 17.8 26.1 15.9 27.4 14.9C34.3 10.6 43.7 5.8 57.3 5.8C70.9 5.8 80.3 10.6 87.2 14.9C88.5 15.9 87.8 17.8 86.3 17.8Z"/><path d="M19.8 22.9H94.8Q96.3 22.9 97.04 24.2Q100.1 27.95 101.76 32.5Q102.5 33.8 101 33.8H77.3C71.3 33.8 68.6 32 65.8 31C63 30 61.4 29.2 57.3 29.2C53.2 29.2 51.6 30 48.8 31C46 32 43.3 33.8 37.3 33.8H13.6Q12.1 33.8 12.84 32.5Q14.5 27.95 17.56 24.2Q18.3 22.9 19.8 22.9Z"/><path d="M34.2 38.8C35.6 38.8 36.3 40.3 35.6 41.4C34.7 42.6 33.4 45.4 32.9 46.9C32.4 48.3 30.8 49.3 29.9 49.3H7.8C7 49.3 6.3 48.5 6.4 47.2C6.7 45.2 7.4 42.1 8.1 40C8.4 39.35 9.1 38.8 9.55 38.8Z"/><path d="M80.4 38.8H105.05C105.75 38.8 106.2 39.35 106.5 40C107.2 42.1 107.9 45.2 108.2 47.2C108.3 48.5 107.6 49.3 106.8 49.3H84.7C83.8 49.3 82.2 48.3 81.7 46.9C81.2 45.4 79.9 42.6 79 41.4C78.3 40.3 79 38.8 80.4 38.8Z"/><path d="M6.8 54.5H107.8Q109.3 54.5 109.3 56L109.3 61.9Q109.3 63.4 107.8 63.4H6.8Q5.3 63.4 5.3 61.9L5.3 56Q5.3 54.5 6.8 54.5Z"/><path d="M8.3 68.3H106.3C107.5 68.3 107.97 69.36 107.6 70.5L106.25 74.7C105.88 75.84 106.25 76.1 105.05 76.1H9.55C8.35 76.1 8.72 75.84 8.35 74.7L7 70.5C6.63 69.36 7.1 68.3 8.3 68.3Z"/><path d="M14 81.5H100.6C101.8 81.5 102.56 83.35 101.9 84.35L99.55 87.9C98.89 88.9 99.65 88.7 98.45 88.7H16.15C14.95 88.7 15.71 88.9 15.05 87.9L12.7 84.35C12.04 83.35 12.8 81.5 14 81.5Z"/><path d="M24.2 93.5H90.4C91.6 93.5 92.27 95.83 91.35 96.6L88.7 98.8C87.78 99.57 88.9 99.3 87.7 99.3H26.9C25.7 99.3 26.82 99.57 25.9 98.8L23.25 96.6C22.33 95.83 23 93.5 24.2 93.5Z"/><path d="M45.6 104.5H69C70.7 104.5 71.2 107.2 69.3 107.5C66 108.3 61.9 108.9 57.3 108.9C52.7 108.9 48.6 108.3 45.3 107.5C43.4 107.2 43.9 104.5 45.6 104.5Z"/></svg>
<h1>${title}</h1><p>${detail}</p></main></body></html>`;

/** Listens on a random loopback port for the browser's redirect back from consent. */
const listenForCallback = async (state: string) => {
  let deliver: (callback: Callback) => void = () => {};
  const received = new Promise<Callback>((resolve) => {
    deliver = resolve;
  });

  const server: Server = createServer((request, response) => {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");

    // Anything else on this port, including a stale tab from an earlier sign-in, is
    // not this sign-in's answer.
    if (url.pathname !== CALLBACK_PATH || url.searchParams.get("state") !== state) {
      response.writeHead(404).end();

      return;
    }

    const failed = url.searchParams.has("error");

    response
      .writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" })
      .end(
        failed
          ? CALLBACK_PAGE("Sign-in cancelled", "You can close this tab. Nothing was connected.")
          : CALLBACK_PAGE(
              "You're signed in",
              "You can close this tab and return to your terminal.",
            ),
      );

    deliver({ params: url.searchParams });
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });

  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;

  return {
    redirectUrl: `http://127.0.0.1:${port}${CALLBACK_PATH}`,
    received,
    // A browser keeps its connection open, which would hold the CLI open after sign-in.
    close: () => {
      server.close();
      server.closeAllConnections();
    },
  };
};

/** Whether a browser on this machine can reach the loopback callback. */
const canOpenBrowser = () => {
  if (process.env.SSH_CONNECTION || process.env.SSH_TTY) return false;

  if (process.platform === "linux")
    return Boolean(process.env.DISPLAY || process.env.WAYLAND_DISPLAY);

  return process.platform === "darwin" || process.platform === "win32";
};

const openBrowser = (url: URL) => {
  const [command, args] =
    process.platform === "darwin"
      ? ["open", [url.href]]
      : process.platform === "win32"
        ? ["rundll32", ["url.dll,FileProtocolHandler", url.href]]
        : ["xdg-open", [url.href]];

  const child = spawn(command, args, { stdio: "ignore", detached: true });
  child.on("error", () => {});
  child.unref();
};

/** Reads the address the browser ended on, for a browser on another machine. */
const readPastedCallback = async (state: string, signal: AbortSignal): Promise<Callback> => {
  const prompt = createInterface({ input: process.stdin, output: process.stderr });
  // Readline swallows Ctrl-C unless something listens; hand it to the process handler.
  prompt.on("SIGINT", () => process.kill(process.pid, "SIGINT"));

  try {
    for (;;) {
      const answer = (
        await prompt.question("If the page after Allow doesn't load, paste its address here: ", {
          signal,
        })
      ).trim();

      if (URL.canParse(answer)) {
        const params = new URL(answer).searchParams;

        if (params.get("state") === state) return { params };
      }

      process.stderr.write(
        "That isn't the address from this sign-in. Copy the whole address bar and try again.\n",
      );
    }
  } finally {
    prompt.close();
  }
};

export type LoginOptions = {
  readonly session: Session;
  readonly browser: boolean;
  /** The address the browser ended on, to finish a sign-in started without a terminal. */
  readonly finish?: string | undefined;
  readonly onAuthorizationUrl: (url: URL, opened: boolean) => void;
  readonly onWaiting: () => void;
};

export type LoginResult =
  | { readonly status: "signed-in"; readonly session: Session }
  | { readonly status: "pending" };

const authorizationParams = (address: string, state: string) => {
  const params = URL.canParse(address) ? new URL(address).searchParams : undefined;

  if (params?.get("state") !== state) {
    throw new CliError(
      "That isn't the address from this sign-in. Copy the browser's whole address bar after you select Allow.",
      EXIT.auth,
    );
  }

  return params;
};

/**
 * Runs the SDK's authorization with the CLI's request timeout. A refused code, a
 * mismatched issuer, or a broken authorization server ends the sign-in with exit 4 and
 * the server's reason, not a stack trace.
 */
const authorize = async (
  provider: SessionAuthProvider,
  options: { authorizationCode?: string; iss?: string },
) => {
  const { auth, AuthorizationServerMismatchError, IssuerMismatchError, OAuthError } =
    await import("@modelcontextprotocol/client");

  try {
    return await auth(provider, {
      serverUrl: mcpUrl(provider.session.server),
      scope: SCOPES,
      fetchFn: oauthFetch,
      ...options,
    });
  } catch (error) {
    if (error instanceof CliError) throw error;

    if (error instanceof OAuthError) {
      throw new CliError(
        error.code === "invalid_grant"
          ? "That sign-in link was already used or has expired. Run `telemetry login` again."
          : `Sign-in failed: ${oneLine(error.message).slice(0, 300)}`,
        EXIT.auth,
      );
    }

    if (error instanceof IssuerMismatchError || error instanceof AuthorizationServerMismatchError) {
      throw new CliError(
        "The sign-in answer came from a different authorization server than this sign-in started with. Run `telemetry login` again.",
        EXIT.auth,
      );
    }

    throw error;
  }
};

/** Redeems the code in the browser's redirect, then saves the new sign-in in place of the old. */
const redeem = async (provider: SessionAuthProvider, params: URLSearchParams) => {
  const error = params.get("error");

  if (error) {
    throw new CliError(
      error === "access_denied"
        ? "Sign-in was cancelled in the browser."
        : `Sign-in failed: ${oneLine(params.get("error_description") ?? error)}.`,
      EXIT.auth,
    );
  }

  const code = params.get("code");

  if (!code) {
    throw new CliError(
      "The browser returned without an authorization code. Run `telemetry login` again.",
      EXIT.auth,
    );
  }

  const iss = params.get("iss");

  await authorize(provider, { authorizationCode: code, ...(iss ? { iss } : {}) });

  const email = await fetchEmail(provider.session);
  const { pending: _finished, ...signedIn } = provider.session;

  return updateSession(signedIn.server, () => ({ ...signedIn, ...(email ? { email } : {}) }));
};

/**
 * Signs in with OAuth 2.1 authorization code and PKCE through a loopback redirect
 * (RFC 8252), the same grant the MCP server issues to agents. Nothing is saved until the
 * new tokens arrive, so a cancelled sign-in keeps the current one.
 *
 * Without a usable browser, the person opens the link anywhere and brings back the
 * address the browser ends on: pasted at a prompt on a terminal, or, from a shell with
 * no terminal such as an agent's, passed to `telemetry login --finish` later, which is
 * why that path saves the PKCE verifier as a pending sign-in.
 */
export const login = async ({
  session,
  browser,
  finish,
  onAuthorizationUrl,
  onWaiting,
}: LoginOptions): Promise<LoginResult> => {
  const signedIn: Session = {
    server: session.server,
    ...(session.client ? { client: session.client } : {}),
    ...(session.discovery ? { discovery: session.discovery } : {}),
  };

  if (finish !== undefined) {
    const { pending } = session;

    if (finish.trim() === "") {
      throw new CliError(
        "Pass the address the browser ended on: telemetry login --finish '<address>'",
        EXIT.usage,
      );
    }

    if (!pending) {
      throw new CliError(
        "No sign-in is waiting to finish. Run `telemetry login --no-browser` to start one.",
        EXIT.auth,
      );
    }

    const provider = new SessionAuthProvider(signedIn, {
      ...pending,
      open: () => {
        throw sessionExpired();
      },
    });

    return {
      status: "signed-in",
      session: await redeem(provider, authorizationParams(finish, pending.state)),
    };
  }

  const state = randomBytes(16).toString("base64url");
  const callback = await listenForCallback(state);
  const abort = new AbortController();
  const useBrowser = browser && canOpenBrowser();
  const deferred = !useBrowser && !process.stdin.isTTY;

  const provider = new SessionAuthProvider(signedIn, {
    redirectUrl: callback.redirectUrl,
    state,
    open: (url) => {
      if (useBrowser) openBrowser(url);
      onAuthorizationUrl(url, useBrowser);
    },
  });

  let timeout: NodeJS.Timeout | undefined;

  try {
    await authorize(provider, {});

    if (deferred) {
      const { client, discovery } = provider.session;
      const pending = {
        redirectUrl: callback.redirectUrl,
        state,
        verifier: provider.codeVerifier(),
      };

      // The current sign-in keeps working until the pending one finishes.
      await updateSession(session.server, (saved) => ({
        ...saved,
        ...(client ? { client } : {}),
        ...(discovery ? { discovery } : {}),
        pending,
      }));

      return { status: "pending" };
    }

    onWaiting();

    const waits: Promise<URLSearchParams>[] = [
      callback.received.then(({ params }) => params),
      new Promise<never>((_, reject) => {
        timeout = setTimeout(
          () =>
            reject(
              new CliError(
                "Sign-in timed out after 10 minutes. Run `telemetry login` to try again.",
                EXIT.auth,
              ),
            ),
          LOGIN_TIMEOUT_MS,
        );
      }),
    ];

    if (!useBrowser) {
      const pasted = readPastedCallback(state, abort.signal).then(({ params }) => params);
      // The prompt rejects when it is aborted after the browser answered first.
      pasted.catch(() => {});
      waits.push(pasted);
    }

    return { status: "signed-in", session: await redeem(provider, await Promise.race(waits)) };
  } finally {
    clearTimeout(timeout);
    abort.abort();
    callback.close();
  }
};

/**
 * Revokes the refresh token at the server, then forgets the tokens. Keeps the client
 * registration so the next sign-in reuses it. A revocation that fails still signs out
 * locally, and reports it so the person can disconnect the CLI in account settings.
 */
export const logout = async (session: Session) => {
  const refresh = session.tokens?.refresh_token;
  let revoked = false;

  if (refresh && session.client) {
    try {
      const { discoverOAuthServerInfo } = await import("@modelcontextprotocol/client");
      const info = await discoverOAuthServerInfo(mcpUrl(session.server), { fetchFn: oauthFetch });

      // RFC 7009 revocation is in the OAuth metadata, outside the SDK's type.
      const metadata = info.authorizationServerMetadata as
        | { revocation_endpoint?: unknown }
        | undefined;
      const endpoint =
        typeof metadata?.revocation_endpoint === "string"
          ? metadata.revocation_endpoint
          : undefined;

      if (endpoint) {
        const response = await oauthFetch(endpoint, {
          method: "POST",
          redirect: "error",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: new URLSearchParams({
            token: refresh,
            token_type_hint: "refresh_token",
            client_id: session.client.client_id,
          }),
        });

        revoked = response.ok;
      }
    } catch {
      revoked = false;
    }
  }

  await updateSession(session.server, (saved) => ({
    server: saved.server,
    ...(saved.client ? { client: saved.client } : {}),
  }));

  return { revoked };
};
