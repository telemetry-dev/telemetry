import { createHash, randomBytes } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import type {
  OAuthDiscoveryState,
  StoredOAuthClientInformation,
  StoredOAuthTokens,
} from "@modelcontextprotocol/client";

import { CliError, usageError } from "./errors.ts";

export const DEFAULT_SERVER = "https://telemetry.dev";

/** The telemetry.dev site the CLI signs in to. TELEMETRY_DEV_URL points it at another deployment. */
export const serverUrl = () => {
  const configured = process.env.TELEMETRY_DEV_URL?.trim() || DEFAULT_SERVER;
  const url = URL.canParse(configured) ? new URL(configured) : undefined;

  if (url?.protocol !== "https:" && url?.protocol !== "http:") {
    throw usageError(
      `TELEMETRY_DEV_URL must be an http or https address, such as ${DEFAULT_SERVER}.`,
    );
  }

  return url.origin;
};

export const mcpUrl = (server: string) => new URL("/mcp", server);

export type Project = { readonly id: string; readonly name: string };

/** A sign-in started by `telemetry login` in a shell with no terminal, waiting for --finish. */
export type PendingLogin = {
  readonly redirectUrl: string;
  readonly state: string;
  readonly verifier: string;
  /** The client the sign-in started with; the saved one stays with the current tokens. */
  readonly client?: StoredOAuthClientInformation;
};

/** What one sign-in keeps on disk. The client registration outlives sign-outs. */
export type Session = {
  server: string;
  client?: StoredOAuthClientInformation;
  tokens?: StoredOAuthTokens;
  email?: string;
  project?: Project;
  pending?: PendingLogin;
  discovery?: OAuthDiscoveryState;
};

export const configDir = () => {
  const override = process.env.TELEMETRY_DEV_CONFIG_DIR?.trim();

  if (override) return override;

  const xdg = process.env.XDG_CONFIG_HOME?.trim();

  return join(xdg || join(homedir(), ".config"), "telemetry-dev");
};

/** The file one server's sign-in lives in, so signing in to another deployment keeps this one. */
export const sessionFile = (server: string) =>
  server === DEFAULT_SERVER
    ? "session.json"
    : `session-${createHash("sha256").update(server).digest("hex").slice(0, 16)}.json`;

const sessionPath = (server: string) => join(configDir(), sessionFile(server));

/** The saved session for `server`, or a new empty one when none is saved for it. */
export const loadSession = async (server: string): Promise<Session> => {
  const text = await readFile(sessionPath(server), "utf8").catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return null;

    throw error;
  });

  if (text === null) return { server };

  let saved: unknown;

  try {
    saved = JSON.parse(text);
  } catch {
    return { server };
  }

  if (typeof saved !== "object" || saved === null || (saved as Session).server !== server) {
    return { server };
  }

  return saved as Session;
};

/**
 * Writes the session readable by its owner only. The file holds a refresh token, so it
 * is written to a private temporary file in the same directory and renamed over the
 * old one: a crash never leaves a partial file, and no other user can read it mid-write.
 */
const saveSession = async (session: Session) => {
  const dir = configDir();
  const temporary = join(dir, `.session.${randomBytes(6).toString("hex")}.tmp`);

  try {
    await writeFile(temporary, `${JSON.stringify(session, null, 2)}\n`, {
      mode: 0o600,
      flag: "wx",
    });
    await rename(temporary, sessionPath(session.server));
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
};

const LOCK_RETRY_MS = 25;
// A lock is held only for a read and a write, so one this old was left by a killed process.
const LOCK_STALE_MS = 3000;
const LOCK_WAIT_MS = 2 * LOCK_STALE_MS;

const readLock = (path: string) => readFile(path, "utf8").catch(() => undefined);

/**
 * Takes the lock at `path` and returns its owner token. Only a lock still holding the
 * token that was read is broken or released, so a command never removes another's lock.
 */
const lock = async (path: string) => {
  const token = randomBytes(12).toString("hex");
  const deadline = Date.now() + LOCK_WAIT_MS;

  for (;;) {
    try {
      await writeFile(path, token, { flag: "wx", mode: 0o600 });

      return token;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    }

    const held = await stat(path).catch(() => undefined);
    const owner = await readLock(path);

    if (held && owner !== undefined && Date.now() - held.mtimeMs > LOCK_STALE_MS) {
      if ((await readLock(path)) === owner) await rm(path, { force: true });
      continue;
    }

    if (Date.now() > deadline) {
      throw new CliError(
        `Another telemetry command is holding the sign-in file. If none is running, delete ${path} and try again.`,
      );
    }

    await sleep(LOCK_RETRY_MS);
  }
};

const unlock = async (path: string, token: string) => {
  if ((await readLock(path)) === token) await rm(path, { force: true });
};

/**
 * Applies `change` to the session as saved now and saves the result, with other telemetry
 * processes locked out between the read and the write. Agents run several commands at
 * once, and each one changes only its own fields, so a command that started earlier never
 * puts back tokens, a default project, or a pending sign-in that another one replaced.
 */
export const updateSession = async (server: string, change: (saved: Session) => Session) => {
  await mkdir(configDir(), { recursive: true, mode: 0o700 });

  const lockPath = `${sessionPath(server)}.lock`;
  const token = await lock(lockPath);

  try {
    const next = change(await loadSession(server));
    await saveSession(next);

    return next;
  } finally {
    await unlock(lockPath, token);
  }
};
