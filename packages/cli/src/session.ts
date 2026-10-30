import { createHash, randomBytes } from "node:crypto";
import { link, mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { homedir, hostname } from "node:os";
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
// For a lock whose holder can't be checked: one written by CLI 0.1.0, which holds only a
// token, or by another machine sharing the config directory. A lock is held only for a
// read and a write, so one this old was most likely left by a killed process.
const LOCK_STALE_MS = 3000;
const LOCK_WAIT_MS = 2 * LOCK_STALE_MS;

type LockOwner = { readonly token: string; readonly pid?: number; readonly host?: string };

// The locks and markers this process holds, by token.
const held = new Set<string>();

const readLock = async (path: string): Promise<LockOwner | undefined> => {
  const text = await readFile(path, "utf8").catch(() => undefined);

  if (text === undefined) return undefined;

  try {
    const owner = JSON.parse(text) as Partial<LockOwner> | null;

    if (typeof owner?.token === "string") return owner as LockOwner;
  } catch {}

  return { token: text };
};

/** Creates `path` naming this process as its owner, and returns its token; undefined when it exists. */
const claim = async (path: string) => {
  const owner: LockOwner = {
    token: randomBytes(12).toString("hex"),
    pid: process.pid,
    host: hostname(),
  };

  const record = JSON.stringify(owner);
  // Written whole first and linked into place, so where hard links work no command finds
  // the file without its owner. Written fresh each time, so its modification time is when
  // it was claimed.
  const draft = `${path}.${owner.token}.tmp`;

  // Recorded before the file appears: another update in this process can read it before
  // the claim resolves, and would take it for one left by an earlier process.
  held.add(owner.token);

  try {
    await writeFile(draft, record, { flag: "wx", mode: 0o600 });
    await link(draft, path).catch(async (error: NodeJS.ErrnoException) => {
      if (error.code === "EEXIST") throw error;

      // A filesystem without hard links; the record is written in place instead.
      await writeFile(path, record, { flag: "wx", mode: 0o600 });
    });
  } catch (error) {
    held.delete(owner.token);

    if ((error as NodeJS.ErrnoException).code === "EEXIST") return undefined;

    throw error;
  } finally {
    // Best effort: a leftover draft has a unique name and nothing reads it, and a failure
    // here must not turn a claim that succeeded into an error.
    await rm(draft, { force: true }).catch(() => {});
  }

  return owner.token;
};

/**
 * Whether the command holding a lock has exited without releasing it. On this machine a
 * running holder keeps its lock however long it pauses, such as a laptop asleep mid-write.
 * A lock from another machine or CLI 0.1.0, or one caught empty while being written on a
 * filesystem without hard links, is judged by its age.
 */
const abandoned = async (path: string, owner: LockOwner) => {
  const { pid } = owner;

  if (
    typeof pid === "number" &&
    Number.isSafeInteger(pid) &&
    pid > 0 &&
    owner.host === hostname()
  ) {
    // An earlier process with this one's pid left it, as when every run is pid 1 in a container.
    if (pid === process.pid) return !held.has(owner.token);

    try {
      process.kill(pid, 0);

      return false;
    } catch (error) {
      return (error as NodeJS.ErrnoException).code === "ESRCH";
    }
  }

  const changed = await stat(path).catch(() => undefined);

  return changed !== undefined && Date.now() - changed.mtimeMs > LOCK_STALE_MS;
};

/**
 * Removes the lock at `path` if it still holds `token`. Every removal, whether a holder
 * releasing its lock or a command breaking an abandoned one, first creates a marker named
 * after the token, so while holders are judged by their pid, two commands never act on
 * one lock, and one can't remove a lock another took after its check. A marker whose
 * owner exited partway is removed the same way. Returns false when another command holds
 * the marker.
 */
const removeLock = async (path: string, token: string): Promise<boolean> => {
  const marker = `${path}.${createHash("sha256").update(token).digest("hex").slice(0, 16)}`;
  const mine = await claim(marker);

  if (!mine) {
    const other = await readLock(marker);

    if (other && (await abandoned(marker, other))) await removeLock(marker, other.token);

    return false;
  }

  try {
    if ((await readLock(path))?.token === token) await rm(path, { force: true });

    return true;
  } finally {
    // Held for one read and one removal. Its owner is this running process, so on this
    // machine no other command removes it meanwhile.
    try {
      await rm(marker, { force: true });
    } finally {
      held.delete(mine);
    }
  }
};

/** Takes the lock at `path` and returns its owner token. */
const lock = async (path: string) => {
  const deadline = performance.now() + LOCK_WAIT_MS;

  for (;;) {
    const token = await claim(path);

    if (token) return token;

    const owner = await readLock(path);

    if (owner && (await abandoned(path, owner))) await removeLock(path, owner.token);

    if (performance.now() > deadline) {
      throw new CliError(
        `Another telemetry command is holding the sign-in file. If none is running, delete ${path} and try again.`,
      );
    }

    await sleep(LOCK_RETRY_MS);
  }
};

// When another command holds the marker, it is breaking this lock under the age rule,
// which can't see that this command is running. Waits for the lock or the marker to go,
// but not past LOCK_WAIT_MS: once this process exits, its lock is broken by its pid.
const unlock = async (path: string, token: string) => {
  const deadline = performance.now() + LOCK_WAIT_MS;

  try {
    while (
      !(await removeLock(path, token)) &&
      (await readLock(path))?.token === token &&
      performance.now() < deadline
    ) {
      await sleep(LOCK_RETRY_MS);
    }
  } finally {
    held.delete(token);
  }
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
