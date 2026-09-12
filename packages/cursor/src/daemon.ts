import { randomUUID } from "node:crypto";
import {
  chmodSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  renameSync,
  rmSync,
  rmdirSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { createConnection, createServer, type Server, type Socket } from "node:net";
import { tmpdir, userInfo } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

import { fileConfig } from "./config.ts";
import { createCursorTelemetry, type CursorTelemetry } from "./telemetry.ts";

type JsonValue = string | number | boolean | null | undefined | JsonValue[] | JsonRecord;

interface JsonRecord {
  [key: string]: JsonValue;
}

/** Exit after this long without events once every turn has settled. */
const IDLE_MS = 15 * 60 * 1000;
/** Stop waiting for in-flight hook connections after this long when shutting down. */
const DRAIN_MS = 2000;
/** A recovery lock older than this belongs to a crashed daemon and may be broken. */
const LOCK_STALE_MS = 10_000;
/** Delivery ids remembered for replay dedupe (hook retries after a lost ack). */
const SEEN_IDS_MAX = 1000;

export function socketPath(): string {
  if (process.platform === "win32") {
    return `\\\\.\\pipe\\telemetry-dev-cursor-${userInfo().username}`;
  }

  return join(socketDir(), "d.sock");
}

/**
 * Per-user 0700 directory for the socket. A predictable name directly in the
 * shared world-writable tmpdir would let another local user pre-bind the
 * socket and receive raw hook payloads; the private directory prevents that
 * and keeps the socket path short. XDG_RUNTIME_DIR is already per user.
 */
function socketDir(): string {
  const base = process.env.XDG_RUNTIME_DIR || tmpdir();
  const dir = join(base, `telemetry-dev-cursor-${userInfo().uid}`);
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  const stat = lstatSync(dir);
  const uid = process.getuid?.();

  if (!stat.isDirectory() || (uid !== undefined && stat.uid !== uid)) {
    throw new Error(`refusing unsafe socket directory ${dir}`);
  }

  // mkdirSync ignores mode for a pre-existing directory; tighten it ourselves.
  if ((stat.mode & 0o077) !== 0) chmodSync(dir, 0o700);

  return dir;
}

/**
 * Long-lived event sink spawned on demand by the hook forwarder. Holds open
 * spans across hook processes, exports batched OTLP, and exits when idle.
 */
export async function runDaemon(): Promise<void> {
  const telemetry = createCursorTelemetry(fileConfig());
  let server: Server | undefined;

  const shutdown = (): void => {
    if (server) void stop(server, telemetry);
  };

  server = await listen(socketPath(), telemetry, shutdown);

  if (!server) {
    await telemetry.settle();

    return;
  }

  let idleTimer: NodeJS.Timeout | undefined;

  const touch = (): void => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      if (telemetry.open()) touch();
      else shutdown();
    }, IDLE_MS);
  };

  server.on("connection", touch);
  touch();

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}

async function stop(server: Server, telemetry: CursorTelemetry): Promise<void> {
  // Let already accepted hook connections finish delivering before flushing,
  // but never hang shutdown on a stuck client.
  const drained = new Promise<void>((resolve) => server.close(() => resolve()));
  await Promise.race([drained, sleep(DRAIN_MS)]);
  await telemetry.settle();
  process.exit(0);
}

/** Exported for tests; production wiring lives in runDaemon. */
export async function listen(
  path: string,
  telemetry: CursorTelemetry,
  onShutdown: () => void = () => {},
): Promise<Server | undefined> {
  const seenIds = new Set<string>();
  const server = createServer((socket) => serve(socket, telemetry, seenIds, onShutdown));

  if (process.platform === "win32") {
    try {
      await bind(server, path);
    } catch (error) {
      const code = errorCode(error);

      if (code !== "EADDRINUSE") throw error;

      if (await alive(path)) return undefined;
      await bind(server, path);
    }

    return server;
  }

  const owner = lockRecovery(path);

  if (!owner) return undefined;

  try {
    try {
      await bind(server, path);
    } catch (error) {
      const code = errorCode(error);

      if (code !== "EADDRINUSE") throw error;

      if (await alive(path)) return undefined;
      const stale = lstatSync(path, { throwIfNoEntry: false });

      if (stale && !stale.isSocket()) {
        throw new Error(`refusing to replace non-socket ${path}`);
      }

      if (stale) unlinkSync(path);
      await bind(server, path);
    }

    return server;
  } finally {
    unlockRecovery(path, owner);
  }
}

/**
 * Publishes a nonempty lock directory with one rename. The owner's unique
 * entry lets stale cleanup and release remove only the lock they observed,
 * never a lock another daemon created in the meantime.
 */
function lockRecovery(path: string): string | undefined {
  const lock = `${path}.lock`;
  const owner = `${process.pid}-${randomUUID()}`;
  const pending = `${lock}.${owner}`;
  mkdirSync(pending, { mode: 0o700 });
  writeFileSync(join(pending, owner), "", { flag: "wx", mode: 0o600 });

  try {
    try {
      renameSync(pending, lock);

      return owner;
    } catch (error) {
      const code = errorCode(error);

      if (code !== "EEXIST" && code !== "ENOTEMPTY") throw error;
    }

    if (!clearStaleLock(lock)) return undefined;

    try {
      renameSync(pending, lock);

      return owner;
    } catch (error) {
      const code = errorCode(error);

      if (code === "EEXIST" || code === "ENOTEMPTY") return undefined;
      throw error;
    }
  } finally {
    rmSync(pending, { recursive: true, force: true });
  }
}

function clearStaleLock(lock: string): boolean {
  let entries: string[];

  try {
    entries = readdirSync(lock);
  } catch (error) {
    return errorCode(error) === "ENOENT";
  }

  if (entries.length > 1) return false;

  if (entries.length === 1) {
    const entry = entries[0];

    if (!entry) return false;
    const held = join(lock, entry);
    let mtimeMs: number;

    try {
      mtimeMs = lstatSync(held).mtimeMs;
    } catch (error) {
      return errorCode(error) === "ENOENT";
    }

    if (Date.now() - mtimeMs <= LOCK_STALE_MS) return false;

    try {
      unlinkSync(held);
    } catch (error) {
      return errorCode(error) === "ENOENT";
    }
  }

  try {
    rmdirSync(lock);

    return true;
  } catch (error) {
    return errorCode(error) === "ENOENT";
  }
}

function unlockRecovery(path: string, owner: string): void {
  const lock = `${path}.lock`;

  try {
    unlinkSync(join(lock, owner));
  } catch {
    return;
  }

  try {
    rmdirSync(lock);
  } catch {
    // The directory is already gone or belongs to a new owner.
  }
}

function errorCode<T>(error: T): string | undefined {
  if (error === null || error instanceof Function || Object(error) !== error) return undefined;
  const value = error as T & { code?: JsonValue };

  return "code" in value ? readString(value.code) : undefined;
}

function bind(server: Server, path: string): Promise<void> {
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  server.once("error", reject);
  server.listen(path, () => {
    server.removeListener("error", reject);
    resolve();
  });

  return promise;
}

function alive(path: string): Promise<boolean> {
  const { promise, resolve } = Promise.withResolvers<boolean>();
  const probe = createConnection(path);
  probe.once("connect", () => {
    probe.destroy();
    resolve(true);
  });
  probe.once("error", () => resolve(false));

  return promise;
}

/** Reads newline-delimited JSON events; acks each line so forwarders can exit. */
function serve(
  socket: Socket,
  telemetry: CursorTelemetry,
  seenIds: Set<string>,
  onShutdown: () => void,
): void {
  let buffer = "";
  socket.setEncoding("utf8");
  socket.on("data", (chunk: string) => {
    buffer += chunk;
    let index = buffer.indexOf("\n");

    while (index >= 0) {
      const line = buffer.slice(0, index);
      buffer = buffer.slice(index + 1);
      index = buffer.indexOf("\n");

      if (line.trim().length === 0) continue;
      let shutdown = false;

      try {
        const parsed = JSON.parse(line) as JsonRecord;

        if (parsed.telemetry_dev_control === "shutdown") {
          // Sent by install/uninstall so the next hook restarts the daemon
          // with the new configuration.
          shutdown = true;
        } else if (!replayed(parsed, seenIds)) {
          telemetry.handle(parsed);
        }
      } catch {
        // Malformed line from a mismatched forwarder version; drop it.
      }

      socket.write("ok\n");

      if (shutdown) onShutdown();
    }
  });
  socket.on("error", () => socket.destroy());
}

/**
 * The forwarder retries a line when the daemon's ack is lost; the delivery id
 * it stamps on each payload makes such replays no-ops instead of duplicate
 * spans and logs.
 */
function replayed(event: JsonRecord, seenIds: Set<string>): boolean {
  const id = readString(event.hook_delivery_id);
  delete event.hook_delivery_id;

  if (id === undefined) return false;

  if (seenIds.has(id)) return true;
  seenIds.add(id);

  if (seenIds.size > SEEN_IDS_MAX) {
    seenIds.delete(seenIds.values().next().value as string);
  }

  return false;
}

function readString<T>(value: T): string | undefined {
  const raw: unknown = value;

  return String(raw) === raw ? raw : undefined;
}
