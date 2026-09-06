import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createConnection, type Socket } from "node:net";
import { setTimeout as sleep } from "node:timers/promises";

import { socketPath } from "./daemon.ts";

type JsonValue = string | number | boolean | null | undefined | JsonValue[] | JsonRecord;
interface JsonRecord {
  [key: string]: JsonValue;
}

const ACK_TIMEOUT_MS = 3000;
/**
 * Total delivery budget, including a cold daemon spawn. Bounds the worst-case
 * stall of one hook: without it, a socket that accepts but never acks could
 * hold the agent loop for the full retry ladder (~65s).
 */
const DELIVER_DEADLINE_MS = 10_000;

/**
 * Hook entry: forwards the stdin JSON payload to the daemon, spawning it on
 * first use. Always prints `{}` and exits 0 — telemetry must never block the
 * agent (Cursor hooks are fail-open, but a hang would still cost the timeout).
 */
export async function runHook(cliPath: string): Promise<void> {
  try {
    const payload = await readStdin();
    if (payload.trim().length > 0) await deliver(payload, cliPath);
  } catch {
    // Fail open.
  }
  process.exitCode = 0;
  process.stdout.write("{}\n");
}

function readStdin(): Promise<string> {
  const { promise, resolve } = Promise.withResolvers<string>();
  let data = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk: string) => (data += chunk));
  process.stdin.on("end", () => resolve(data));
  process.stdin.on("error", () => resolve(data));
  return promise;
}

async function deliver(payload: string, cliPath: string): Promise<void> {
  const line = frame(payload);
  const deadline = Date.now() + DELIVER_DEADLINE_MS;
  try {
    await send(line, deadline);
    return;
  } catch {
    spawnDaemon(cliPath);
  }
  // The daemon needs a moment to bind its socket after spawn.
  while (Date.now() < deadline) {
    await sleep(100);
    try {
      await send(line, deadline);
      return;
    } catch {
      continue;
    }
  }
}

/**
 * One JSON line per event. The delivery id lets the daemon drop retried
 * duplicates: an ack can get lost after the daemon already processed the
 * line, and the retry would otherwise double-emit spans and logs.
 */
function frame(payload: string): string {
  try {
    const parsed: unknown = JSON.parse(payload);
    const record = asRecord(parsed);
    if (record) {
      record.hook_delivery_id = randomUUID();
      return `${JSON.stringify(record)}\n`;
    }
  } catch {
    // Not JSON; forward as-is (the daemon drops undecodable lines).
  }
  return `${payload.replaceAll("\n", " ")}\n`;
}

function spawnDaemon(cliPath: string): void {
  const child = spawn(process.execPath, [cliPath, "daemon"], {
    detached: true,
    stdio: "ignore",
    env: process.env,
  });
  // A spawn failure (missing binary, fork limit) emits an async error event;
  // unhandled it would crash the hook before its fail-open `{}` output.
  child.once("error", () => {});
  child.unref();
}

function send(line: string, deadline: number): Promise<void> {
  const budget = Math.min(ACK_TIMEOUT_MS, deadline - Date.now());
  if (budget <= 0) return Promise.reject(new Error("delivery deadline exceeded"));
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  const socket: Socket = createConnection(socketPath());
  const timer = setTimeout(() => {
    socket.destroy();
    reject(new Error("ack timeout"));
  }, budget);
  socket.once("error", (error) => {
    clearTimeout(timer);
    reject(error);
  });
  socket.once("connect", () => socket.write(line));
  socket.once("data", () => {
    clearTimeout(timer);
    socket.end();
    resolve();
  });
  return promise;
}

function asRecord<T>(value: T): JsonRecord | undefined {
  return value !== null && !(value instanceof Function) && Object(value) === value
    ? (value as JsonRecord)
    : undefined;
}
