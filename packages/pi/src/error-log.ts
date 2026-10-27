import { appendFile, rename, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";

/** pi moves its own `mcp.log` to `mcp.log.1` when the file is more than 5 MB. */
const MAX_LOG_BYTES = 5 * 1024 * 1024;

/** Entries waiting for the disk; new entries are dropped while this many are pending. */
const MAX_PENDING_ENTRIES = 100;

/** How long `flushErrorLog` waits for a stalled disk before giving up. */
const FLUSH_DEADLINE_MS = 1000;

let pending = 0;
let queue: Promise<void> = Promise.resolve();

/**
 * Queues an error for `telemetry-dev.log` in pi's agent directory and returns without waiting
 * on the disk. pi draws its TUI on the terminal, so an error written to stderr garbles the
 * screen and is not kept.
 */
export function writeErrorLog(error: unknown): void {
  if (pending >= MAX_PENDING_ENTRIES) return;

  const path = join(agentDir(), "telemetry-dev.log");
  const entry = `${new Date().toISOString()} ${inspect(error).replaceAll("\n", "\n    ")}\n`;

  pending += 1;
  queue = queue
    .then(() => append(path, entry))
    .finally(() => {
      pending -= 1;
    });
}

/** Resolves once every queued entry has been written or has failed. Never rejects. */
export function drainErrorLogForTesting(): Promise<void> {
  return queue;
}

/**
 * Resolves once every queued entry has been written or has failed, or after `FLUSH_DEADLINE_MS`
 * if a write stalls. Never rejects.
 */
export async function flushErrorLog(): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<void>((resolve) => {
    timer = setTimeout(resolve, FLUSH_DEADLINE_MS);
    timer.unref?.();
  });

  try {
    await Promise.race([queue, deadline]);
  } finally {
    clearTimeout(timer);
  }
}

async function append(path: string, entry: string): Promise<void> {
  try {
    const size = await stat(path).then(
      (stats) => stats.size,
      () => 0,
    );

    if (size > MAX_LOG_BYTES) await rename(path, `${path}.1`);

    await appendFile(path, entry, { mode: 0o600 });
  } catch {
    // The log is best effort; telemetry must never fail a pi session.
  }
}

/** pi's agent directory: `PI_CODING_AGENT_DIR`, else `~/.pi/agent`. */
function agentDir(): string {
  const dir = process.env.PI_CODING_AGENT_DIR;

  if (!dir) return join(homedir(), ".pi", "agent");

  return dir === "~" || dir.startsWith("~/") ? join(homedir(), dir.slice(1)) : dir;
}
