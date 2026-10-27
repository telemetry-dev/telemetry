import { appendFileSync, renameSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { inspect } from "node:util";

/** pi moves its own `mcp.log` to `mcp.log.1` when the file is more than 5 MB. */
const MAX_LOG_BYTES = 5 * 1024 * 1024;

/**
 * Appends an error to `telemetry-dev.log` in pi's agent directory. pi draws its TUI on the
 * terminal, so an error written to stderr garbles the screen and is not kept.
 */
export function writeErrorLog(error: unknown): void {
  const path = join(agentDir(), "telemetry-dev.log");
  const entry = `${new Date().toISOString()} ${inspect(error).replaceAll("\n", "\n    ")}\n`;

  try {
    if ((statSync(path, { throwIfNoEntry: false })?.size ?? 0) > MAX_LOG_BYTES) {
      renameSync(path, `${path}.1`);
    }

    appendFileSync(path, entry, { mode: 0o600 });
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
