import { randomUUID } from "node:crypto";
import {
  closeSync,
  fchmodSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createConnection, type Socket } from "node:net";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";

import { configPath } from "./config.ts";
import { socketPath } from "./daemon.ts";
import { HOOK_EVENTS } from "./telemetry.ts";

type JsonValue = string | number | boolean | null | undefined | JsonValue[] | JsonRecord;
interface JsonRecord {
  [key: string]: JsonValue;
}

interface HookEntry extends JsonRecord {
  command: string;
}

/**
 * Wires every observed hook event in ~/.cursor/hooks.json to this CLI and
 * writes the api key to ~/.cursor/telemetry-dev.json. Idempotent: existing
 * telemetry-dev entries are replaced, other hooks are preserved.
 */
export async function runInstall(cliPath: string, args: string[]): Promise<void> {
  const apiKey = flag(args, "--api-key") ?? process.env.TELEMETRY_DEV_API_KEY;
  if (!apiKey) {
    process.stderr.write(
      "Missing api key. Pass --api-key td_live_… or set TELEMETRY_DEV_API_KEY.\n",
    );
    process.exit(1);
  }

  const config = Object.fromEntries([["apiKey", apiKey]]);
  const baseUrl = flag(args, "--base-url") ?? process.env.TELEMETRY_DEV_BASE_URL;
  if (baseUrl) config.baseUrl = baseUrl;
  const environment = flag(args, "--environment") ?? process.env.TELEMETRY_DEV_ENVIRONMENT;
  if (environment) config.environment = environment;
  const file = configPath();
  writeSecretJson(file, config);

  const hooksPath = join(homedir(), ".cursor", "hooks.json");
  const hooks = readJson(hooksPath);
  const command = hookCommand(cliPath);
  const table = (hooks.hooks ?? {}) as Record<string, HookEntry[]>;
  for (const event of HOOK_EVENTS) {
    const kept = (table[event] ?? []).filter((entry) => entry.telemetryDev !== true);
    kept.push({ command, telemetryDev: true });
    table[event] = kept;
  }
  writeJson(hooksPath, { version: 1, ...hooks, hooks: table });
  // A running daemon read the old config at startup; stop it so the next hook
  // restarts it with the new settings.
  await stopDaemon();

  process.stdout.write(`Wrote ${hooksPath} and ${configPath()}.\n`);
  process.stdout.write("Restart Cursor (or save any hooks.json) to load the hooks.\n");
}

/** Removes telemetry-dev entries from ~/.cursor/hooks.json. */
export async function runUninstall(): Promise<void> {
  const hooksPath = join(homedir(), ".cursor", "hooks.json");
  const hooks = readJson(hooksPath);
  const table = (hooks.hooks ?? {}) as Record<string, HookEntry[]>;
  for (const [event, entries] of Object.entries(table)) {
    table[event] = entries.filter((entry) => entry.telemetryDev !== true);
    if (table[event].length === 0) delete table[event];
  }
  writeJson(hooksPath, { ...hooks, hooks: table });
  await stopDaemon();
  process.stdout.write(`Removed telemetry-dev hooks from ${hooksPath}.\n`);
}

/** Asks a running daemon to exit; resolves quietly when none is listening. */
function stopDaemon(): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  let socket: Socket | undefined;
  const finish = (): void => {
    clearTimeout(timer);
    socket?.destroy();
    resolve();
  };
  const timer = setTimeout(() => finish(), 2000);
  try {
    socket = createConnection(socketPath());
    socket.once("error", finish);
    socket.once("data", finish);
    socket.once("connect", () => socket?.write('{"telemetry_dev_control":"shutdown"}\n'));
  } catch {
    finish();
  }
  return promise;
}

function flag(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    process.stderr.write(`Missing value for ${name}.\n`);
    process.exit(1);
  }
  return value;
}

function hookCommand(cliPath: string): string {
  if (process.platform === "win32") {
    const exe = process.execPath.replaceAll("'", "''");
    const cli = cliPath.replaceAll("'", "''");
    const script = `& '${exe}' '${cli}' hook`;
    const encoded = Buffer.from(script, "utf16le").toString("base64");
    return `powershell.exe -NoProfile -NonInteractive -EncodedCommand ${encoded}`;
  }
  const exe = `'${process.execPath.replaceAll("'", "'\\''")}'`;
  const cli = `'${cliPath.replaceAll("'", "'\\''")}'`;
  return `${exe} ${cli} hook`;
}

function readJson(path: string): JsonRecord {
  try {
    const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
    if (parsed !== null && !(parsed instanceof Function) && Object(parsed) === parsed) {
      return parsed as JsonRecord;
    }
  } catch {
    // Missing or invalid file; start fresh.
  }
  return {};
}

function writeSecretJson<T>(path: string, value: T): void {
  const dir = dirname(path);
  mkdirSync(dir, { recursive: true });
  const current = lstatSync(path, { throwIfNoEntry: false });
  if (current && !current.isFile()) {
    throw new Error(`refusing to replace non-regular file ${path}`);
  }

  const temp = join(dir, `.${basename(path)}.${process.pid}.${randomUUID()}`);
  let handle: number | undefined;
  try {
    handle = openSync(temp, "wx", 0o600);
    fchmodSync(handle, 0o600);
    writeFileSync(handle, `${JSON.stringify(value, null, 2)}\n`, "utf8");
    closeSync(handle);
    handle = undefined;
    renameSync(temp, path);
  } finally {
    if (handle !== undefined) closeSync(handle);
    rmSync(temp, { force: true });
  }
}

function writeJson<T>(path: string, value: T): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
