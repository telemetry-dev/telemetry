import { randomBytes } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

import packageJson from "../package.json" with { type: "json" };

import { UPDATE } from "./invocation.ts";
import { configDir } from "./session.ts";
import { style } from "./terminal.ts";

/**
 * Says when npm has a newer CLI. The check asks the registry at most once a day and
 * never delays a command: it runs alongside the command, and is dropped, for the next
 * run to repeat, when the command finishes first. It never installs anything: the
 * CLI can't reliably tell how it was installed, and agents run it.
 */

const CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;
const CHECK_TIMEOUT_MS = 5000;
const STABLE = /^(\d{1,9})\.(\d{1,9})\.(\d{1,9})$/;

type Check = { latest?: string; checkedAt: number };

const cachePath = () => join(configDir(), "update-check.json");

/** Turned off with either variable, or for a project's dependency, which its lockfile pins. */
const checksAllowed = () =>
  !process.env.TELEMETRY_DEV_NO_UPDATE_CHECK && !process.env.NO_UPDATE_NOTIFIER && Boolean(UPDATE);

const stable = (value: unknown) =>
  typeof value === "string" && STABLE.test(value) ? value : undefined;

const readCheck = async (): Promise<Check | undefined> => {
  try {
    const saved = JSON.parse(await readFile(cachePath(), "utf8")) as Record<string, unknown> | null;
    const checkedAt = typeof saved?.checked_at === "string" ? Date.parse(saved.checked_at) : NaN;

    if (Number.isNaN(checkedAt)) return undefined;

    const latest = stable(saved?.latest);

    return { checkedAt, ...(latest ? { latest } : {}) };
  } catch {
    return undefined;
  }
};

// Written like the session, through a private temporary file renamed into place, but
// without the session's lock: commands that check at once write the same answer.
const writeCheck = async (check: Check) => {
  const dir = configDir();
  const temporary = join(dir, `.update-check.${randomBytes(6).toString("hex")}.tmp`);
  const saved = {
    ...(check.latest ? { latest: check.latest } : {}),
    checked_at: new Date(check.checkedAt).toISOString(),
  };

  try {
    await mkdir(dir, { recursive: true, mode: 0o700 });
    await writeFile(temporary, `${JSON.stringify(saved)}\n`, { mode: 0o600, flag: "wx" });
    await rename(temporary, cachePath());
  } catch {
    await rm(temporary, { force: true }).catch(() => {});
  }
};

const registryUrl = () => {
  const configured = process.env.npm_config_registry?.trim() || "https://registry.npmjs.org/";
  const base = configured.endsWith("/") ? configured : `${configured}/`;

  return new URL(`-/package/${packageJson.name.replace("/", "%2f")}/dist-tags`, base);
};

/** Whether `latest` is newer than `current`; a prerelease is older than its release. */
const isNewer = (latest: string, current: string) => {
  const [core = "", prerelease] = current.split("-", 2);
  const have = core.split(".").map(Number);
  const want = latest.split(".").map(Number);

  for (let index = 0; index < 3; index += 1) {
    if (want[index]! !== have[index]) return want[index]! > (have[index] ?? 0);
  }

  return prerelease !== undefined;
};

/**
 * The latest version the last check found, for `status --json`; never asks the registry.
 * Null when that is older than this CLI, as after an update the check hasn't seen.
 */
export const cachedLatestVersion = async () => {
  const latest = checksAllowed() ? (await readCheck())?.latest : undefined;

  return latest && (latest === packageJson.version || isNewer(latest, packageJson.version))
    ? latest
    : null;
};

let running:
  | { saved: Promise<Check | undefined>; fresh: Promise<Check | undefined>; abort: () => void }
  | undefined;

/**
 * Starts the day's check when a person will see its answer: on a terminal, outside CI,
 * and not with --json, which agents read.
 */
export const startUpdateCheck = (json: boolean) => {
  if (json || !process.stdout.isTTY || !process.stderr.isTTY || process.env.CI) return;

  if (!checksAllowed()) return;

  const controller = new AbortController();
  let answered = false;
  let dropped = false;
  const saved = readCheck();

  const fresh = saved.then(async (check) => {
    const now = Date.now();

    if (check && check.checkedAt <= now && now - check.checkedAt < CHECK_INTERVAL_MS) {
      return undefined;
    }

    let latest: string | undefined;
    const timeout = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS).unref();

    try {
      const response = await fetch(registryUrl(), {
        headers: { accept: "application/json" },
        signal: controller.signal,
      });

      latest = response.ok
        ? stable(((await response.json()) as Record<string, unknown> | null)?.latest)
        : undefined;
      answered = true;
    } catch {
      // Dropped because the command finished first: the next run asks again.
      if (dropped) return undefined;
    } finally {
      clearTimeout(timeout);
    }

    // A failed check counts too, so a machine offline asks once a day, not every run.
    const found = latest ?? check?.latest;
    const next = { checkedAt: now, ...(found ? { latest: found } : {}) };
    await writeCheck(next);

    return next;
  });

  running = {
    saved,
    fresh,
    // An unanswered request, or a body that stalls, would keep the process open after the
    // command finished. Once npm's answer is read, saving it takes milliseconds.
    abort: () => {
      if (answered) return;

      dropped = true;
      controller.abort();
    },
  };
};

/** After the command's output: ends the check, then prints one line if npm has a newer CLI. */
export const finishUpdateCheck = async () => {
  if (!running) return;

  const { saved, fresh, abort } = running;
  running = undefined;
  abort();

  const check = (await fresh.catch(() => undefined)) ?? (await saved);
  const latest = check?.latest;

  if (!latest || !isNewer(latest, packageJson.version)) return;

  process.stderr.write(
    `${style.dim(`telemetry ${latest} is available (you have ${packageJson.version}). ${UPDATE}`, process.stderr)}\n`,
  );
};
