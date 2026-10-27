import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test, vi } from "vitest";

import { flushErrorLog, writeErrorLog } from "../src/error-log.ts";

const appendFile = vi.hoisted(() => vi.fn(() => new Promise<void>(() => {})));

// A separate file, so the stalled write below cannot hold up the queue in other tests.
vi.mock("node:fs/promises", async (importOriginal) => ({
  ...(await importOriginal<typeof import("node:fs/promises")>()),
  appendFile,
}));

test("flushErrorLog waits for a stalled write until its deadline, then gives up", async () => {
  const directory = await mkdtemp(join(tmpdir(), "telemetry-pi-agent-"));
  vi.stubEnv("PI_CODING_AGENT_DIR", directory);

  try {
    writeErrorLog(new Error("stalled"));
    await vi.waitFor(() => expect(appendFile).toHaveBeenCalledOnce());

    const started = performance.now();
    await flushErrorLog();
    const elapsed = performance.now() - started;

    expect(elapsed).toBeGreaterThanOrEqual(900);
    expect(elapsed).toBeLessThan(3000);
  } finally {
    vi.unstubAllEnvs();
    await rm(directory, { recursive: true, force: true });
  }
});
