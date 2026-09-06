#!/usr/bin/env node
import { fileURLToPath } from "node:url";

import { runDaemon } from "./daemon.ts";
import { runHook } from "./hook.ts";
import { runInstall, runUninstall } from "./install.ts";

const cliPath = fileURLToPath(import.meta.url);
const [command, ...rest] = process.argv.slice(2);

switch (command) {
  case "hook":
    await runHook(cliPath);
    break;
  case "daemon":
    await runDaemon();
    break;
  case "install":
    await runInstall(cliPath, rest);
    break;
  case "uninstall":
    await runUninstall();
    break;
  default:
    process.stderr.write(
      "Usage: telemetry-dev-cursor <install|uninstall|hook|daemon>\n" +
        "  install [--api-key td_live_…] [--base-url url] [--environment name]\n",
    );
    process.exit(1);
}
