import { existsSync, realpathSync } from "node:fs";

import packageJson from "../package.json" with { type: "json" };

/**
 * Package runners start the CLI from a cache directory and put no `telemetry` on PATH,
 * so a hint that says `telemetry traces` fails once the runner exits. The paths are
 * where npx, pnpm dlx, and bunx unpack a package; a global install, a project's
 * devDependency, and `npm run` all start it from elsewhere.
 */
const RUNNERS = [
  { path: /\/_npx\//, run: "npx", install: "npm install --global" },
  { path: /\/pnpm[^/]*\/dlx\//, run: "pnpm dlx", install: "pnpm add --global" },
  { path: /\/bunx-/, run: "bunx", install: "bun add --global" },
];

/**
 * Where each package manager keeps global packages. Volta's packages also sit under a
 * `lib/node_modules`, so it is checked before npm. npm's folder names are common
 * project names too, so a `node_modules` next to a package.json is a project's. Any
 * other `node_modules` is a project's dependency, whose version the lockfile pins.
 */
const GLOBALS = [
  { path: /\/\.volta\//, update: "volta install" },
  { path: /\/pnpm\/global\//, update: "pnpm add --global" },
  { path: /\/\.bun\/install\/global\//, update: "bun add --global" },
  { path: /\/yarn\/global\//, update: "yarn global add" },
  { path: /\/(lib|npm|nodejs)\/node_modules\//, update: "npm install --global", commonName: true },
  { path: /\/node_modules\//, update: undefined },
];

const slashes = (path: string) => path.replaceAll("\\", "/");

// The path as started keeps a runner's or a bin directory's symlink; the real path is
// where the package manager keeps the package.
const started = slashes(process.argv[1] ?? "");
const real = (() => {
  try {
    return slashes(realpathSync(started));
  } catch {
    return started;
  }
})();

const runner = RUNNERS.find(({ path }) => path.test(started) || path.test(real));
const placed = (script: string) =>
  GLOBALS.find(({ path, commonName }) => {
    const match = path.exec(script);
    const folder = match && script.slice(0, match.index + match[0].length - "node_modules/".length);

    return folder !== null && !(commonName && existsSync(`${folder}package.json`));
  });
const installed = placed(started) ?? placed(real);

/** The command that runs this CLI again from the shell that started it. */
export const BIN = runner ? `${runner.run} ${packageJson.name}` : "telemetry";

/** How to install `telemetry` on PATH, when a package runner started this one. */
export const INSTALL = runner && `${runner.install} ${packageJson.name}`;

/**
 * What to run for the newest version, or undefined when a project's lockfile pins it.
 * An install it can't place, such as a custom npm prefix, gets npm's command.
 */
export const UPDATE = runner
  ? `Run ${runner.run} ${packageJson.name}@latest, or install it: ${INSTALL}`
  : installed && !installed.update
    ? undefined
    : `Update: ${installed?.update ?? "npm install --global"} ${packageJson.name}@latest`;
