import { existsSync, realpathSync } from "node:fs";

import packageJson from "../package.json" with { type: "json" };

/**
 * Package runners start the CLI from a cache directory and put no `telemetry` on PATH,
 * so a hint that says `telemetry traces` fails once the runner exits. The paths are
 * where npx, pnpm dlx, Yarn's dlx, and bunx unpack a package; a global install, a
 * project's devDependency, and `npm run` all start it from elsewhere. Yarn has no
 * global installs since version 2, so its install is npm's.
 */
const RUNNERS = [
  { path: /\/_npx\//, run: "npx", install: "npm install --global" },
  // <cache-dir>/dlx/<hash>/<id>/node_modules, where the cache directory can be anywhere.
  {
    path: /\/pnpm[^/]*\/dlx\/|\/dlx\/[0-9a-f]{32,}\/[^/]+\/node_modules\//,
    run: "pnpm dlx",
    install: "pnpm add --global",
  },
  { path: /\/xfs-[0-9a-f]+\/dlx-\d+\//, run: "yarn dlx", install: "npm install --global" },
  // bunx-<uid>-<package>, so a project folder named bunx-something isn't one.
  { path: /\/bunx-\d+-/, run: "bunx", install: "bun add --global" },
];

/**
 * Where package managers other than npm keep global packages, matched
 * case-insensitively for Windows' AppData folders.
 */
const GLOBALS = [
  { path: /\/\.?volta\/tools\//i, update: "volta install" },
  // Under a PNPM_HOME of any name, pnpm 11+ keeps them in global/v<n>/<sha256>.
  {
    path: /\/pnpm\/global\/|\/global\/v\d+\/[0-9a-f]{64}\/node_modules\//i,
    update: "pnpm add --global",
  },
  { path: /\/\.bun\/install\/global\//i, update: "bun add --global" },
  { path: /\/yarn\/(data\/)?global\//i, update: "yarn global add" },
];

const NPM_GLOBAL = { update: "npm install --global" };
const PROJECT = { update: undefined };
const PACKAGE_FOLDER = `/node_modules/${packageJson.name}/`;

/**
 * How the CLI at `script` was installed. npm keeps global packages in
 * `<prefix>/lib/node_modules` on macOS and Linux, and in `<prefix>\node_modules` next
 * to the `telemetry.cmd` it writes on Windows, also under nvm-windows, fnm, or Scoop.
 * A project named lib has a package.json there instead. Every other `node_modules` is
 * a project's: next to its package.json, in Yarn's Plug'n'Play cache, in pnpm's
 * virtual store, or in Deno's.
 */
const placed = (script: string) => {
  const known = GLOBALS.find(({ path }) => path.test(script));

  if (known) return known;

  const at = script.indexOf(PACKAGE_FOLDER);
  const folder = script.slice(0, at);
  const npmGlobal =
    at !== -1 &&
    !existsSync(`${folder}/package.json`) &&
    (folder.endsWith("/lib") || existsSync(`${folder}/telemetry.cmd`));

  if (npmGlobal) return NPM_GLOBAL;

  return script.includes("/node_modules/") ? PROJECT : undefined;
};

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
const installed = placed(started) ?? placed(real);

/** The command that runs this CLI again from the shell that started it. */
export const BIN = runner ? `${runner.run} ${packageJson.name}` : "telemetry";

/** How to install `telemetry` on PATH, when a package runner started this one. */
export const INSTALL = runner && `${runner.install} ${packageJson.name}`;

/**
 * What to run for the newest version, or undefined when a project's lockfile pins it.
 * An install it can't place, such as one started from source, gets npm's command.
 */
export const UPDATE = runner
  ? `Run ${runner.run} ${packageJson.name}@latest, or install it: ${INSTALL}`
  : installed && !installed.update
    ? undefined
    : `Update: ${installed?.update ?? "npm install --global"} ${packageJson.name}@latest`;
