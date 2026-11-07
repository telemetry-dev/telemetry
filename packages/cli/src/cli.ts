#!/usr/bin/env node
import { parseArgs } from "node:util";

import { VERSION } from "./api.ts";
import {
  type Context,
  findingsCommand,
  home,
  jsonText,
  loginCommand,
  logoutCommand,
  logsCommand,
  payloadCommand,
  projectsCommand,
  sessionsCommand,
  statusCommand,
  TAGLINE,
  traceCommand,
  tracesCommand,
  usageCommand,
  useCommand,
} from "./commands.ts";
import { CliError, type ErrorCode, EXIT, usageError } from "./errors.ts";
import { BIN } from "./invocation.ts";
import { clean, disableColor, SHOW_CURSOR, shellQuote, shellWords, style } from "./terminal.ts";
import { finishUpdateCheck, startUpdateCheck } from "./update.ts";

type Option = {
  readonly type: "string" | "boolean";
  readonly short?: string;
  readonly value?: string;
  readonly description: string;
};

type Command = {
  readonly summary: string;
  readonly usage: string;
  /** How many arguments the command takes after its name. */
  readonly arguments?: number;
  readonly options: Readonly<Record<string, Option>>;
  readonly examples: readonly string[];
  readonly run: (context: Context) => Promise<void>;
};

const json: Option = { type: "boolean", description: "Print JSON for scripts and agents" };
const project: Option = {
  type: "string",
  short: "p",
  value: "name|id",
  description: `Project to read (default: ${BIN} use)`,
};
const limit: Option = {
  type: "string",
  short: "n",
  value: "count",
  description: "Results per page, 1 to 50 (default 20)",
};
const cursor: Option = {
  type: "string",
  value: "cursor",
  description: "Continue from a previous page",
};
const since: Option = {
  type: "string",
  value: "when",
  description: "Start of the window: 30m, 2h, or 7d ago, a date, or a time",
};
const until: Option = {
  type: "string",
  value: "when",
  description: "End of the window, in the same forms (default now)",
};

const COMMANDS: Readonly<Record<string, Command>> = {
  login: {
    summary: "Sign in with your browser",
    usage: `${BIN} login [--no-browser | --finish <address>] [--json]`,
    options: {
      "no-browser": {
        type: "boolean",
        description: "Print the sign-in link instead of opening a browser",
      },
      finish: {
        type: "string",
        value: "address",
        description:
          "Finish a sign-in started without a terminal, with the browser's final address",
      },
      json: {
        type: "boolean",
        description: "Print JSON: the sign-in link without waiting, or the sign-in after --finish",
      },
    },
    examples: [
      `${BIN} login`,
      `${BIN} login --no-browser   # over SSH, in a container, or from an agent's shell`,
      `${BIN} login --finish ${shellQuote("http://127.0.0.1:53124/callback?code=…")}`,
      `${BIN} login --json         # a script passes the link on, then runs --finish`,
    ],
    run: loginCommand,
  },
  logout: {
    summary: "Sign out on this machine",
    usage: `${BIN} logout [--json]`,
    options: { json },
    examples: [`${BIN} logout`],
    run: logoutCommand,
  },
  status: {
    summary: "Show who you're signed in as",
    usage: `${BIN} status [--json]`,
    options: { json },
    examples: [`${BIN} status`, `${BIN} status --json`],
    run: statusCommand,
  },
  projects: {
    summary: "List the workspace's projects",
    usage: `${BIN} projects [--json]`,
    options: { json },
    examples: [`${BIN} projects`],
    run: projectsCommand,
  },
  use: {
    summary: "Set the default project",
    usage: `${BIN} use <name|id>`,
    arguments: 1,
    options: { json },
    examples: [`${BIN} use support-bot`],
    run: useCommand,
  },
  traces: {
    summary: "Find traces, newest first",
    usage: `${BIN} traces [options]`,
    options: {
      errors: { type: "boolean", description: "Only traces that failed" },
      name: { type: "string", value: "text", description: "Trace name contains this text" },
      env: { type: "string", value: "name", description: "Environment, such as production" },
      session: { type: "string", value: "id", description: "Traces in this session" },
      user: { type: "string", value: "id", description: "Traces for this end user" },
      framework: {
        type: "string",
        value: "name",
        description: "Traces from this framework, such as vercel-ai-sdk",
      },
      service: { type: "string", value: "name", description: "Traces from this service" },
      model: { type: "string", value: "name", description: "Traces with a span on this model" },
      provider: {
        type: "string",
        value: "name",
        description: "Traces with a span on this provider, such as openai",
      },
      operation: {
        type: "string",
        value: "name",
        description: "Traces with a span of this operation, such as chat",
      },
      agent: { type: "string", value: "name", description: "Traces with a span of this agent" },
      tool: { type: "string", value: "name", description: "Traces that called this tool" },
      since,
      until,
      limit,
      cursor,
      project,
      json,
    },
    examples: [
      `${BIN} traces --errors --since 2h`,
      `${BIN} traces --name checkout --env production`,
      `${BIN} traces --model gpt-6-luna --since 2026-10-29 --until 2026-10-30`,
      `${BIN} traces --tool search_docs --errors --since 1d`,
      `${BIN} traces --session conv_8f2 --json`,
    ],
    run: tracesCommand,
  },
  trace: {
    summary: "Show a trace's spans, timeline, and logs",
    usage: `${BIN} trace <trace-id> [options]`,
    arguments: 1,
    options: { project, json },
    examples: [`${BIN} trace 4bf92f3577b34da6a3ce929d0e0e4736`],
    run: traceCommand,
  },
  sessions: {
    summary: "Find sessions by the start of their latest trace, newest first",
    usage: `${BIN} sessions [options]`,
    options: {
      range: {
        type: "string",
        value: "24h|72h|7d|30d",
        description: "Active within this window (default 24h)",
      },
      since,
      until,
      errors: { type: "boolean", description: "Only sessions with a failed trace" },
      contains: {
        type: "string",
        value: "text",
        description: "Session id or a trace name contains this text",
      },
      agent: { type: "string", value: "name", description: "Sessions run by this agent" },
      model: { type: "string", value: "name", description: "Sessions that used this model" },
      user: { type: "string", value: "id", description: "Sessions for this end user" },
      limit,
      cursor,
      project,
      json,
    },
    examples: [
      `${BIN} sessions --range 7d --errors`,
      `${BIN} sessions --agent support-bot --since 2026-10-29T09:00`,
    ],
    run: sessionsCommand,
  },
  logs: {
    summary: "List recent logs, newest first",
    usage: `${BIN} logs [options]`,
    options: {
      level: {
        type: "string",
        value: "level",
        description: "debug, info, warn, error, or all (default all)",
      },
      range: { type: "string", value: "7d|30d|90d", description: "Window to search (default 7d)" },
      since,
      until,
      limit,
      cursor,
      project,
      json,
    },
    examples: [
      `${BIN} logs --level error`,
      `${BIN} logs --level warn --since 30m`,
      `${BIN} logs --range 30d --json`,
    ],
    run: logsCommand,
  },
  findings: {
    summary: "List alerts and failed checks; exit 3 when there are any",
    usage: `${BIN} findings [options]`,
    options: {
      since: {
        ...since,
        description:
          "Start of the window, such as the time of a deploy: 30m, 2h, or 7d ago, a date, or a time (default 7d)",
      },
      until,
      kind: {
        type: "string",
        value: "alert|check",
        description: "Only alerts, or only failed checks",
      },
      limit,
      cursor,
      project,
      json,
    },
    examples: [
      `${BIN} findings --since 2h`,
      `${BIN} findings --since 2026-11-06T14:00:00Z   # since a deploy, in CI or an agent's shell`,
      `${BIN} findings --kind check --json`,
    ],
    run: findingsCommand,
  },
  usage: {
    summary: "Summarize requests, errors, latency, tokens, and cost",
    usage: `${BIN} usage [options]`,
    options: {
      range: {
        type: "string",
        value: "7d|30d|90d",
        description: "Window to summarize (default 7d)",
      },
      by: {
        type: "string",
        value: "group",
        description: "model, provider, user, or operation (default model)",
      },
      project,
      json,
    },
    examples: [`${BIN} usage`, `${BIN} usage --range 30d --by user`],
    run: usageCommand,
  },
  payload: {
    summary: "Print a span's captured input and output",
    usage: `${BIN} payload <trace-id> <span-id> [options]`,
    arguments: 2,
    options: {
      field: {
        type: "string",
        value: "input|output|error",
        description: "Print only this field as plain text, for a pipe or a file",
      },
      project,
      json,
    },
    examples: [
      `${BIN} payload 4bf92f35… 00f067aa…`,
      `${BIN} payload 4bf92f35… 00f067aa… --field output > answer.txt`,
    ],
    run: payloadCommand,
  },
};

const GLOBAL: Readonly<Record<string, Option>> = {
  help: { type: "boolean", short: "h", description: "Show help" },
  version: { type: "boolean", short: "v", description: "Print the version" },
  "no-color": { type: "boolean", description: "Print without color" },
};

const optionLines = (options: Readonly<Record<string, Option>>) => {
  const entries = Object.entries(options).map(([name, option]) => {
    const flag = `${option.short ? `-${option.short}, ` : "    "}--${name}${option.value ? ` <${option.value}>` : ""}`;

    return [flag, option.description] as const;
  });

  const width = Math.max(...entries.map(([flag]) => flag.length));

  return entries.map(([flag, description]) => `  ${flag.padEnd(width)}  ${style.dim(description)}`);
};

const topHelp = () => {
  const names = Object.keys(COMMANDS);
  const width = Math.max(...names.map((name) => name.length));

  return [
    `${style.bold("telemetry")} ${style.dim(VERSION)}  ${TAGLINE}`,
    "",
    style.bold("Usage"),
    `  ${BIN} <command> [options]`,
    "",
    style.bold("Commands"),
    ...names.map((name) => `  ${name.padEnd(width)}  ${style.dim(COMMANDS[name]!.summary)}`),
    "",
    style.bold("Options"),
    ...optionLines(GLOBAL),
    "",
    style.bold("Environment"),
    `  TELEMETRY_DEV_PROJECT          ${style.dim("Project to read when --project is not given")}`,
    `  TELEMETRY_DEV_URL              ${style.dim("telemetry.dev deployment (default https://telemetry.dev)")}`,
    `  TELEMETRY_DEV_CONFIG_DIR       ${style.dim("Where the sign-in is kept (default ~/.config/telemetry-dev)")}`,
    `  NO_COLOR                       ${style.dim("Print without color")}`,
    `  TELEMETRY_DEV_NO_UPDATE_CHECK  ${style.dim("Don't check npm for a newer version")}`,
    "",
    `Run ${style.bold(`${BIN} <command> --help`)} for a command's options and examples.`,
  ].join("\n");
};

const commandHelp = (name: string, command: Command) =>
  [
    `${style.bold(`telemetry ${name}`)}  ${command.summary}`,
    "",
    style.bold("Usage"),
    `  ${command.usage}`,
    ...(Object.keys(command.options).length > 0
      ? ["", style.bold("Options"), ...optionLines(command.options)]
      : []),
    "",
    style.bold("Examples"),
    ...command.examples.map((example) => `  ${example}`),
  ].join("\n");

/** The closest command name, for "did you mean". */
const closest = (input: string, names: readonly string[]) => {
  const distance = (a: string, b: string) => {
    const row = Array.from({ length: b.length + 1 }, (_, index) => index);

    for (let i = 1; i <= a.length; i += 1) {
      let previous = row[0]!;
      row[0] = i;

      for (let j = 1; j <= b.length; j += 1) {
        const current = row[j]!;
        row[j] = Math.min(row[j]! + 1, row[j - 1]! + 1, previous + (a[i - 1] === b[j - 1] ? 0 : 1));
        previous = current;
      }
    }

    return row[b.length]!;
  };

  const [best] = names
    .map((name) => ({ name, score: distance(input, name) }))
    .sort((left, right) => left.score - right.score);

  return best && best.score <= 2 ? best.name : undefined;
};

const commandNamed = (name: string | undefined) =>
  name !== undefined && Object.hasOwn(COMMANDS, name) ? COMMANDS[name] : undefined;

/** Whether some command reads `arg` as an option followed by its value, as in `--name x`. */
const takesValue = (arg: string | undefined) =>
  Object.values(COMMANDS).some((command) =>
    Object.entries(command.options).some(
      ([name, option]) =>
        option.type === "string" &&
        (arg === `--${name}` || (option.short !== undefined && arg === `-${option.short}`)),
    ),
  );

// Arguments after `--` are values, never options.
const optionArgs = (argv: readonly string[]) => {
  const end = argv.indexOf("--");

  return end === -1 ? argv : argv.slice(0, end);
};

const jsonOutput = optionArgs(process.argv.slice(2)).includes("--json");

const main = async (argv: readonly string[]) => {
  // With --json, stderr carries no spinner or color, so an error line is plain JSON.
  if (argv.includes("--no-color") || jsonOutput) disableColor();

  // The command comes first; only --no-color may precede it.
  const first = argv.findIndex((arg) => arg !== "--no-color");
  const name = first === -1 || argv[first]!.startsWith("-") ? undefined : argv[first];

  if (name === undefined) {
    if (argv.includes("--version") || argv.includes("-v"))
      return void process.stdout.write(`${VERSION}\n`);

    if (argv.includes("--help") || argv.includes("-h")) {
      return void process.stdout.write(`${topHelp()}\n`);
    }

    const unknown = argv.find((arg) => arg !== "--no-color");

    // As in `telemetry --json traces`. With one command named, it moves to the front;
    // with several, or one right after an option that takes a value, a name may be that
    // value, so no order is guessed.
    const named = optionArgs(argv).filter((arg) => commandNamed(arg));

    if (named.length > 0) {
      const at = argv.indexOf(named[0]!);
      const reordered =
        named.length === 1 && !takesValue(argv[at - 1])
          ? shellWords([argv[at]!, ...argv.slice(0, at), ...argv.slice(at + 1)])
          : undefined;

      throw usageError(
        reordered
          ? `Put the command before its options: ${BIN} ${reordered}`
          : `Put the command before its options, as in ${BIN} traces --json.`,
        reordered ? { hint: `${BIN} ${reordered}` } : {},
      );
    }

    if (unknown !== undefined) throw usageError(`Unknown option ${unknown}. Run ${BIN} --help.`);

    // A script or agent that runs `telemetry` alone gets the reference, not the welcome.
    return process.stdout.isTTY ? home() : void process.stdout.write(`${topHelp()}\n`);
  }

  if (name === "help") {
    const topic = argv[first + 1];

    if (topic === undefined) return void process.stdout.write(`${topHelp()}\n`);

    const command = commandNamed(topic);

    if (!command) {
      throw usageError(`Unknown command "${topic}". Run ${BIN} --help.`, {
        code: "unknown_command",
        hint: `${BIN} --help`,
      });
    }

    return void process.stdout.write(`${commandHelp(topic, command)}\n`);
  }

  const command = commandNamed(name);

  if (!command) {
    const suggestion = closest(name, Object.keys(COMMANDS));

    throw usageError(
      `Unknown command "${name}".${suggestion ? ` Did you mean ${BIN} ${suggestion}?` : ""} Run ${BIN} --help.`,
      { code: "unknown_command", hint: `${BIN} --help` },
    );
  }

  const rest = [...argv.slice(0, first), ...argv.slice(first + 1)];

  // -h and --help win over everything else on the line, even bad options.
  if (rest.includes("-h") || rest.includes("--help"))
    return void process.stdout.write(`${commandHelp(name, command)}\n`);

  const options = Object.fromEntries(
    Object.entries({ ...command.options, ...GLOBAL }).map(([flag, option]) => [
      flag,
      option.short ? { type: option.type, short: option.short } : { type: option.type },
    ]),
  );

  let parsed: ReturnType<typeof parseArgs>;

  try {
    parsed = parseArgs({ args: rest, options, allowPositionals: true, strict: true });
  } catch (error) {
    const message = error instanceof Error ? error.message.split("\n")[0] : String(error);

    throw usageError(`${message} Run ${BIN} ${name} --help.`, {
      hint: `${BIN} ${name} --help`,
    });
  }

  const extra = parsed.positionals[command.arguments ?? 0];

  if (extra !== undefined) {
    throw usageError(`Unexpected argument "${extra}". Run ${BIN} ${name} --help.`, {
      hint: `${BIN} ${name} --help`,
    });
  }

  await command.run({
    values: parsed.values as Context["values"],
    positionals: parsed.positionals,
    argv: [...argv],
  });
};

const fail = (error: unknown) => {
  if (!jsonOutput) process.stderr.write(SHOW_CURSOR.repeat(process.stderr.isTTY ? 1 : 0));

  const known = error instanceof CliError ? error : undefined;
  const exitCode = known?.exitCode ?? EXIT.failure;
  process.exitCode = exitCode;

  // With --json, a script or agent reads the error as one JSON line on stderr; stdout
  // stays empty, so it never mistakes an error for data.
  if (jsonOutput) {
    process.stderr.write(
      `${jsonText({
        error: {
          code: known?.code ?? ("internal" satisfies ErrorCode),
          message: known ? known.message : error instanceof Error ? error.message : String(error),
          ...(known?.hint ? { hint: known.hint } : {}),
          retryable: known?.retryable ?? false,
          exit_code: exitCode,
          cli_version: VERSION,
        },
      })}\n`,
    );

    return;
  }

  // Tool errors are written by the server and can quote what the request named. An
  // unexpected error keeps its stack for a bug report; it can still quote a server.
  const message = known
    ? known.message
    : error instanceof Error
      ? (error.stack ?? error.message)
      : String(error);

  process.stderr.write(`${style.red("error", process.stderr)} ${clean(message)}\n`);
};

// The MCP SDK warns app developers on the console, for instance when it retries a refresh.
// The CLI reports what the person needs itself, so those notes would only be noise.
console.warn = () => {};

process.once("SIGINT", () => {
  if (process.stderr.isTTY)
    process.stderr.write(`${SHOW_CURSOR}\r\u001b[2K${style.dim("Cancelled.", process.stderr)}\n`);
  process.exit(EXIT.interrupted);
});

// A closed pipe, as in `telemetry traces | head`, is the reader being done, not an error.
process.stdout.on("error", (error: NodeJS.ErrnoException) => {
  if (error.code === "EPIPE") process.exit(process.exitCode ?? EXIT.ok);

  throw error;
});

startUpdateCheck(jsonOutput);
await main(process.argv.slice(2)).catch(fail);
await finishUpdateCheck();
