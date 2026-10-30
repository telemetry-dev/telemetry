import { Api } from "./api.ts";
import { grantedScopes, login, logout, PAYLOADS_SCOPE } from "./auth.ts";
import { CliError, EXIT, notSignedIn, usageError } from "./errors.ts";
import { printMark } from "./mark.ts";
import {
  DEFAULT_SERVER,
  loadSession,
  type Project,
  serverUrl,
  type Session,
  updateSession,
} from "./session.ts";
import {
  type Cell,
  clean,
  type Column,
  DASH,
  formatAge,
  formatCost,
  formatCount,
  formatDuration,
  formatInteger,
  formatPercent,
  fit,
  formatTimestamp,
  oneLine,
  outputWidth,
  type Paint,
  reportStatus,
  style,
  table,
} from "./terminal.ts";
import type {
  LogList,
  PayloadPage,
  ProjectList,
  SessionPage,
  Span,
  TraceDetailPage,
  TracePage,
  UsageSummary,
  UsageTotals,
} from "./tools.ts";

export type Values = Record<string, string | boolean | undefined>;

export type Context = {
  readonly values: Values;
  readonly positionals: readonly string[];
  /** The command line as typed after `telemetry`, to suggest the next page. */
  readonly argv: readonly string[];
};

export const TAGLINE = "Traces, sessions, logs, and AI usage from your terminal.";

const out = (text = "") => process.stdout.write(`${text}\n`);
const note = (text = "") => process.stderr.write(`${text}\n`);

/**
 * JSON for scripts and agents. JSON.stringify escapes C0 controls but not C1 controls
 * or bidirectional overrides, which some terminals act on, or the line and paragraph
 * separators some line readers split on, so those are escaped too.
 */
export const jsonText = (value: unknown, indent?: number) =>
  JSON.stringify(value, null, indent).replace(
    /[\u0080-\u009f\u061c\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]/g,
    (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );

const printJson = (value: unknown) => out(jsonText(value, 2));

const json = (context: Context) => context.values.json === true;

const string = (context: Context, name: string) => {
  const value = context.values[name];

  return typeof value === "string" && value !== "" ? value : undefined;
};

const signedInSession = async () => {
  const session = await loadSession(serverUrl());

  if (!session.tokens) throw notSignedIn();

  return session;
};

/** Runs `work` with an API client and closes it afterwards. */
const withApi = async <A>(work: (api: Api, session: Session) => Promise<A>) => {
  const session = await signedInSession();
  const api = new Api(session);

  try {
    return await work(api, session);
  } finally {
    await api.close();
  }
};

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Any workspace member can name a project, so a name is captured text like any other.
const toProject = (project: ProjectList["projects"][number]): Project => ({
  id: oneLine(project.id),
  name: oneLine(project.name),
});

const projectNames = (projects: ProjectList["projects"]) =>
  projects.map((project) => oneLine(project.name)).join(", ");

const matchProject = (projects: ProjectList["projects"], wanted: string): Project => {
  const byId = projects.find((project) => project.id === wanted);

  if (byId) return toProject(byId);

  const byName = projects.filter(
    (project) => oneLine(project.name).toLowerCase() === wanted.toLowerCase(),
  );

  if (byName.length === 1) return toProject(byName[0]!);

  const names = projectNames(projects);

  if (byName.length > 1) {
    throw usageError(
      `Several projects are named "${wanted}". Pass its id instead: telemetry projects`,
      { code: "project_ambiguous", hint: "telemetry projects" },
    );
  }

  throw usageError(`No project "${wanted}" in this workspace. Projects: ${names || "none"}.`, {
    code: "project_not_found",
    hint: "telemetry projects",
  });
};

/**
 * The project a command reads: --project, then TELEMETRY_DEV_PROJECT, then the default
 * set with `telemetry use`, then the workspace's only project.
 */
const resolveProject = async (api: Api, context: Context): Promise<Project> => {
  const wanted =
    string(context, "project") ?? (process.env.TELEMETRY_DEV_PROJECT?.trim() || undefined);

  if (!wanted && api.session.project) return api.session.project;

  if (wanted && UUID.test(wanted)) return { id: wanted, name: wanted };

  const { projects } = await api.call<ProjectList>("list_projects", {}, "Finding projects");

  if (wanted) return matchProject(projects, wanted);

  if (projects.length === 1) return toProject(projects[0]!);

  if (projects.length === 0)
    throw new CliError("This workspace has no projects yet. Create one at telemetry.dev.", {
      code: "no_projects",
    });

  throw usageError(
    `Choose a project with --project <name>, or set a default with \`telemetry use <name>\`. Projects: ${projectNames(projects)}.`,
    { code: "project_required", hint: "telemetry projects" },
  );
};

const limit = (context: Context, maximum = 50) => {
  const value = string(context, "limit");

  if (value === undefined) return undefined;

  const parsed = Number(value);

  if (!Number.isInteger(parsed) || parsed < 1 || parsed > maximum) {
    throw usageError(`--limit must be a whole number from 1 to ${maximum}.`);
  }

  return parsed;
};

const choice = <T extends string | number>(
  context: Context,
  name: string,
  map: Readonly<Record<string, T>>,
) => {
  const value = string(context, name);

  if (value === undefined) return undefined;

  const key = value.toLowerCase();

  if (!Object.hasOwn(map, key))
    throw usageError(`--${name} must be one of: ${Object.keys(map).join(", ")}.`);

  return map[key]!;
};

const DAYS = { "7d": 7, "30d": 30, "90d": 90 } as const;

/** `arg` quoted for a POSIX shell, so a suggested command runs as shown when pasted. */
const shellQuote = (arg: string) =>
  /^[\w@%+=:,./-]+$/.test(arg) ? arg : `'${arg.replaceAll("'", `'\\''`)}'`;

/** Suggests the same command with the next page's cursor. */
const suggestNextPage = (context: Context, cursor: string | null) => {
  if (!cursor || json(context)) return;

  const kept: string[] = [];

  for (let index = 0; index < context.argv.length; index += 1) {
    const arg = context.argv[index]!;

    if (arg === "--cursor") {
      index += 1;
      continue;
    }

    if (arg.startsWith("--cursor=")) continue;
    kept.push(arg);
  }

  const command = [...kept, "--cursor", cursor].map((arg) => shellQuote(clean(arg))).join(" ");

  note(style.dim(`\nMore: telemetry ${command}`, process.stderr));
};

const statusCell = (status: string): Cell =>
  status === "error" ? { text: "error", paint: style.red } : { text: "ok", paint: style.green };

const tokens = (input: number | null, output: number | null) =>
  input === null && output === null ? null : (input ?? 0) + (output ?? 0);

const dim: Paint = style.dim;

const nothing = (what: string) => note(style.dim(what, process.stderr));

export async function home() {
  const session = await loadSession(serverUrl());
  const signedIn = Boolean(session.tokens);

  out();
  await printMark(TAGLINE, { animate: !signedIn });
  out();

  if (!signedIn) {
    out(`Get started: ${style.bold("telemetry login")}`);
    out(style.dim("Then run telemetry --help to see every command."));
    out();

    return;
  }

  const who = session.email ? `Signed in as ${style.bold(oneLine(session.email))}` : "Signed in";
  const project = session.project ? ` · project ${style.bold(oneLine(session.project.name))}` : "";
  out(`${who}${project}`);
  out();

  const suggestions: [string, string][] = [
    ["telemetry traces --errors", "Recent failed runs"],
    ["telemetry trace <id>", "A run's spans, timing, and logs"],
    ["telemetry sessions", "Conversations and agent runs"],
    ["telemetry usage", "Requests, tokens, and cost"],
    ["telemetry logs --level error", "Recent error logs"],
  ];

  const width = Math.max(...suggestions.map(([command]) => command.length));

  for (const [command, summary] of suggestions)
    out(`  ${command.padEnd(width)}  ${style.dim(summary)}`);
  out();
  out(style.dim("Run telemetry --help for every command."));
  out();
}

export async function loginCommand(context: Context) {
  const server = serverUrl();
  const session = await loadSession(server);
  const stream = process.stderr;

  const finish = typeof context.values.finish === "string" ? context.values.finish : undefined;

  if (process.stderr.isTTY && finish === undefined) {
    stream.write("\n");
    await printMark(TAGLINE, { animate: true, stream });
    stream.write("\n");
  }

  const result = await login({
    session,
    browser: context.values["no-browser"] !== true,
    finish,
    onAuthorizationUrl: (url, opened) => {
      note(
        opened
          ? "Opening your browser to sign in to telemetry.dev."
          : "Open this link in a browser to sign in to telemetry.dev:",
      );
      note(opened ? style.dim(`If it doesn't open, visit: ${url.href}`, stream) : url.href);
      note();
    },
    onWaiting: () => {
      reportStatus("blocked", "auth");
      note(style.dim("Waiting for you to choose a workspace and select Allow…", stream));
    },
  }).catch((error: unknown) => {
    reportStatus("error");
    throw error;
  });

  if (result.status === "pending") {
    note(
      "After you select Allow, the browser opens an http://127.0.0.1 address that doesn't load. Copy the whole address and run:",
    );
    note();
    note(`  telemetry login --finish '<address>'`);

    return;
  }

  const signedIn = result.session;

  reportStatus("done");

  const api = new Api(signedIn);

  try {
    const { projects } = await api.call<ProjectList>("list_projects", {}, "Loading projects");

    if (projects.length === 1) {
      const project = toProject(projects[0]!);

      signedIn.project = project;
      await updateSession(signedIn.server, (saved) => ({ ...saved, project }));
    }

    note();
    note(
      `${style.green("✓", stream)} Signed in${signedIn.email ? ` as ${style.bold(signedIn.email, stream)}` : ""}`,
    );

    if (!grantedScopes(signedIn).has(PAYLOADS_SCOPE)) {
      note(
        style.dim(
          "  Captured prompts and outputs stay hidden. Sign in again and allow them to use `telemetry payload`.",
          stream,
        ),
      );
    }

    if (signedIn.project) {
      note(`  Project ${style.bold(signedIn.project.name, stream)}`);
      note();
      note(
        `Try ${style.bold("telemetry traces", stream)} or ${style.bold("telemetry usage", stream)}.`,
      );
    } else if (projects.length > 1) {
      note(`  ${projects.length} projects: ${projectNames(projects)}`);
      note();
      note(
        `Choose a default: ${style.bold(`telemetry use ${shellQuote(toProject(projects[0]!).name)}`, stream)}`,
      );
    } else {
      note("  This workspace has no projects yet. Create one at telemetry.dev.");
    }
  } finally {
    await api.close();
  }
}

export async function logoutCommand() {
  const session = await loadSession(serverUrl());

  if (!session.tokens) {
    note("You're not signed in.");

    return;
  }

  const { revoked } = await logout(session);
  note(
    `${style.green("✓", process.stderr)} Signed out${session.email ? ` of ${session.email}` : ""}.`,
  );

  if (!revoked)
    note(
      style.dim(
        "The server didn't confirm the sign-out, so the sign-in may still work elsewhere. To end it, disconnect the CLI in account settings at telemetry.dev.",
        process.stderr,
      ),
    );
}

export async function statusCommand(context: Context) {
  const server = serverUrl();
  const session = await loadSession(server);
  const scopes = grantedScopes(session);

  if (json(context)) {
    printJson({
      signed_in: Boolean(session.tokens),
      server,
      email: session.email ?? null,
      project: session.project ?? null,
      payloads_allowed: scopes.has(PAYLOADS_SCOPE),
    });
  } else if (session.tokens) {
    out(
      `${style.green("✓")} Signed in${session.email ? ` as ${style.bold(oneLine(session.email))}` : ""}`,
    );
    out(
      `  Project   ${session.project ? oneLine(session.project.name) : style.dim("none set · telemetry use <name>")}`,
    );
    out(
      `  Payloads  ${scopes.has(PAYLOADS_SCOPE) ? "allowed" : style.dim("not allowed · sign in again to allow")}`,
    );

    if (server !== DEFAULT_SERVER) out(`  Server    ${server}`);
  } else {
    out(`Not signed in. Run ${style.bold("telemetry login")}.`);
  }

  if (!session.tokens) process.exitCode = EXIT.auth;
}

export async function projectsCommand(context: Context) {
  await withApi(async (api, session) => {
    const result = await api.call<ProjectList>("list_projects", {}, "Loading projects");

    if (json(context)) return printJson(result);

    if (result.projects.length === 0) return nothing("This workspace has no projects yet.");

    out(
      table(
        [
          { header: "" },
          { header: "NAME", flex: true },
          { header: "ID" },
          { header: "RETENTION", align: "right" },
          { header: "CREATED" },
        ],
        result.projects.map((project) => [
          project.id === session.project?.id ? { text: "●", paint: style.green } : "",
          oneLine(project.name),
          { text: oneLine(project.id), paint: dim },
          `${project.retention_days}d`,
          formatAge(project.created_at),
        ]),
      ),
    );
  });
}

export async function useCommand(context: Context) {
  const wanted = context.positionals[0];

  if (!wanted) throw usageError("Name the project to use: telemetry use <name|id>");

  await withApi(async (api, session) => {
    const { projects } = await api.call<ProjectList>("list_projects", {}, "Finding project");
    const project = matchProject(projects, wanted);

    await updateSession(session.server, (saved) => ({ ...saved, project }));

    if (json(context)) return printJson({ project });

    note(
      `${style.green("✓", process.stderr)} Default project is now ${style.bold(project.name, process.stderr)}.`,
    );
  });
}

export async function tracesCommand(context: Context) {
  await withApi(async (api) => {
    const project = await resolveProject(api, context);

    const page = await api.call<TracePage>(
      "search_traces",
      {
        project_id: project.id,
        name_contains: string(context, "name"),
        status: context.values.errors ? "error" : undefined,
        environment: string(context, "env"),
        session_id: string(context, "session"),
        user_id: string(context, "user"),
        limit: limit(context),
        cursor: string(context, "cursor"),
      },
      "Searching traces",
    );

    if (json(context)) return printJson(page);

    if (page.traces.length === 0) return nothing("No traces match.");

    out(
      table(
        [
          { header: "ID" },
          { header: "NAME", flex: true },
          { header: "STATUS" },
          { header: "DURATION", align: "right" },
          { header: "TOKENS", align: "right" },
          { header: "COST", align: "right" },
          { header: "MODEL" },
          { header: "STARTED" },
        ],
        page.traces.map((trace) => [
          { text: oneLine(trace.id), paint: dim },
          oneLine(trace.name),
          statusCell(trace.status),
          formatDuration(trace.duration_ms),
          formatCount(tokens(trace.input_tokens, trace.output_tokens)),
          formatCost(trace.cost_usd),
          oneLine(trace.models.join(", ")) || DASH,
          formatAge(trace.started_at),
        ]),
      ),
    );

    suggestNextPage(context, page.next_cursor);
  });
}

export async function sessionsCommand(context: Context) {
  const RANGE = { "24h": 24, "72h": 72, "7d": 168, "30d": 720 } as const;

  await withApi(async (api) => {
    const project = await resolveProject(api, context);

    const page = await api.call<SessionPage>(
      "search_sessions",
      {
        project_id: project.id,
        range_hours: choice(context, "range", RANGE),
        contains: string(context, "contains"),
        agent: string(context, "agent"),
        model: string(context, "model"),
        user_id: string(context, "user"),
        errors_only: context.values.errors === true ? true : undefined,
        limit: limit(context),
        cursor: string(context, "cursor"),
      },
      "Searching sessions",
    );

    if (json(context)) return printJson(page);

    if (page.sessions.length === 0)
      return nothing(
        string(context, "range") === "30d"
          ? "No sessions match."
          : "No sessions match. Widen the window with --range 7d or --range 30d.",
      );

    out(
      table(
        [
          { header: "ID", flex: true },
          { header: "AGENT" },
          { header: "TRACES", align: "right" },
          { header: "ERRORS", align: "right" },
          { header: "TOKENS", align: "right" },
          { header: "COST", align: "right" },
          { header: "USER" },
          { header: "ACTIVE" },
        ],
        page.sessions.map((session) => [
          oneLine(session.id),
          oneLine(session.agent_name ?? "") || DASH,
          formatInteger(session.trace_count),
          session.error_trace_count > 0
            ? { text: formatInteger(session.error_trace_count), paint: style.red }
            : { text: "0", paint: dim },
          formatCount(tokens(session.input_tokens, session.output_tokens)),
          formatCost(session.cost_usd),
          oneLine(session.user_id ?? "") || DASH,
          formatAge(session.last_active_at),
        ]),
      ),
    );

    suggestNextPage(context, page.next_cursor);
  });
}

// get_trace returns at most 100 spans and logs a page; this many pages covers 2,000.
const MAX_TRACE_PAGES = 20;

// Bars are three-quarter blocks, like the bands of the mark, so stacked rows keep a gap.
const BAR = "▆";

/** A bar from `offset` for `duration` on a timeline `total` ms long, at least one column. */
const timelineBar = (offset: number, duration: number, total: number, width: number) => {
  if (total <= 0) return BAR;

  const start = Math.min(width - 1, Math.max(0, Math.round((offset / total) * width)));
  const length = Math.min(width - start, Math.max(1, Math.round((duration / total) * width)));

  return " ".repeat(start) + BAR.repeat(length);
};

const spanPaint = (span: Span): Paint =>
  span.status === "error"
    ? style.red
    : span.model
      ? style.cyan
      : span.tool_name
        ? style.yellow
        : style.dim;

type TreeRow = { span: Span; prefix: string };

/** Spans in tree order with their guide lines. A span whose parent is missing is a root. */
const spanTree = (spans: readonly Span[]): TreeRow[] => {
  const ids = new Set(spans.map((span) => span.id));
  const children = new Map<string | null, Span[]>();

  for (const span of spans) {
    const parent = span.parent_span_id && ids.has(span.parent_span_id) ? span.parent_span_id : null;
    children.set(parent, [...(children.get(parent) ?? []), span]);
  }

  const rows: TreeRow[] = [];
  const visited = new Set<string>();

  // An explicit stack: a deep agent loop could nest far past the call stack.
  const stack: { span: Span; guide: string; last: boolean; root: boolean }[] = (
    children.get(null) ?? []
  )
    .map((span) => ({ span, guide: "", last: true, root: true }))
    .reverse();

  while (stack.length > 0) {
    const { span, guide, last, root } = stack.pop()!;

    if (visited.has(span.id)) continue;
    visited.add(span.id);
    rows.push({ span, prefix: root ? "" : `${guide}${last ? "└─ " : "├─ "}` });

    const kids = children.get(span.id) ?? [];
    const childGuide = root ? "" : `${guide}${last ? "   " : "│  "}`;

    for (let index = kids.length - 1; index >= 0; index -= 1) {
      stack.push({
        span: kids[index]!,
        guide: childGuide,
        last: index === kids.length - 1,
        root: false,
      });
    }
  }

  return rows;
};

const spanLabel = (span: Span) => {
  const name = oneLine(span.name);
  const extras = [span.model, span.tool_name, span.agent_name]
    .filter((value): value is string => Boolean(value) && !name.includes(value!))
    .map(oneLine);

  return extras.length > 0 ? `${name}  ${extras.join(" · ")}` : name;
};

const levelPaint = (level: string): Paint =>
  level === "error" || level === "fatal"
    ? style.red
    : level === "warn"
      ? style.yellow
      : level === "debug"
        ? style.dim
        : (text) => text;

export async function traceCommand(context: Context) {
  const traceId = context.positionals[0];

  if (!traceId) throw usageError("Name the trace to show: telemetry trace <trace-id>");

  await withApi(async (api) => {
    const project = await resolveProject(api, context);
    let page = await api.call<TraceDetailPage>(
      "get_trace",
      { project_id: project.id, trace_id: traceId },
      "Loading trace",
    );
    const spans = [...page.spans];
    const logs = [...page.logs];
    let pages = 1;

    while (page.next_cursor && pages < MAX_TRACE_PAGES) {
      page = await api.call<TraceDetailPage>(
        "get_trace",
        { project_id: project.id, trace_id: traceId, cursor: page.next_cursor },
        `Loading spans (${spans.length} of ${page.span_count})`,
      );
      spans.push(...page.spans);
      logs.push(...page.logs);
      pages += 1;
    }

    const complete = page.next_cursor === null;

    if (json(context)) {
      return printJson({
        trace: page.trace,
        span_count: page.span_count,
        log_count: page.log_count,
        complete,
        spans,
        logs,
      });
    }

    renderTrace(page, spans, logs);

    if (!complete) {
      note(
        style.dim(
          `\nShowing the first ${spans.length} of ${page.span_count} spans. The CLI reads up to ${MAX_TRACE_PAGES * 100}, and --json stops at the same point with complete: false.`,
          process.stderr,
        ),
      );
    }
  });
}

function renderTrace(page: TraceDetailPage, spans: readonly Span[], logs: TraceDetailPage["logs"]) {
  const { trace } = page;
  const failed = trace.status === "error";
  const icon = failed ? style.red("✗") : style.green("✓");
  const errorType = failed && trace.error_type ? `  ${style.red(oneLine(trace.error_type))}` : "";

  out(`${icon} ${style.bold(oneLine(trace.name))}${errorType}`);

  const facts = [
    formatDuration(trace.duration_ms),
    trace.input_tokens !== null || trace.output_tokens !== null
      ? `${formatCount(tokens(trace.input_tokens, trace.output_tokens))} tokens`
      : null,
    trace.cost_usd !== null ? formatCost(trace.cost_usd) : null,
    trace.environment,
    trace.service_name,
    formatAge(trace.started_at),
  ].filter((fact): fact is string => Boolean(fact));

  out(`  ${facts.map(oneLine).join(style.dim(" · "))}`);

  const ids = [
    `trace ${oneLine(trace.id)}`,
    trace.session_id ? `session ${oneLine(trace.session_id)}` : null,
    trace.user_id ? `user ${oneLine(trace.user_id)}` : null,
  ].filter(Boolean);

  out(style.dim(`  ${ids.join("  ")}`));
  out();

  if (spans.length === 0) {
    nothing("This trace has no spans yet.");

    return;
  }

  const starts = spans.map((span) => Date.parse(span.started_at));
  const origin = Math.min(...starts);
  const end = Math.max(...spans.map((span, index) => starts[index]! + (span.duration_ms ?? 0)));
  const total = Math.max(end - origin, trace.duration_ms ?? 0);

  const terminal = process.stdout.isTTY === true;
  const timelineWidth = terminal ? Math.min(60, Math.max(16, outputWidth() - 72)) : 0;

  const columns: Column[] = [
    { header: "SPAN", flex: true },
    { header: "DURATION", align: "right" },
    { header: "TOKENS", align: "right" },
    terminal ? { header: "TIMELINE" } : { header: "START", align: "right" },
  ];

  const rows: (Cell | string)[][] = [];
  // Error text goes on its own line under the span, outside the columns, so a long
  // message never widens the table.
  const errors = new Map<number, string>();

  for (const { span, prefix } of spanTree(spans)) {
    const offset = Date.parse(span.started_at) - origin;
    const paint = spanPaint(span);
    const failedSpan = span.status === "error";
    const label = `${prefix}${spanLabel(span)}${failedSpan ? `  ✗ ${oneLine(span.error_type ?? "error")}` : ""}`;

    rows.push([
      { text: label, paint: failedSpan ? style.red : undefined },
      formatDuration(span.duration_ms),
      { text: formatCount(tokens(span.input_tokens, span.output_tokens)), paint: dim },
      terminal
        ? { text: timelineBar(offset, span.duration_ms ?? 0, total, timelineWidth), paint }
        : offset === 0
          ? "+0"
          : `+${formatDuration(offset)}`,
    ]);

    if (failedSpan && span.error) {
      const guide = prefix.replace(/├─ $/, "│  ").replace(/└─ $/, "   ");
      errors.set(rows.length - 1, `${guide}   ${oneLine(span.error)}`);
    }
  }

  const lines = table(columns, rows).split("\n");

  // Line 0 is the header, so row i is line i + 1.
  out(
    lines
      .flatMap((line, index) => {
        const error = errors.get(index - 1);

        return error === undefined
          ? [line]
          : [line, style.dim(style.red(fit(error, outputWidth())))];
      })
      .join("\n"),
  );

  if (logs.length > 0) {
    out();
    out(style.bold(`Logs ${style.dim(`(${formatInteger(page.log_count)})`)}`));

    for (const log of logs) {
      const level = oneLine(log.level).toUpperCase().padEnd(5);
      out(
        `${style.dim(formatTimestamp(log.timestamp))}  ${levelPaint(log.level)(level)}  ${oneLine(log.message)}`,
      );
    }
  }
}

export async function logsCommand(context: Context) {
  const LEVELS = {
    all: "all",
    debug: "debug",
    info: "info",
    warn: "warn",
    error: "error",
  } as const;

  await withApi(async (api) => {
    const project = await resolveProject(api, context);

    const result = await api.call<LogList>(
      "list_logs",
      {
        project_id: project.id,
        level: choice(context, "level", LEVELS),
        range_days: choice(context, "range", DAYS),
        limit: limit(context),
      },
      "Loading logs",
    );

    if (json(context)) return printJson(result);

    const { counts } = result;

    out(
      style.dim(
        `${formatInteger(counts.totalCount)} logs in the last ${result.range_days} days · ${formatInteger(counts.errorCount)} error · ${formatInteger(counts.warnCount)} warn · ${formatInteger(counts.infoCount)} info · ${formatInteger(counts.debugCount)} debug`,
      ),
    );

    if (result.logs.length === 0) return nothing("No logs match.");

    out();
    out(
      table(
        [
          { header: "TIME" },
          { header: "LEVEL" },
          { header: "SERVICE" },
          { header: "MESSAGE", flex: true },
          { header: "TRACE" },
        ],
        result.logs.map((log) => [
          { text: formatTimestamp(log.timestamp), paint: dim },
          { text: oneLine(log.level).toUpperCase(), paint: levelPaint(log.level) },
          oneLine(log.service_name ?? "") || DASH,
          oneLine(log.message),
          { text: oneLine(log.trace_id ?? "") || DASH, paint: dim },
        ]),
      ),
    );

    if (result.has_more)
      note(
        style.dim(
          "\nShowing the newest logs. Raise --limit (up to 50) or narrow with --level.",
          process.stderr,
        ),
      );
  });
}

/** The change from the previous range; red or green only where up is worse. */
const change = (current: number, previous: number | undefined, worseWhenUp?: boolean) => {
  if (previous === undefined) return "";

  if (previous === 0) return current === 0 ? style.dim("  –") : style.dim("  new");

  const delta = ((current - previous) / previous) * 100;

  if (Math.abs(delta) < 0.5) return style.dim("  ±0%");

  const text = `  ${delta > 0 ? "↑" : "↓"} ${formatPercent(Math.abs(delta))}`;

  if (worseWhenUp === undefined) return style.dim(text);

  return delta > 0 === worseWhenUp ? style.red(text) : style.green(text);
};

export async function usageCommand(context: Context) {
  const GROUPS = {
    model: "model",
    provider: "provider",
    user: "user",
    operation: "operation",
  } as const;

  await withApi(async (api) => {
    const project = await resolveProject(api, context);

    const summary = await api.call<UsageSummary>(
      "get_usage_summary",
      {
        project_id: project.id,
        range_days: choice(context, "range", DAYS),
        group_by: choice(context, "by", GROUPS),
      },
      "Summarizing usage",
    );

    if (json(context)) return printJson(summary);

    const now: UsageTotals = summary.current;
    const before = summary.previous ?? undefined;
    const comparing = before ? ` vs the ${summary.range_days} days before` : "";

    out(
      `${style.bold(oneLine(project.name))}  ${style.dim(`last ${summary.range_days} days${comparing}`)}`,
    );
    out();

    const metrics: [string, string, string][] = [
      [
        "Requests",
        formatCount(now.totalRequests),
        change(now.totalRequests, before?.totalRequests),
      ],
      [
        "Errors",
        formatCount(now.errorCount),
        `${style.dim(`  ${formatPercent(now.errorRate)} of requests`)}${change(now.errorCount, before?.errorCount, true)}`,
      ],
      [
        "p95 latency",
        formatDuration(now.p95DurationMs),
        change(now.p95DurationMs, before?.p95DurationMs, true),
      ],
      [
        "Tokens",
        formatCount(now.totalInputTokens + now.totalOutputTokens),
        `${style.dim(`  ${formatCount(now.totalInputTokens)} in · ${formatCount(now.totalOutputTokens)} out`)}${change(now.totalInputTokens + now.totalOutputTokens, before && before.totalInputTokens + before.totalOutputTokens)}`,
      ],
      ["Cost", formatCost(now.totalCostUsd), change(now.totalCostUsd, before?.totalCostUsd)],
      ["Active users", formatCount(now.activeUsers), change(now.activeUsers, before?.activeUsers)],
    ];

    const labelWidth = Math.max(...metrics.map(([label]) => label.length));
    const valueWidth = Math.max(...metrics.map(([, value]) => value.length));

    for (const [label, value, detail] of metrics) {
      out(
        `${style.dim(label.padEnd(labelWidth))}  ${style.bold(value.padStart(valueWidth))}${detail}`,
      );
    }

    if (summary.data_incomplete)
      note(
        style.dim("\nSome telemetry was unavailable, so these totals may be low.", process.stderr),
      );

    if (summary.breakdown.length === 0) return;

    const busiest = Math.max(...summary.breakdown.map((row) => row.totalRequests));
    const barWidth = process.stdout.isTTY ? 12 : 0;

    out();
    out(
      table(
        [
          { header: `BY ${oneLine(summary.group_by).toUpperCase()}`, flex: true },
          { header: "REQUESTS", align: "right" },
          ...(barWidth ? [{ header: "" }] : []),
          { header: "ERRORS", align: "right" },
          { header: "P95", align: "right" },
          { header: "TOKENS", align: "right" },
          { header: "COST", align: "right" },
        ],
        summary.breakdown.map((row) => [
          oneLine(row.category ?? "") || style.dim("(none)"),
          formatCount(row.totalRequests),
          ...(barWidth
            ? [{ text: timelineBar(0, row.totalRequests, busiest, barWidth), paint: style.cyan }]
            : []),
          row.errorCount > 0
            ? { text: formatPercent(row.errorRate), paint: style.red }
            : { text: "0%", paint: dim },
          formatDuration(row.p95DurationMs),
          formatCount(row.totalTokens),
          formatCost(row.totalCostUsd),
        ]),
      ),
    );

    if (summary.breakdown.length >= summary.breakdown_limit) {
      note(
        style.dim(`\nThe ${summary.breakdown_limit} busiest groups by requests.`, process.stderr),
      );
    }
  });
}

// Each page holds up to 20,000 characters, and stored payloads are capped well below this.
const MAX_PAYLOAD_PAGES = 50;
const FIELDS = { input: "input", output: "output", error: "error" } as const;

type Payload = Omit<PayloadPage, "offset" | "next_offset"> & {
  /** False when the CLI stopped reading before the end of the stored text. */
  readonly complete: boolean;
};

const readPayload = async (
  api: Api,
  args: Record<string, unknown>,
  field: keyof typeof FIELDS,
): Promise<Payload> => {
  let offset: number | null = 0;
  let text = "";
  let page: PayloadPage | undefined;

  for (let pages = 0; offset !== null && pages < MAX_PAYLOAD_PAGES; pages += 1) {
    page = await api.call<PayloadPage>(
      "get_span_payload",
      { ...args, field, offset, max_characters: 20_000 },
      `Reading ${field}`,
    );
    text += page.text;
    offset = page.next_offset;
  }

  return {
    field,
    total_characters: page?.total_characters ?? text.length,
    stored_text_truncated: page?.stored_text_truncated ?? false,
    complete: offset === null,
    text,
  };
};

const payloadWarning = (payload: Payload) =>
  !payload.complete
    ? `The CLI read the first ${formatInteger(payload.text.length)} of ${formatInteger(payload.total_characters)} characters.`
    : payload.stored_text_truncated
      ? "telemetry.dev stored only the start of this text."
      : undefined;

/** Pretty-prints JSON text, so a captured message list reads as structure, not one long line. */
const readable = (text: string) => {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
};

export async function payloadCommand(context: Context) {
  const [traceId, spanId] = context.positionals;

  if (!traceId || !spanId) {
    throw usageError(
      "Name the span: telemetry payload <trace-id> <span-id>. Span ids are in `telemetry trace <trace-id> --json`.",
    );
  }

  const requested = choice(context, "field", FIELDS);

  await withApi(async (api, session) => {
    if (!grantedScopes(session).has(PAYLOADS_SCOPE)) {
      throw new CliError(
        'This sign-in can\'t read captured prompts and outputs. Run `telemetry login`, check "Also read captured prompts, outputs, and tool arguments", and select Allow.',
        { exitCode: EXIT.auth, code: "payloads_not_allowed", hint: "telemetry login" },
      );
    }

    const project = await resolveProject(api, context);
    const args = { project_id: project.id, trace_id: traceId, span_id: spanId };
    const fields = requested ? [requested] : (["input", "output"] as const);
    const payloads: Payload[] = [];

    for (const field of fields) payloads.push(await readPayload(api, args, field));

    if (json(context))
      return printJson(Object.fromEntries(payloads.map((payload) => [payload.field, payload])));

    // One field is plain text, ready for a pipe or a file; any warning goes to stderr.
    if (requested) {
      const [payload] = payloads;
      process.stdout.write(clean(payload!.text));

      if (process.stdout.isTTY && !payload!.text.endsWith("\n")) out();

      const warning = payloadWarning(payload!);

      if (warning) note(style.dim(warning, process.stderr));

      return;
    }

    for (const payload of payloads) {
      out(style.bold(payload.field === "input" ? "Input" : "Output"));
      out(payload.text === "" ? style.dim("(empty)") : clean(readable(payload.text)));

      const warning = payloadWarning(payload);

      if (warning) out(style.dim(`(${warning})`));
      out();
    }
  });
}
