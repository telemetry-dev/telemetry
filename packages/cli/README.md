# @telemetry-dev/cli

The telemetry.dev command line. Search traces, sessions, logs, and AI usage from a terminal, a script, or an agent's shell, without connecting an MCP client.

The CLI signs in to the same read-only API that the telemetry.dev MCP server gives agents, with the same permissions. The `--json` output of each command is the result of the matching MCP tool.

## Install

```sh
npm install --global @telemetry-dev/cli
```

Node.js 20 or later is required. The CLI doesn't update itself; to update it, run `npm install --global @telemetry-dev/cli@latest`. It tells you when a newer version is out: see [Updates](#updates).

To try it without installing, start each command with `npx @telemetry-dev/cli` instead of `telemetry`. The CLI's own hints then use the same prefix, and the same goes for `pnpm dlx`, `yarn dlx`, and `bunx`.

```sh
npx @telemetry-dev/cli login
npx @telemetry-dev/cli traces
```

## Sign in

```sh
telemetry login
```

`login` opens your browser. Choose a workspace and select **Allow**. To let the CLI read captured prompts, outputs, and tool arguments, also check **Also read captured prompts, outputs, and tool arguments**. Without that permission, every command except `payload` works.

Over SSH, in a container, or anywhere without a browser, run `telemetry login --no-browser`. Open the printed link on any device. After you select **Allow**, the browser tries to load an address that starts with `http://127.0.0.1`. The page doesn't load. Copy the whole address and paste it into the terminal. `telemetry login` takes a pasted address too, for when the browser doesn't open or you sign in on another device.

In a shell with no terminal to paste into, such as an agent's, `telemetry login --no-browser` prints the link and exits. After you select **Allow**, finish the sign-in with the address the browser ended on:

```sh
telemetry login --finish 'http://127.0.0.1:53124/callback?code=…'
```

On Windows, use double quotes instead, which cmd.exe and PowerShell both read. The commands the CLI prints use them there.

A script or agent can run `telemetry login --json` instead. It never opens a browser or waits. It prints the link to pass on to a person, then exits:

```json
{
  "status": "pending",
  "authorization_url": "https://telemetry.dev/api/auth/oauth2/authorize?…",
  "next": "telemetry login --finish '<address>'"
}
```

`telemetry login --finish '<address>' --json` then prints the sign-in, with the same fields as `telemetry status --json` and `"status": "signed_in"`. The default project stays when the chosen workspace still has it. Otherwise, when the workspace has exactly one project, it becomes the default. `project` is `null` in any other case, and also when the projects couldn't be loaded.

A sign-in covers one workspace. To read another workspace, run `telemetry login` again and choose it. The new sign-in replaces the old one, and the CLI asks telemetry.dev to revoke the old one.

## Use

```sh
telemetry projects                       # list projects
telemetry use support-bot                # set the default project
telemetry traces --errors                # recent failed traces
telemetry trace 4bf92f3577b34da6a3ce929d0e0e4736  # failures, repeated calls, spans, logs, and a dashboard link
telemetry sessions --range 7d --errors   # conversations with a failed trace
telemetry logs --level error
telemetry usage --range 30d --by user
telemetry payload <trace-id> <span-id> --field output
telemetry findings --since 2h            # alerts and failed checks; exits 3 when there are any
```

Run `telemetry <command> --help` to see each command's options and examples. Options go after the command: `telemetry traces --json`, not `telemetry --json traces`.

In human output, `telemetry trace` shows a summary above the span tree. It lists up to 5 of the failed spans where failures began, with their path. It also lists up to 5 of the operations that ran more than once, with their wall time, which counts overlapping calls once. In the span tree, each failed span has its id and error under it, for `telemetry payload`. When they list results, `traces`, `sessions`, and `findings` end with a likely next command on stderr. `trace` ends with the `payload` command that reads the full error, when a span where a failure began has an error message and the sign-in can read payloads.

A command reads its project from `--project <name|id>`. Without that option, it uses `TELEMETRY_DEV_PROJECT`, then the default that you set with `telemetry use`. If the workspace has only one project, the command uses it.

List commands return 20 results by default. Use `--limit` to return from 1 to 50 results. When more results are available, the command prints the same command with `--cursor` for the next page. On Windows, when an option holds text that no quoting keeps as typed, such as `%` or `$`, it prints only the `--cursor` to add. Keep the other options as they are when you pass `--cursor`. Only `--limit` may change between pages.

### Findings

`telemetry findings` lists what telemetry.dev found wrong in the project, newest first: alerts that opened, such as a runaway trace or a tool failure cluster, and failed checks, which are fail scores from an automatic pass/fail evaluator such as an LLM judge. Each finding has a link to its alert page, or to the trace at the span that was checked. Without `--since`, it lists the last 7 days. `--kind alert` or `--kind check` lists one kind.

The command exits with 3 when the window has at least one finding, also with `--json`, and with 0 when it has none. To check a deploy from CI or a coding agent, pass the time of the deploy:

```sh
telemetry findings --since 2026-11-06T14:00:00Z --json
```

### Time windows

`traces`, `sessions`, `logs`, and `findings` take `--since` and `--until` to choose the time they search:

```sh
telemetry traces --errors --since 2h
telemetry logs --level warn --since 30m
telemetry sessions --since 2026-10-29 --until 2026-10-30
```

Each option takes one of these forms:

| Form                | Example                                             | Meaning                                                                |
| ------------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| Duration            | `30m`, `2h`, `7d`                                   | That many minutes, hours, or days before now, by telemetry.dev's clock |
| Date                | `2026-10-29`                                        | The start of that day, in your time zone                               |
| Time without a zone | `2026-10-29T14:00`, `2026-10-29T14:00:30`           | That time in your time zone                                            |
| Time with a zone    | `2026-10-29T14:00:00Z`, `2026-10-29T14:00:00-04:00` | That instant, in RFC 3339 form, with seconds                           |
| `now`               | `--until now`                                       | The same as leaving out `--until`                                      |

A duration has up to five digits. The CLI sends a date or a time without a zone with your zone's offset on that day, so a day that changes to or from daylight saving time still starts at the start of your local day. Without `--since`, a command uses its usual window, and without `--until`, the window ends now. telemetry.dev doesn't search past the project's retention: when a window reaches further back, it starts at the retention limit, and the CLI says so on stderr.

`--range` on `sessions` and `logs` still picks one of the usual windows, and it can't be combined with `--since` or `--until`.

`traces` also takes `--framework` and `--service`, which match the trace, and `--model`, `--provider`, `--operation`, `--agent`, and `--tool`, which match traces with a span that started in the window. `--tool` matches tool calls by the tool name telemetry.dev records, or by the span name when there is none.

## Scripts and agents

Add `--json` to any command for machine-readable output on stdout. Messages and progress go to stderr, so `telemetry traces --json > traces.json` writes only data. List commands print the MCP tool's result as returned, with the fields that the [MCP server documentation](https://telemetry.dev/docs/platform/mcp-server) describes.

The results of `traces`, `sessions`, `logs`, and `findings` include the window that telemetry.dev searched, with both ends resolved to ISO times. `capped_by_retention` is true when the window starts at the start of the project's retention, so older data wasn't searched. That is also the case for `traces` without `--since`, which searches the whole retention:

```json
"window": { "since": "2026-10-30T12:00:00.000Z", "until": "2026-10-30T14:00:00.000Z", "capped_by_retention": false }
```

`logs` and `findings` results also have `next_cursor`, which is `null` on the last page. In `logs` results, `range_days` is `null` when the command used `--since` or `--until`.

An older telemetry.dev deployment may ignore `--since`, `--until`, the `traces` filters, and `logs --cursor` and return unfiltered results, or the first page of logs again. So when the CLI sends one of them and the result has no `window`, or no `next_cursor` for `logs --cursor`, the command fails with `server_unsupported` and prints no results. A deployment that refuses `--agent` or `--tool` because it doesn't know them also fails with `server_unsupported`; one that refuses another argument fails with `tool_error`. `findings` on a deployment from before findings also fails with `server_unsupported`.

`telemetry trace` and `telemetry payload` read every page of a trace or a captured text, up to a limit: 2,000 spans for a trace, and 1,000,000 characters for each payload field. Their JSON output has `complete: false` when the limit cut it short, and human output says so on stderr.

Captured text comes from your users, models, and tools. telemetry.dev removes ANSI color and cursor sequences, bidirectional overrides, and Unicode tag characters from results before they reach the CLI; it does not clean the text of error messages. Human-readable output, including `payload --field`, removes the remaining escape sequences and control characters. JSON output escapes them instead of removing them, so use `--json` when you need everything the server sent.

| Exit code | Meaning                                                         |
| --------- | --------------------------------------------------------------- |
| 0         | Success                                                         |
| 1         | The request failed                                              |
| 2         | The command or its options were not valid                       |
| 3         | `findings` found at least one finding in its window             |
| 4         | You are not signed in, the sign-in expired, or access is denied |
| 130       | Interrupted with Ctrl-C                                         |

With `--json`, an error is one line of JSON on stderr, and stdout stays empty. Ctrl-C exits with 130 and no JSON, and `status --json` reports a missing sign-in as data on stdout with exit code 4. `status` checks a saved sign-in with telemetry.dev, so one that expired, was revoked, or lost access to the workspace is reported the same way, with `"signed_in": false`. When the check itself fails, for example offline or while telemetry.dev can't renew the sign-in, `status` still reports the saved sign-in and says on stderr that it couldn't check it. With `--json`, `"sign_in_verified": false` says the same.

```text
{"error":{"code":"not_signed_in","message":"You're not signed in. Run telemetry login first.","hint":"telemetry login","retryable":false,"exit_code":4,"cli_version":"0.1.0"}}
```

Branch on `code`, not on `message`, whose wording can change. `hint` is a command to run next, when there is one. `telemetry login` needs a person at a browser, so an agent should run `telemetry login --json` and pass the link on rather than run `telemetry login`. `retryable` is true when running the same command again may succeed. New codes may be added, and existing codes keep their meaning.

| Code                   | Exit | Meaning                                                                          |
| ---------------------- | ---- | -------------------------------------------------------------------------------- |
| `invalid_arguments`    | 2    | An option or argument is not valid, by the CLI's checks or telemetry.dev's       |
| `unknown_command`      | 2    | There is no command with that name                                               |
| `project_not_found`    | 2    | No project in the workspace has that name or id                                  |
| `invalid_cursor`       | 2    | `--cursor` isn't from an earlier page of this command with the same options      |
| `project_ambiguous`    | 2    | Several projects have that name; pass its id                                     |
| `project_required`     | 2    | The workspace has several projects and none was chosen                           |
| `no_projects`          | 1    | The workspace has no projects yet                                                |
| `not_signed_in`        | 4    | No sign-in is saved for this deployment                                          |
| `sign_in_expired`      | 4    | The sign-in expired or was disconnected                                          |
| `payloads_not_allowed` | 4    | The sign-in doesn't include reading captured prompts and outputs                 |
| `access_denied`        | 4    | The account no longer has access to the workspace                                |
| `sign_in_failed`       | 4    | The sign-in was cancelled, refused, or used an address from another sign-in      |
| `no_pending_sign_in`   | 4    | `login --finish` found no sign-in waiting to finish                              |
| `trace_not_found`      | 1    | No trace has that id, or it is past the project's retention                      |
| `span_not_found`       | 1    | The trace has no span with that id, or it is past the project's retention        |
| `unreachable`          | 1    | telemetry.dev could not be reached                                               |
| `timeout`              | 1    | telemetry.dev took too long to answer                                            |
| `server_error`         | 1    | telemetry.dev failed to complete the request; try again                          |
| `tool_error`           | 1    | telemetry.dev returned another error; `message` says what it was                 |
| `result_too_large`     | 1    | The result was too large to read; narrow it with filters or `--limit`            |
| `server_unsupported`   | 1    | The deployment doesn't support the command, or an option it would have ignored   |
| `request_failed`       | 1    | The request failed for another reason                                            |
| `failed`               | 1    | The command failed for another reason                                            |
| `internal`             | 1    | An unexpected error, such as a configuration file the CLI couldn't read or write |

## Configuration

| Variable                        | Default                   | Purpose                                                                               |
| ------------------------------- | ------------------------- | ------------------------------------------------------------------------------------- |
| `TELEMETRY_DEV_PROJECT`         | none                      | The project to read when you don't pass `--project`                                   |
| `TELEMETRY_DEV_URL`             | `https://telemetry.dev`   | The telemetry.dev deployment to sign in to                                            |
| `TELEMETRY_DEV_CONFIG_DIR`      | `~/.config/telemetry-dev` | Where the sign-in is saved. The default follows `XDG_CONFIG_HOME` when set            |
| `NO_COLOR`                      | none                      | Turns off color. `--no-color` does the same                                           |
| `TELEMETRY_DEV_NO_UPDATE_CHECK` | none                      | Any value turns off the check for a newer version. `NO_UPDATE_NOTIFIER` does the same |

The sign-in is saved in `session.json`, which only your user can read. A sign-in to another deployment through `TELEMETRY_DEV_URL` gets its own file next to it. The file holds a refresh token, so keep it private. `telemetry logout` revokes the token at telemetry.dev and removes it from the file, and says so when the server doesn't confirm the revocation. `telemetry logout --json` prints `{"was_signed_in": true, "revoked": false}` in that case; `revoked` is `null` when there was no sign-in. To disconnect the CLI from your account entirely, open account settings at telemetry.dev.

## Updates

When you run the CLI on a terminal, it checks npm for a newer version at most once a day. If one is out, it prints a line on stderr after the command's output, with the command that updates the way you installed it:

```text
telemetry 0.2.0 is available (you have 0.1.6). Update: npm install --global @telemetry-dev/cli@latest
```

The check never delays a command. It runs while the command does, and when the command finishes first, the next run checks again. It reads the CLI's `latest` tag from the npm registry, or from `npm_config_registry` when that is set, so npm sees your IP address and the package name. telemetry.dev isn't involved. The answer is saved in `update-check.json` next to the sign-in.

There is no check with `--json`, when stdout or stderr isn't a terminal, in CI (`CI` is set), for a project that has the CLI as a dependency, or when `TELEMETRY_DEV_NO_UPDATE_CHECK` or `NO_UPDATE_NOTIFIER` is set. `telemetry status --json` reports `cli_version` and `latest_cli_version`, the latest version the last check found. `latest_cli_version` is `null` before the first check, when checks are off, and when the last check found a version older than this CLI, as after an update.

Separately, every request the CLI sends to telemetry.dev names the CLI and its version, as MCP clients do, so telemetry.dev can tell which versions are in use.
