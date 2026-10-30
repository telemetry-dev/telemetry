# @telemetry-dev/cli

The telemetry.dev command line. Search traces, sessions, logs, and AI usage from a terminal, a script, or an agent's shell, without connecting an MCP client.

The CLI signs in to the same read-only API that the telemetry.dev MCP server gives agents, with the same permissions. The `--json` output of each command is the result of the matching MCP tool.

## Install

```sh
npm install --global @telemetry-dev/cli
```

Or run it without installing:

```sh
npx @telemetry-dev/cli login
```

Node.js 20 or later is required.

## Sign in

```sh
telemetry login
```

`login` opens your browser. Choose a workspace and select **Allow**. To let the CLI read captured prompts, outputs, and tool arguments, also check **Also read captured prompts, outputs, and tool arguments**. Without that permission, every command except `payload` works.

Over SSH, in a container, or anywhere without a browser, run `telemetry login --no-browser`. Open the printed link on any device. After you select **Allow**, the browser tries to load an address that starts with `http://127.0.0.1`. The page doesn't load. Copy the whole address and paste it into the terminal.

In a shell with no terminal to paste into, such as an agent's, `telemetry login --no-browser` prints the link and exits. After you select **Allow**, finish the sign-in with the address the browser ended on:

```sh
telemetry login --finish 'http://127.0.0.1:53124/callback?code=…'
```

A sign-in covers one workspace. To read another workspace, run `telemetry login` again and choose it. The new sign-in replaces the old one, and the CLI asks telemetry.dev to revoke the old one.

## Use

```sh
telemetry projects                       # list projects
telemetry use support-bot                # set the default project
telemetry traces --errors                # recent failed traces
telemetry trace 4bf92f3577b34da6a3ce929d0e0e4736
telemetry sessions --range 7d --errors   # conversations with a failed trace
telemetry logs --level error
telemetry usage --range 30d --by user
telemetry payload <trace-id> <span-id> --field output
```

Run `telemetry <command> --help` to see each command's options and examples.

A command reads its project from `--project <name|id>`. Without that option, it uses `TELEMETRY_DEV_PROJECT`, then the default that you set with `telemetry use`. If the workspace has only one project, the command uses it.

List commands return 20 results by default. Use `--limit` to return from 1 to 50 results. When more results are available, the command prints a `--cursor` command for the next page.

## Scripts and agents

Add `--json` to any command except `login` and `logout` for machine-readable output on stdout. Messages and progress go to stderr, so `telemetry traces --json > traces.json` writes only data. List commands print the MCP tool's result as returned, with the fields that the [MCP server documentation](https://telemetry.dev/docs/platform/mcp-server) describes.

`telemetry trace` and `telemetry payload` read every page of a trace or a captured text, up to a limit: 2,000 spans for a trace, and 1,000,000 characters for each payload field. Their JSON output has `complete: false` when the limit cut it short, and human output says so on stderr.

Captured text comes from your users, models, and tools. telemetry.dev removes ANSI color and cursor sequences, bidirectional overrides, and Unicode tag characters from results before they reach the CLI; it does not clean the text of error messages. Human-readable output, including `payload --field`, removes the remaining escape sequences and control characters. JSON output escapes them instead of removing them, so use `--json` when you need everything the server sent.

| Exit code | Meaning                                                         |
| --------- | --------------------------------------------------------------- |
| 0         | Success                                                         |
| 1         | The request failed                                              |
| 2         | The command or its options were not valid                       |
| 4         | You are not signed in, the sign-in expired, or access is denied |
| 130       | Interrupted with Ctrl-C                                         |

With `--json`, an error is one line of JSON on stderr, and stdout stays empty. Ctrl-C exits with 130 and no JSON, and `status --json` reports a missing sign-in as data on stdout with exit code 4.

```text
{"error":{"code":"not_signed_in","message":"You're not signed in. Run `telemetry login` first.","hint":"telemetry login","retryable":false,"exit_code":4,"cli_version":"0.1.0"}}
```

Branch on `code`, not on `message`, whose wording can change. `hint` is a command to run next, when there is one; `telemetry login` needs a person at a browser, so an agent should ask for it rather than run it. `retryable` is true when running the same command again may succeed. New codes may be added, and existing codes keep their meaning.

| Code                   | Exit | Meaning                                                                          |
| ---------------------- | ---- | -------------------------------------------------------------------------------- |
| `invalid_arguments`    | 2    | An option or argument is not valid                                               |
| `unknown_command`      | 2    | There is no command with that name                                               |
| `project_not_found`    | 2    | No project in the workspace has that name or id                                  |
| `project_ambiguous`    | 2    | Several projects have that name; pass its id                                     |
| `project_required`     | 2    | The workspace has several projects and none was chosen                           |
| `no_projects`          | 1    | The workspace has no projects yet                                                |
| `not_signed_in`        | 4    | No sign-in is saved for this deployment                                          |
| `sign_in_expired`      | 4    | The sign-in expired or was disconnected                                          |
| `payloads_not_allowed` | 4    | The sign-in doesn't include reading captured prompts and outputs                 |
| `unreachable`          | 1    | telemetry.dev could not be reached                                               |
| `timeout`              | 1    | telemetry.dev took too long to answer                                            |
| `tool_error`           | 1    | telemetry.dev returned an error; `message` says what it was                      |
| `result_too_large`     | 1    | The result was too large to read; narrow it with filters or `--limit`            |
| `request_failed`       | 1    | The request failed for another reason                                            |
| `failed`               | 1    | The command failed for another reason                                            |
| `internal`             | 1    | An unexpected error, such as a configuration file the CLI couldn't read or write |

## Configuration

| Variable                   | Default                   | Purpose                                                                    |
| -------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| `TELEMETRY_DEV_PROJECT`    | none                      | The project to read when you don't pass `--project`                        |
| `TELEMETRY_DEV_URL`        | `https://telemetry.dev`   | The telemetry.dev deployment to sign in to                                 |
| `TELEMETRY_DEV_CONFIG_DIR` | `~/.config/telemetry-dev` | Where the sign-in is saved. The default follows `XDG_CONFIG_HOME` when set |
| `NO_COLOR`                 | none                      | Turns off color. `--no-color` does the same                                |

The sign-in is saved in `session.json`, which only your user can read. A sign-in to another deployment through `TELEMETRY_DEV_URL` gets its own file next to it. The file holds a refresh token, so keep it private. `telemetry logout` revokes the token at telemetry.dev and removes it from the file, and says so when the server doesn't confirm the revocation. To disconnect the CLI from your account entirely, open account settings at telemetry.dev.
