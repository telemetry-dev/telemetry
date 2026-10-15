# @telemetry-dev/cursor

Telemetry integration for the [Cursor](https://cursor.com) agent and
[telemetry.dev](https://telemetry.dev). It observes the agent loop through
[Cursor hooks](https://cursor.com/docs/hooks). It records turns, assistant
responses, thinking blocks, tool executions, subagents, and lifecycle events.
It does not change the behavior of Cursor.

## Install

```sh
npm install -g @telemetry-dev/cursor
telemetry-dev-cursor install --api-key td_live_…
```

The `install` command connects each observed hook event to this package in
`~/.cursor/hooks.json`. It also writes the key to `~/.cursor/telemetry-dev.json`.
The config file is necessary because Cursor started from the macOS dock does
not get shell env vars. The command keeps your other hooks. If you do the
command again, it replaces only the telemetry-dev entries and stops a running
telemetry daemon, so the new settings apply to the next event.
The `telemetry-dev-cursor uninstall` command removes the telemetry-dev entries
but keeps `~/.cursor/telemetry-dev.json`. Remove that file to delete the saved
API key.

Restart Cursor to load the hooks. The Hooks tab in **Customize** shows the
configured hooks and execution errors.

## Architecture

Cursor starts a new process for each hook event. Thus the integration has two
parts:

- `telemetry-dev-cursor hook` is the command in `hooks.json`. The command sends
  the event payload to the daemon through a local socket. The command always
  returns `{}` and exit code 0. A hook error does not stop the agent, but the
  first delivery can add up to 10 seconds.
- `telemetry-dev-cursor daemon` — the first hook starts it on demand. It
  holds open spans across hook processes and exports batched OTLP. It stops
  after 15 minutes without events.

## Configuration

The `install` command writes `~/.cursor/telemetry-dev.json`. Env vars have
priority over the file. The last three settings exist only as keys in the
file:

| Variable / key                              | Required | Default                        | Notes                                                                                        |
| ------------------------------------------- | -------- | ------------------------------ | -------------------------------------------------------------------------------------------- |
| `TELEMETRY_DEV_API_KEY` / `apiKey`          | yes      | —                              | Ingest key (`td_live_…`). Without a key, the integration is a full no-op.                    |
| `TELEMETRY_DEV_BASE_URL` / `baseUrl`        | no       | `https://ingest.telemetry.dev` | The integration removes trailing slashes.                                                    |
| `TELEMETRY_DEV_ENVIRONMENT` / `environment` | no       | `production`                   | Environment label on each trace and log.                                                     |
| `OTEL_SERVICE_NAME` / `serviceName`         | no       | `cursor`                       | OpenTelemetry service name.                                                                  |
| `captureInput`                              | no       | `true`                         | Export prompts, subagent tasks, and tool arguments. See [Content capture](#content-capture). |
| `captureOutput`                             | no       | `true`                         | Export assistant and thinking text and tool results.                                         |
| `enabled`                                   | no       | `true`                         | `false` stops all export. The hooks and the daemon still run.                                |

The three boolean keys must be JSON `true` or `false`. Any other value, such
as the string `"false"`, counts as `false`. If the file exists but is not a
readable JSON object, the daemon exports nothing, and `install` stops without
changing any file.

Set the capture keys at install time with `--capture-input false` and
`--capture-output false`. A later `install` without these flags keeps the
values of `captureInput`, `captureOutput`, and `enabled` that are already in
the file. The daemon reads the file only when it starts, so a manual edit
takes effect only after the daemon restarts. To apply an edit to these keys,
run `install` again: it keeps them and stops the daemon, and the next hook
event starts it with the new settings. `install` rewrites the other keys, so
pass `--api-key` and any `--base-url` or `--environment` again (or set their
env vars). It does not keep `serviceName`.

Overlapping `install` and `uninstall` runs wait for each other, so a
reinstall never writes back capture settings that another `install` has just
changed. A manual edit made while `install` runs can be overwritten.

## Trace shape

All user turns in one conversation share one trace. Each turn is a sibling
`invoke_agent` span. Subagent conversations are child spans in the parent
turn:

```text
invoke_agent cursor               (prompt input, last assistant text output)
├── thought                       (thinking block, host-reported duration)
├── execute_tool {toolName}       (tool call: arguments, result, error state)
├── invoke_agent {subagentType}   (Task-tool subagent: task, summary, status)
│   ├── thought
│   ├── execute_tool {toolName}   (tool calls of the subagent)
│   └── chat {model}
└── chat {model}                  (assistant message, output text)
```

- Cursor writes chat titles to `~/.cursor/chats/<hash>/<conversationId>/meta.json`.
  If a title is available when a turn ends, the turn span takes the title as
  its name. This also applies to subagent chats. Agent CLI sessions get no
  title from Cursor, so a turn without one takes the first line of the user
  prompt (or the subagent task) as its name, shortened to 60 characters. A
  turn with no title and no prompt keeps the `invoke_agent …` name, and so
  does every turn when `captureInput` is `false`.
- The events `sessionStart`, `sessionEnd`, `afterFileEdit`, `subagentStart`,
  and `preCompact` become logs with the hook event as `eventName`.
- Each span and log contains `gen_ai.conversation.id` (the `conversation_id`
  from Cursor), `cursor.generation_id`, `cursor.workspace`, and
  `cursor.user_email`. All turns of one conversation share one trace, each
  turn as a sibling in start order.
- Tool spans start at the `preToolUse` timestamp when it is available. If it
  is not available, they use the host-reported `duration`.
- The `stop` and `sessionEnd` events close the turn with status `completed`,
  `aborted`, or `error`, and then flush. If a `stop` event does not come, the
  next turn closes the open turn as `incomplete`.
- Cursor hook payloads have no field that links a subagent conversation to
  its parent, and the CLI does not emit `subagentStart`. A `preToolUse` for
  the `Task` tool therefore registers a pending call, and the next unknown
  conversation (one not opened by `beforeSubmitPrompt`) claims the oldest
  pending call as its parent. The parent's `postToolUse` for `Task` closes
  the subagent's span with the tool output as its result.
- On hosts that do emit `subagentStart`/`subagentStop`, those events also
  link and close subagent spans. A subagent that sends no hook events at all
  makes one `invoke_agent {subagentType}` span from `subagentStop`, as a
  child of the turn that started it.

## Content capture

Prompt input, assistant and thinking text, tool arguments, and tool results
obey the SDK settings `captureInput` and `captureOutput`. The default for the
two settings is `true`. When `captureInput` is `false`, turn spans keep the
`invoke_agent …` name instead of the chat title or a name from the prompt or
task, because Cursor generates chat titles from the prompt. A subagent that
`subagentStop` reports with status `error` then records `SubagentError`
without its task text.

The two settings do not cover these values, which are always exported:

- The user email, workspace and file paths, and conversation, generation,
  subagent, and tool call ids.
- Model, tool, and subagent names, and lifecycle metadata such as statuses,
  durations, compaction counts, and edit counts.
- Error messages from Cursor and exception stack traces. The `error_message`
  of `sessionEnd` goes to the `cursor.error.message` log attribute and to a
  `SessionError` exception. The `error_message` of a failed tool or `Task`
  call becomes the exception message of its span.

To export none of this, set `enabled` to `false`.

## Limitations

- Cursor hooks do not report token usage or cost. Thus `chat` spans contain
  only the model and the output text. The start time of a chat span is an
  approximation: the end of the last assistant message.
- The headless CLI (`cursor-agent`) fires a subset of the hooks. It does not
  fire `beforeSubmitPrompt`, `afterAgentResponse`, or `stop` (as of CLI
  2026.08). The first observed event opens the turn, and `sessionEnd` closes
  it. Thus tool spans, thought spans, and the turn status also come from CLI
  runs, but the prompt and the response text do not.
- Cloud agents load only the project-level `.cursor/hooks.json`. This
  installer writes user-level hooks. For cloud coverage, commit equivalent
  entries to the repository and supply the api key in the cloud environment.
- The integration does not register Tab (inline completion) hooks. It traces
  only agent sessions.
