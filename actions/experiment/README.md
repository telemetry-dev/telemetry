# telemetry.dev experiment action

This composite action runs your eval script in a GitHub Actions workflow. The script calls
`runExperiment` (`@telemetry-dev/sdk`) or `run_experiment` (`telemetry-dev`). The action writes the
result to the job summary and to a comment on the pull request. The job fails when the check of the
pass marks fails, when the experiment does not complete, or when the script exits with an error.

```yaml
on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  evals:
    # A pull request from a fork gets no secrets, so the job runs only for branches of this
    # repository.
    if: github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    env:
      TELEMETRY_DEV_API_KEY: ${{ secrets.TELEMETRY_DEV_API_KEY }}
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-node@v5
        with:
          node-version: 24
      # Installs tsx too, which runs the .ts script.
      - run: npm ci
      - uses: telemetry-dev/telemetry/actions/experiment@main
        with:
          dataset: support-golden
          script: evals/reply.eval.ts
          min-scores: correctness=0.9
```

`evals/reply.eval.ts` runs the experiment:

```ts
import { init, runExperiment } from "@telemetry-dev/sdk";

init();

await runExperiment({
  dataset: "support-golden",
  task: (item) => replyToCustomer(item.input),
  evaluators: ["correctness"],
});
```

Pin the action to a commit SHA of this repository for a run that does not change when `main`
changes.

## Inputs

| Input               | Description                                                                                                                                                                                                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `script`            | The eval script. A `.ts`, `.mts`, `.cts`, or `.tsx` script runs with the `tsx` of the project, from `node_modules/.bin` in the working directory or a directory above it, a `.js`, `.mjs`, or `.cjs` script with `node`, and a `.py` script with the `python` (or `python3`) of the PATH. Give `script` or `command`. |
| `command`           | A shell command that runs the eval script, for example `uv run python evals/reply_eval.py` or `pnpm eval`.                                                                                                                                                                                                            |
| `dataset`           | The dataset to run on. The action puts it in `TELEMETRY_DEV_EXPERIMENT_DATASET`. When the script also names a dataset, the two must be the same.                                                                                                                                                                      |
| `min-scores`        | Pass marks, as `name=value` pairs that commas or new lines separate. A pass rate is 0 to 1, and a mean is on the scale of its score. The action puts them in `TELEMETRY_DEV_EXPERIMENT_MIN_SCORES`, and they win over the `minScores` of the script for the same name.                                                |
| `api-key`           | A telemetry.dev API key with the `datasets:read` and `experiments:write` scopes, for all environments or for `experiment`. Default: the `TELEMETRY_DEV_API_KEY` variable of the job.                                                                                                                                  |
| `base-url`          | The telemetry.dev API. Default: `https://ingest.telemetry.dev`.                                                                                                                                                                                                                                                       |
| `working-directory` | The directory that the script or the command runs in. Default: `.`.                                                                                                                                                                                                                                                   |
| `comment`           | Set to `false` to write the job summary only.                                                                                                                                                                                                                                                                         |
| `github-token`      | The token that writes the comment. Default: `github.token`, which needs the `pull-requests: write` permission.                                                                                                                                                                                                        |

The action sets `TELEMETRY_DEV_ENVIRONMENT` to `experiment` when the job does not set it, so the
traces of the items stay out of your production pages. Thus the key must be for all environments
or for `experiment`. A key for another environment, for example your production key, cannot send
the traces of the items: the experiment then completes with its outputs, but with no traces and no
cost. `CI` is `true` in GitHub Actions, thus the SDK runs in CI mode: it records the repository,
the commit, the run, and the pull request with the experiment, and it exits with an error when the
check fails.

The action does not download packages. Add `tsx` to the `devDependencies` of the project for a
`.ts` script, or give a `command`.

The outputs, the errors, and the scorer reasons of the items go to telemetry.dev and stay with the
experiment. The capture options of the SDK and the capture setting of the environment in the
project do not apply to them.

## Outputs

| Output         | Description                                                           |
| -------------- | --------------------------------------------------------------------- |
| `check-status` | `passed`, `failed`, or `none`, for the last experiment of the script. |
| `url`          | The page of the last experiment in telemetry.dev.                     |

## The comment

The action adds one comment to the pull request for each workflow, job, and `dataset`, `script`, or
`command`. A later push changes that comment. The action changes only a comment that its token
wrote: with a copy of the comment by someone else, it adds a new comment. When the token cannot
write comments, the action logs a warning, and the job summary still has the result.

A pull request from a fork gets no secrets, thus no API key, and the action stops with the error
"Set the api-key input or the TELEMETRY_DEV_API_KEY variable." The `if` of the example above skips
the job for those pull requests.

The comment and the summary show the name of each experiment, its status, the count of the items
that ran, and the value, the change from the baseline, and the pass mark of each score. The SDK
writes the result for the action to a file in `RUNNER_TEMP` with mode `0600`.
