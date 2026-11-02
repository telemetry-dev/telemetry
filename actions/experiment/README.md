# telemetry.dev experiment action

This composite action runs your eval script in a GitHub Actions workflow. The script calls
`runExperiment` (`@telemetry-dev/sdk`) or `run_experiment` (`telemetry-dev`). The action writes the
result to the job summary and to a comment on the pull request. The job fails when the check of the
pass marks of an experiment fails, when an experiment does not complete, or when the script exits
with an error. A script that catches the error of one experiment and runs the next still fails the
job, with an SDK release that writes the CI mode to the results file; with an older SDK release,
only the exit code of the script fails it, as in version 1.0.0. An experiment that runs with
`ci: false` (`ci=False` in Python) shows its result but does not fail the job. With `runsPerItem`
above 1, the result also says how many items the experiment has, and how many runs each.

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
      - uses: telemetry-dev/telemetry/actions/experiment@experiment-action-v1
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

## Versions

The tag `experiment-action-v1` moves to each 1.x release of the action. A change that can break a
workflow starts a new major version, `experiment-action-v2`, and `experiment-action-v1` stays at
the last 1.x release. For a run that does not change, use the tag of one release, for example
`experiment-action-v1.0.0`, or its commit SHA. The
[releases](https://github.com/telemetry-dev/telemetry/releases) of this repository list the
changes of each release.

## Inputs

| Input               | Description                                                                                                                                                                                                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `script`            | The eval script. A `.ts`, `.mts`, `.cts`, or `.tsx` script runs with the `tsx` of the project, from `node_modules/.bin` in the working directory or a directory above it, a `.js`, `.mjs`, or `.cjs` script with `node`, and a `.py` script with the `python` (or `python3`) of the PATH. Give `script` or `command`. |
| `command`           | A shell command that runs the eval script, for example `uv run python evals/reply_eval.py` or `pnpm eval`.                                                                                                                                                                                                            |
| `dataset`           | The dataset to run on. The action puts it in `TELEMETRY_DEV_EXPERIMENT_DATASET`. When the script also names a dataset, the two must be the same.                                                                                                                                                                      |
| `min-scores`        | Pass marks, as `name=value` pairs that commas or new lines separate. A pass rate is 0 to 1, and a mean is on the scale of its score. The action puts them in `TELEMETRY_DEV_EXPERIMENT_MIN_SCORES`, and they win over the `minScores` of the script for the same name.                                                |
| `api-key`           | A telemetry.dev API key with the `datasets:read` and `experiments:write` scopes, for all environments or for `experiment`. Default: the `TELEMETRY_DEV_API_KEY` variable of the job.                                                                                                                                  |
| `base-url`          | The telemetry.dev API. Default: `https://api.telemetry.dev`.                                                                                                                                                                                                                                                          |
| `working-directory` | The directory that the script or the command runs in. Default: `.`.                                                                                                                                                                                                                                                   |
| `comment`           | `true` or `false`. Set to `false` to write the job summary only. Another value gets a warning and counts as `true`. Default: `true`.                                                                                                                                                                                  |
| `github-token`      | The token that writes the comment. Default: `github.token`, which needs the `pull-requests: write` permission. When it is empty on a pull request, the action writes no comment and logs a warning.                                                                                                                   |

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

The action adds one comment to the pull request for each workflow, job, matrix leg,
`working-directory`, and `dataset`, `script`, or `command`. The comment of a matrix leg shows the
values of its matrix. A matrix of a job that calls a reusable workflow is not in the `matrix` of
the jobs of that workflow, thus those legs share one comment. A later push changes that comment,
and the comment names the commit of its result and links the workflow run. After a rename of the
workflow, the job, the dataset, or the script, the action adds a new comment, and the old one keeps
the result of its older commit. In version 1.0.0, the legs of a matrix and the working directories
shared one comment. Thus after an update from 1.0.0 they add new comments, and you can delete the
old one.

The action changes only a comment that its token wrote: with a copy of the comment by someone else,
it adds a new comment. The token of a GitHub App cannot read its own login, thus with an app token
the action changes a copy of the comment by any app. When the token cannot write comments, the
action logs a warning with the reason, and the job summary still has the result. A result too long
for a comment is cut, and the job summary has all of it.

The action looks for its comment on the 5 oldest and the 5 newest pages of the comments of the
pull request, 100 comments to a page. On a pull request with more than 1,000 comments, a comment
on a page between them is not found, and the action adds a new one.

A pull request from a fork gets no secrets, thus no API key, and the action stops with the error
"Set the api-key input or the TELEMETRY_DEV_API_KEY variable." The `if` of the example above skips
the job for those pull requests.

The comment and the summary show the name of each experiment, its status, the count of the runs
that completed and, when runs failed, of the runs that failed, and the value, the change from the
baseline, and the pass mark of each score. With `runsPerItem` above 1, each item runs more than one
time, and each run counts. The SDK writes the result for the action to a file in `RUNNER_TEMP`
with mode `0600`.
