#!/usr/bin/env bash
# Runs the experiment script of the experiment action. A failed script does not fail this step:
# the report step reads the exit code from the output exit-code, and the last step exits with it.
set -uo pipefail

script=${INPUT_SCRIPT:-}
command=${INPUT_COMMAND:-}

fail() {
  echo "::error title=telemetry.dev experiment::$1"
  echo "exit-code=2" >>"$GITHUB_OUTPUT"
  exit 0
}

if [ -n "$script" ] && [ -n "$command" ]; then fail "Give script or command, not the two."; fi
if [ -z "$script" ] && [ -z "$command" ]; then fail "Give the script or the command that runs the experiment."; fi

if [ -n "${INPUT_API_KEY:-}" ]; then
  echo "::add-mask::$INPUT_API_KEY"
  export TELEMETRY_DEV_API_KEY=$INPUT_API_KEY
fi
if [ -z "${TELEMETRY_DEV_API_KEY:-}" ]; then fail "Set the api-key input or the TELEMETRY_DEV_API_KEY variable."; fi
if [ -n "${INPUT_BASE_URL:-}" ]; then export TELEMETRY_DEV_BASE_URL=$INPUT_BASE_URL; fi
if [ -n "${INPUT_DATASET:-}" ]; then export TELEMETRY_DEV_EXPERIMENT_DATASET=$INPUT_DATASET; fi
if [ -n "${INPUT_MIN_SCORES:-}" ]; then export TELEMETRY_DEV_EXPERIMENT_MIN_SCORES=$INPUT_MIN_SCORES; fi
# The traces of the items stay out of production unless the job names an environment.
export TELEMETRY_DEV_ENVIRONMENT=${TELEMETRY_DEV_ENVIRONMENT:-experiment}

# mktemp makes the file with mode 0600, in a directory of this job.
results=$(mktemp "${RUNNER_TEMP:-/tmp}/telemetry-dev-experiment.XXXXXX") || fail "Could not make the results file."
echo "results=$results" >>"$GITHUB_OUTPUT"
export TELEMETRY_DEV_EXPERIMENT_RESULTS=$results

# The tsx of the project, from node_modules/.bin here or in a directory above. The action does not
# download one: a package from npm at run time would run with the API key in its environment.
project_tsx() {
  dir=$PWD
  while :; do
    if [ -x "$dir/node_modules/.bin/tsx" ]; then
      echo "$dir/node_modules/.bin/tsx"
      return 0
    fi
    [ "$dir" = / ] && return 1
    dir=$(dirname "$dir")
  done
}

code=0
if [ -n "$command" ]; then
  bash -c "$command" || code=$?
else
  case "$script" in
    *.ts | *.mts | *.cts | *.tsx)
      tsx=$(project_tsx) || fail "No tsx in node_modules. Add tsx to the devDependencies of the project and install them, or give a command."
      "$tsx" "$script" || code=$?
      ;;
    *.js | *.mjs | *.cjs) node "$script" || code=$? ;;
    *.py)
      python=$(command -v python || command -v python3) || fail "No python on the PATH. Use actions/setup-python, or give a command."
      "$python" "$script" || code=$?
      ;;
    *) fail "The script must end in .ts, .js, or .py. For another runner, give a command." ;;
  esac
fi

echo "exit-code=$code" >>"$GITHUB_OUTPUT"
