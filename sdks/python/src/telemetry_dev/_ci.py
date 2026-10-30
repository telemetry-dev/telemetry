"""The GitHub Actions run of an experiment, and the variables of the experiment action."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, cast

from ._config import report_error
from ._serialize import utf16_length

_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_COMMIT = re.compile(r"[0-9a-fA-F]{7,64}")
_HTTPS_URL = re.compile(r"https://\S+")
# A plain decimal, so that the TypeScript and Python SDKs read the same marks.
_DECIMAL = re.compile(r"-?\d+(\.\d+)?", re.ASCII)
_RUN_ID = re.compile(r"\d+", re.ASCII)
_MAX_REPOSITORY_LENGTH = 200
_MAX_URL_LENGTH = 2048
_MAX_PR_TITLE_LENGTH = 256


def in_ci(env: Mapping[str, str]) -> bool:
    """CI mode is on when ``CI`` is ``true`` or ``1``, as CI services set it."""
    return env.get("CI") in ("true", "1")


def github_run(env: Mapping[str, str]) -> dict[str, Any] | None:
    """The run that GitHub Actions describes in its variables, or None outside GitHub Actions.

    On a pull request the commit is the head of the pull request: ``GITHUB_SHA`` is then a merge
    commit that nobody pushed.
    """
    repository = env.get("GITHUB_REPOSITORY") or ""
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or len(repository) > _MAX_REPOSITORY_LENGTH
        or not _REPOSITORY.fullmatch(repository)
    ):
        return None
    pull = _pull_request(env.get("GITHUB_EVENT_PATH"))
    commit = pull.get("commit") or env.get("GITHUB_SHA")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        return None
    server = (env.get("GITHUB_SERVER_URL") or "https://github.com").rstrip("/")
    run_id = env.get("GITHUB_RUN_ID") or ""
    run: dict[str, Any] = {
        "provider": "github_actions",
        "repository": repository,
        "commit": commit,
        "prNumber": pull.get("number"),
        "prTitle": pull.get("title"),
        "prUrl": pull.get("url"),
        "runUrl": _https_url(f"{server}/{repository}/actions/runs/{run_id}")
        if _RUN_ID.fullmatch(run_id)
        else None,
    }
    return {key: value for key, value in run.items() if value is not None}


def min_scores_from_env(env: Mapping[str, str]) -> dict[str, float]:
    """The pass marks in ``TELEMETRY_DEV_EXPERIMENT_MIN_SCORES``, which the experiment action
    sets from its ``min-scores`` input: ``name=value`` pairs, separated by commas or new lines."""
    scores: dict[str, float] = {}
    for pair in re.split(r"[,\n]", env.get("TELEMETRY_DEV_EXPERIMENT_MIN_SCORES") or ""):
        if not pair.strip():
            continue
        parts = [part.strip() for part in pair.split("=")]
        if len(parts) != 2 or not parts[0] or not _DECIMAL.fullmatch(parts[1]):
            raise TypeError(
                f'TELEMETRY_DEV_EXPERIMENT_MIN_SCORES: "{pair.strip()}" is not a name=number pair'
            )
        scores[parts[0]] = float(parts[1])
    return scores


def write_result(
    env: Mapping[str, str],
    result: object,
    on_error: Callable[[BaseException], None] | None,
) -> None:
    """Add the result to the file in ``TELEMETRY_DEV_EXPERIMENT_RESULTS`` as one line of JSON.
    The experiment action reads the file to write the job summary and the pull request comment."""
    path = env.get("TELEMETRY_DEV_EXPERIMENT_RESULTS")
    if not path:
        return
    line = json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as file:
            file.write(line)
    except OSError as exc:
        report_error(on_error, "writing TELEMETRY_DEV_EXPERIMENT_RESULTS failed", exc)


def _pull_request(event_path: str | None) -> dict[str, Any]:
    if not event_path:
        return {}
    try:
        with open(event_path, encoding="utf-8") as file:
            event = json.load(file)
    except (OSError, ValueError):
        return {}
    pull = cast("dict[str, Any]", event).get("pull_request") if isinstance(event, dict) else None
    if not isinstance(pull, dict):
        return {}
    fields = cast("dict[str, Any]", pull)
    number = fields.get("number")
    title = fields.get("title")
    url = fields.get("html_url")
    head = fields.get("head")
    sha = cast("dict[str, Any]", head).get("sha") if isinstance(head, dict) else None
    return {
        "number": number
        if isinstance(number, int) and not isinstance(number, bool) and number > 0
        else None,
        "title": _cut(title, _MAX_PR_TITLE_LENGTH) if isinstance(title, str) else None,
        "url": _https_url(url) if isinstance(url, str) else None,
        "commit": sha if isinstance(sha, str) else None,
    }


def _https_url(url: str) -> str | None:
    return url if utf16_length(url) <= _MAX_URL_LENGTH and _HTTPS_URL.fullmatch(url) else None


def _cut(text: str, max_units: int) -> str:
    """``text`` cut to ``max_units`` UTF-16 units, the length that the API counts, with no half of
    a surrogate pair at the end: the API stores the run as jsonb, and Postgres rejects a lone
    surrogate there."""
    encoded = text.encode("utf-16-le", "surrogatepass")
    if len(encoded) <= max_units * 2:
        return text
    return encoded[: max_units * 2].decode("utf-16-le", "ignore")
