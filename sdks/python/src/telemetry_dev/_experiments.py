"""Experiments of your code: run a task on each item of a dataset and record the result."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import math
import os
import threading
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, TypeAlias, TypeGuard, cast

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context

from ._api import Api, request, resolve_api
from ._ci import github_run, in_ci, min_scores_from_env, write_result
from ._client import flush, get_client
from ._config import logger, report_error
from ._context import context_with_propagated_attributes, propagated_attributes
from ._scores import (
    MAX_REASON_LENGTH,
    MAX_SCORE_NAME_LENGTH,
    USER_FEEDBACK,
    check_value,
    is_finite,
)
from ._semconv import ATTR_DATASET_ITEM_ID, ATTR_EXPERIMENT_ID, ATTR_SESSION_ID
from ._serialize import json_dumps, truncate, utf16_length
from ._spans import SpanHandle, start_span

# The experiment environment, which pages, metrics, and alerts leave out by default.
_EXPERIMENT_ENVIRONMENT = "experiment"
_MAX_RUNS_PER_ITEM = 5
_MAX_CONCURRENCY = 100
# Server evaluators of one experiment, and scorers: an item takes at most 20 scores.
_MAX_EVALUATORS = 20
_MAX_MIN_SCORES = 20
_MAX_ERROR_LENGTH = 4000
_MAX_OUTPUT_LENGTH = 100_000
_MAX_DATASET_ITEMS = 10_000
# Items in one POST /v1/experiments/{id}/items.
_MAX_BATCH_ITEMS = 100
# Bytes of JSON in one post of items, well under the 16 MiB body limit of the API.
_MAX_BATCH_BYTES = 4 * 1024 * 1024
_REQUEST_TIMEOUT_S = 30.0
# How long run_experiment waits for the server evaluators after the last item.
_MAX_WAIT_S = 30 * 60.0
_MAX_POLL_INTERVAL_S = 5.0
_FINISHED = frozenset({"completed", "failed", "stopped"})
_RATE_KINDS = frozenset({"pass_fail", "up_down"})


@dataclass(frozen=True)
class DatasetItem:
    """An item of a dataset version, as ``GET /v1/datasets/{name}`` returns it. ``id`` is the key
    of the item: it stays the same across dataset versions."""

    id: str
    number: int
    input: Mapping[str, Any]
    expected_output: str | None
    metadata: Mapping[str, Any]
    tags: tuple[str, ...]
    split: str | None


@dataclass(frozen=True)
class ExperimentItem(DatasetItem):
    """One run of a dataset item, ``run_index`` 0 to ``runs_per_item`` - 1."""

    run_index: int


ScorerResult: TypeAlias = "bool | int | float | str | Mapping[str, Any] | None"
"""What a scorer returns. True and False are a pass and a fail, a number is the value, and a
string is the label. A mapping can have ``value``, ``label``, and ``reason``. None gives no
score."""

Scorer: TypeAlias = "Callable[[Any, ExperimentItem], ScorerResult]"
"""Scores the output of an item on your machine. The score has the name of the scorer."""

AsyncScorer: TypeAlias = "Callable[[Any, ExperimentItem], ScorerResult | Awaitable[ScorerResult]]"
"""A scorer of ``run_experiment_async``, which can be async."""


@dataclass(frozen=True)
class ExperimentScore:
    """One score name of an experiment. ``value`` is the pass rate (0 to 1) of a pass/fail or
    up/down score, or the mean of a number score. ``kind`` is None before the score has values.
    ``baseline_value`` is the value in the experiment that this one compares with, by default the
    dataset baseline. ``delta`` is whole points of a pass rate, or the change of a mean to one
    decimal. ``passed`` is None when the score has no pass mark. ``items_failed`` is the count of
    runs that failed: they have no score, so ``value`` leaves them out, and the check fails. It is
    None from a server that does not send it."""

    name: str
    kind: str | None
    value: float | None
    pass_count: int
    fail_count: int
    baseline_value: float | None
    delta: float | None
    min_score: float | None
    passed: bool | None
    items_failed: int | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """The result of an experiment. ``status`` is ``completed``, ``failed``, or ``stopped``.
    ``check_status`` is None when the experiment has no pass marks. ``error`` tells why the
    experiment failed."""

    id: str
    name: str
    status: str
    check_status: Literal["passed", "failed"] | None
    items_total: int
    items_done: int
    items_failed: int
    error: str | None
    scores: tuple[ExperimentScore, ...]
    url: str


class ExperimentError(Exception):
    """Why run_experiment failed. ``code`` is the code of the API error (``dataset_not_found``,
    ``insufficient_scope``, ``invalid_api_key``, ...), or ``missing_api_key``, ``timeout``,
    ``network_error``, ``invalid_response``, or ``dataset_empty``. In CI mode it is also
    ``check_failed`` or ``experiment_failed``, and ``result`` holds the result. A response with a
    different status and no API error gives ``http_<status>``. A scorer that raises or returns a
    value that is not a score sends a ``scorer_failed`` error to ``on_error``."""

    def __init__(
        self,
        code: str,
        message: str,
        status: int | None = None,
        *,
        result: ExperimentResult | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.result = result


def run_experiment(
    *,
    task: Callable[[ExperimentItem], Any],
    dataset: str | None = None,
    name: str | None = None,
    dataset_version: int | None = None,
    split: str | None = None,
    evaluators: Sequence[str] | None = None,
    scorers: Mapping[str, Scorer] | None = None,
    min_scores: Mapping[str, float] | None = None,
    runs_per_item: int = 1,
    concurrency: int = 4,
    metadata: Mapping[str, Any] | None = None,
    ci: bool | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ExperimentResult:
    """Run ``task`` on each item of a dataset and record the result as an experiment.

    Each item run is its own trace, with ``td.experiment.id`` and ``td.dataset.item_id`` on each
    span. The outputs go to telemetry.dev in batches of at most 100, with the scores of
    ``scorers``. Then run_experiment waits for the ``evaluators`` of the server, prints the result,
    and returns it. ``concurrency`` (1 to 100, default 4) threads run the items, each in a copy of
    the context of the caller.

    ``dataset`` defaults to the ``dataset`` input of the experiment action, which it sets in
    ``TELEMETRY_DEV_EXPERIMENT_DATASET``. When the two are set, they must be the same.
    ``min_scores`` holds pass marks by score name: the lowest pass rate (0 to 1) of a pass/fail
    score, or the lowest mean of a number score. The ``min-scores`` input of the action adds to
    them and wins for the same name. ``runs_per_item`` is 1 to 5.

    The run cannot start without its dataset, so run_experiment raises ``ExperimentError`` when a
    request fails. When the run stops after it made the experiment, it first sends the complete
    request, so that telemetry.dev ends the experiment with the item runs that it has. A task that
    raises fails only its item. In CI mode (``ci``, default on when ``CI`` is ``true`` or ``1``),
    it records the GitHub Actions run, and it raises when the check fails or the experiment does
    not complete, so the process exits with an error.
    """
    run, api = _prepare(
        task,
        dataset,
        name,
        dataset_version,
        split,
        evaluators,
        scorers,
        min_scores,
        runs_per_item,
        concurrency,
        metadata,
        ci,
        api_key,
        base_url,
        asynchronous=False,
    )
    runs, created = _start(api, run)
    try:
        _announce(run, created)
        _run_items(api, run, created, runs)
    except BaseException:
        _complete_after_failure(api, created)
        raise
    return _finish(api, run, created)


async def run_experiment_async(
    *,
    task: Callable[[ExperimentItem], Any],
    dataset: str | None = None,
    name: str | None = None,
    dataset_version: int | None = None,
    split: str | None = None,
    evaluators: Sequence[str] | None = None,
    scorers: Mapping[str, AsyncScorer] | None = None,
    min_scores: Mapping[str, float] | None = None,
    runs_per_item: int = 1,
    concurrency: int = 4,
    metadata: Mapping[str, Any] | None = None,
    ci: bool | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ExperimentResult:
    """``run_experiment`` for async tasks and scorers. ``concurrency`` asyncio tasks run the
    items, and the requests go in worker threads. A task that is not async runs on the event
    loop, so it blocks the other items while it runs: use ``run_experiment`` for it."""
    run, api = _prepare(
        task,
        dataset,
        name,
        dataset_version,
        split,
        evaluators,
        scorers,
        min_scores,
        runs_per_item,
        concurrency,
        metadata,
        ci,
        api_key,
        base_url,
        asynchronous=True,
    )
    runs, created = await asyncio.to_thread(_start, api, run)
    try:
        _announce(run, created)
        await _run_items_async(api, run, created, runs)
    except BaseException:
        await asyncio.to_thread(_complete_after_failure, api, created)
        raise
    return await asyncio.to_thread(_finish, api, run, created)


@dataclass(frozen=True)
class _Run:
    dataset: str
    task: Callable[[ExperimentItem], Any]
    name: str | None
    dataset_version: int | None
    split: str | None
    evaluators: list[str] | None
    scorers: list[tuple[str, Callable[[Any, ExperimentItem], Any]]]
    min_scores: dict[str, float] | None
    runs_per_item: int
    concurrency: int
    metadata: dict[str, Any] | None
    ci: dict[str, Any] | None
    in_ci: bool


@dataclass(frozen=True)
class _Created:
    id: str
    name: str
    items_total: int
    url: str


@dataclass(frozen=True)
class _Page:
    version: int
    items: list[DatasetItem]
    next_cursor: str | None


def _prepare(
    task: object,
    dataset: object,
    name: object,
    dataset_version: object,
    split: object,
    evaluators: object,
    scorers: object,
    min_scores: object,
    runs_per_item: object,
    concurrency: object,
    metadata: object,
    ci: object,
    api_key: str | None,
    base_url: str | None,
    *,
    asynchronous: bool,
) -> tuple[_Run, Api]:
    env = os.environ
    if not callable(task):
        raise TypeError("task must be a function")
    if not asynchronous and inspect.iscoroutinefunction(task):
        raise TypeError("task is async: use run_experiment_async")
    for key, value in (("dataset", dataset), ("name", name), ("split", split)):
        if value is not None and (not isinstance(value, str) or not value):
            raise TypeError(f"{key} must be a non-empty string")
    action_dataset = env.get("TELEMETRY_DEV_EXPERIMENT_DATASET") or None
    chosen = cast("str | None", dataset) if dataset is not None else action_dataset
    if chosen is None:
        raise TypeError("Pass dataset, or set the dataset input of the experiment action")
    if action_dataset is not None and action_dataset != chosen:
        raise TypeError(
            f"dataset is {chosen}, but the experiment action names {action_dataset}. "
            "Name one dataset."
        )
    if dataset_version is not None and not (_is_int(dataset_version) and dataset_version >= 1):
        raise ValueError("dataset_version must be a positive integer")

    evaluator_names: list[str] | None = None
    if evaluators is not None:
        names = (
            list(cast("Sequence[object]", evaluators))
            if isinstance(evaluators, Sequence) and not isinstance(evaluators, str)
            else None
        )
        if (
            names is None
            or len(names) > _MAX_EVALUATORS
            or not all(isinstance(n, str) and _valid_score_name(n) for n in names)
        ):
            raise TypeError(
                f"evaluators must be a list of at most {_MAX_EVALUATORS} evaluator names"
            )
        evaluator_names = cast("list[str]", names)

    if scorers is not None and not isinstance(scorers, Mapping):
        raise TypeError("scorers must be a mapping of score names to functions")
    scorer_list = list(cast("Mapping[object, object]", scorers or {}).items())
    if len(scorer_list) > _MAX_EVALUATORS:
        raise TypeError(f"scorers can have at most {_MAX_EVALUATORS} scorers")
    folded = {n.lower() for n in evaluator_names or []}
    for scorer_name, scorer in scorer_list:
        if (
            not isinstance(scorer_name, str)
            or not _valid_score_name(scorer_name)
            or not callable(scorer)
        ):
            raise TypeError(
                f"scorers must map score names of 1 to {MAX_SCORE_NAME_LENGTH} characters "
                "to functions"
            )
        if not asynchronous and inspect.iscoroutinefunction(scorer):
            raise TypeError(f"The scorer {scorer_name} is async: use run_experiment_async")
        if scorer_name == USER_FEEDBACK:
            raise TypeError(
                f"{USER_FEEDBACK} is the score of user feedback. Name the scorer another way."
            )
        if scorer_name.lower() in folded:
            raise TypeError(
                f"The scorer {scorer_name} has the name of an evaluator. Rename one of them."
            )

    if min_scores is not None and not isinstance(min_scores, Mapping):
        raise TypeError(f"min_scores must map at most {_MAX_MIN_SCORES} score names to numbers")
    marks: dict[object, object] = {
        **cast("Mapping[object, object]", min_scores or {}),
        **min_scores_from_env(env),
    }
    if len(marks) > _MAX_MIN_SCORES or not all(
        isinstance(n, str) and _valid_score_name(n) and is_finite(v) for n, v in marks.items()
    ):
        raise TypeError(f"min_scores must map at most {_MAX_MIN_SCORES} score names to numbers")

    if not _is_int(runs_per_item) or not 1 <= runs_per_item <= _MAX_RUNS_PER_ITEM:
        raise ValueError(f"runs_per_item must be an integer from 1 to {_MAX_RUNS_PER_ITEM}")
    if not _is_int(concurrency) or not 1 <= concurrency <= _MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be an integer from 1 to {_MAX_CONCURRENCY}")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if ci is not None and not isinstance(ci, bool):
        raise TypeError("ci must be a bool")
    ci_mode = in_ci(env) if ci is None else ci

    api = resolve_api(api_key, base_url)
    if not api.api_key:
        raise ExperimentError(
            "missing_api_key", "No API key: pass api_key or set TELEMETRY_DEV_API_KEY"
        )
    run = _Run(
        dataset=chosen,
        task=task,
        name=cast("str | None", name),
        dataset_version=dataset_version,
        split=cast("str | None", split),
        evaluators=evaluator_names,
        scorers=cast("list[tuple[str, Callable[[Any, ExperimentItem], Any]]]", scorer_list),
        min_scores={str(n): float(cast("float", v)) for n, v in marks.items()} or None,
        runs_per_item=runs_per_item,
        concurrency=concurrency,
        metadata=dict(cast("Mapping[str, Any]", metadata)) if metadata is not None else None,
        ci=github_run(env) if ci_mode else None,
        in_ci=ci_mode,
    )
    return run, api


def _valid_score_name(name: str) -> bool:
    """A trimmed name of 1 to 64 characters, as the API stores score names."""
    return 0 < utf16_length(name) <= MAX_SCORE_NAME_LENGTH and name.strip() == name


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _warn_about_client() -> None:
    client = get_client()
    environment = client.config.environment if client is not None and client.enabled else None
    if environment is None:
        logger.warning(
            "run_experiment: no telemetry client, so the items record no traces. "
            'Call init(environment="experiment") first.'
        )
    elif environment.strip().lower() != _EXPERIMENT_ENVIRONMENT:
        logger.warning(
            "run_experiment: the traces of the items go to the environment %s. "
            'Call init(environment="experiment") to keep them out of it.',
            environment,
        )


def _start(api: Api, run: _Run) -> tuple[_Runs, _Created]:
    _warn_about_client()
    first = _fetch_page(api, run, run.dataset_version, None)
    if not first.items:
        scope = "" if run.split is None else f" in the split {run.split}"
        raise ExperimentError("dataset_empty", f"The dataset {run.dataset} has no items{scope}")
    body = {
        "name": run.name,
        "dataset": run.dataset,
        "datasetVersion": first.version,
        "split": run.split,
        "evaluators": run.evaluators,
        "minScores": run.min_scores,
        "runsPerItem": run.runs_per_item,
        "ci": run.ci,
        "metadata": run.metadata,
    }
    created = _parse_created(
        request(
            api,
            ExperimentError,
            "POST",
            "/v1/experiments",
            body={key: value for key, value in body.items() if value is not None},
            timeout_s=_REQUEST_TIMEOUT_S,
            # A repeat after a lost response would start a second experiment.
            idempotent=False,
        )
    )
    return _Runs(api, run, first), created


def _announce(run: _Run, created: _Created) -> None:
    each = f", {run.runs_per_item} runs each" if run.runs_per_item > 1 else ""
    print(
        f"Experiment {created.name}: {created.items_total // run.runs_per_item} items{each}. "
        f"{created.url}",
        flush=True,
    )


def _complete_after_failure(api: Api, created: _Created) -> None:
    """Without the complete, the experiment would stay running: the server ends it with the item
    runs that it has, or fails it when it has none."""
    try:
        _complete(api, created)
    except Exception as exc:
        report_error(api.on_error, "completing the experiment after a failed run failed", exc)


class _Runs:
    """The runs of each item of the dataset version, page by page, so that only one page is in
    memory. A lock lets the workers share the generator."""

    def __init__(self, api: Api, run: _Run, first: _Page) -> None:
        self._lock = threading.Lock()
        self._runs = _item_runs(api, run, first)

    def next(self) -> ExperimentItem | None:
        with self._lock:
            return next(self._runs, None)

    def close(self) -> None:
        with self._lock:
            self._runs.close()


def _item_runs(api: Api, run: _Run, first: _Page) -> Generator[ExperimentItem]:
    page = first
    count = 0
    while True:
        count += len(page.items)
        if count > _MAX_DATASET_ITEMS:
            raise ExperimentError(
                "invalid_response", f"telemetry.dev returned more than {_MAX_DATASET_ITEMS} items"
            )
        for item in page.items:
            # A copy for each run, so that a task that changes its input does not change the next
            # run.
            for run_index in range(run.runs_per_item):
                yield ExperimentItem(**copy.deepcopy(vars(item)), run_index=run_index)
        if page.next_cursor is None:
            return
        page = _fetch_page(api, run, page.version, page.next_cursor)


def _fetch_page(api: Api, run: _Run, version: int | None, cursor: str | None) -> _Page:
    """One page of a dataset version. A page after the first names the version of the first."""
    page = _parse_page(
        request(
            api,
            ExperimentError,
            "GET",
            f"/v1/datasets/{urllib.parse.quote(run.dataset, safe='')}",
            query={
                "version": None if version is None else str(version),
                "split": run.split,
                "cursor": cursor,
            },
            timeout_s=_REQUEST_TIMEOUT_S,
            idempotent=True,
        )
    )
    if (
        page is None
        or (version is not None and page.version != version)
        or (page.next_cursor is not None and (not page.items or page.next_cursor == cursor))
    ):
        raise ExperimentError("invalid_response", "telemetry.dev returned a dataset it cannot read")
    return page


class _Batch:
    """Collects item results and posts them in order, one post at a time."""

    def __init__(self, post: Callable[[list[dict[str, Any]]], object]) -> None:
        self._post = post
        self._pending: list[dict[str, Any]] = []
        self._bytes = 0
        self._lock = threading.Lock()

    def add(self, result: dict[str, Any]) -> None:
        size = len(json_dumps(result, default=None).encode())
        with self._lock:
            if self._pending and self._bytes + size > _MAX_BATCH_BYTES:
                self._send()
            self._pending.append(result)
            self._bytes += size
            if len(self._pending) >= _MAX_BATCH_ITEMS:
                self._send()

    def flush(self) -> None:
        with self._lock:
            if self._pending:
                self._send()

    def _send(self) -> None:
        results = self._pending
        self._pending = []
        self._bytes = 0
        self._post(results)


def _batch(api: Api, created: _Created) -> _Batch:
    path = f"/v1/experiments/{urllib.parse.quote(created.id, safe='')}/items"
    return _Batch(
        lambda results: request(
            api,
            ExperimentError,
            "POST",
            path,
            body={"items": results},
            timeout_s=_REQUEST_TIMEOUT_S,
            # The server skips an item run that it has, so a repeat is safe.
            idempotent=True,
        )
    )


def _run_items(api: Api, run: _Run, created: _Created, runs: _Runs) -> None:
    """Run the items, ``concurrency`` at a time, and post each result. A post or a page fetch
    that fails stops the run: no new item starts, and _run_items raises when the running items
    end."""
    batch = _batch(api, created)
    failures: list[BaseException] = []
    caller = contextvars.copy_context()

    def worker() -> None:
        while not failures:
            try:
                item = runs.next()
                if item is None:
                    return
                # A copy for each item: one Context cannot run in two threads at the same time.
                batch.add(caller.copy().run(_run_item, run, created, item, api.on_error))
            except Exception as exc:
                failures.append(exc)

    with ThreadPoolExecutor(run.concurrency, thread_name_prefix="telemetry-dev-experiment") as pool:
        futures = [pool.submit(worker) for _ in range(run.concurrency)]
        try:
            for future in futures:
                future.result()
        except BaseException as exc:
            # Stops the workers, for example after Ctrl-C.
            failures.append(exc)
            raise
    _end_items(batch, runs, failures)


async def _run_items_async(api: Api, run: _Run, created: _Created, runs: _Runs) -> None:
    batch = _batch(api, created)
    failures: list[BaseException] = []

    async def worker() -> None:
        while not failures:
            try:
                item = await asyncio.to_thread(runs.next)
                if item is None:
                    return
                result = await _run_item_async(run, created, item, api.on_error)
                await asyncio.to_thread(batch.add, result)
            except Exception as exc:
                failures.append(exc)

    # gather runs each worker in its own task, with its own copy of the context.
    await asyncio.gather(*(worker() for _ in range(run.concurrency)))
    await asyncio.to_thread(_end_items, batch, runs, failures)


def _end_items(batch: _Batch, runs: _Runs, failures: list[BaseException]) -> None:
    if not failures:
        try:
            batch.flush()
        except Exception as exc:
            failures.append(exc)
    if failures:
        runs.close()
        raise failures[0]


def _run_item(
    run: _Run,
    experiment: _Created,
    item: ExperimentItem,
    on_error: Callable[[BaseException], None] | None,
) -> dict[str, Any]:
    started = perf_counter()
    output: Any = None
    error: Exception | None = None
    with start_span(
        f"experiment {experiment.name}",
        input=dict(item.input),
        parent=_item_context(experiment.id, item),
    ) as span:
        try:
            output = run.task(item)
            if inspect.isawaitable(output):
                _close(output)
                raise TypeError("task returned an awaitable: use run_experiment_async")
            span.update(output=output)
        except Exception as exc:
            error = exc
            span.update(error=exc)
        trace_id = _trace_id(span)
    result = _item_result(item, trace_id, started, output, error)
    if error is None:
        _add_scores(
            result, [_score(name, scorer, output, item, on_error) for name, scorer in run.scorers]
        )
    return result


async def _run_item_async(
    run: _Run,
    experiment: _Created,
    item: ExperimentItem,
    on_error: Callable[[BaseException], None] | None,
) -> dict[str, Any]:
    started = perf_counter()
    output: Any = None
    error: Exception | None = None
    with start_span(
        f"experiment {experiment.name}",
        input=dict(item.input),
        parent=_item_context(experiment.id, item),
    ) as span:
        try:
            output = run.task(item)
            if inspect.isawaitable(output):
                output = await output
            span.update(output=output)
        except Exception as exc:
            error = exc
            span.update(error=exc)
        trace_id = _trace_id(span)
    result = _item_result(item, trace_id, started, output, error)
    if error is None:
        scores: list[dict[str, Any] | None] = []
        for name, scorer in run.scorers:
            try:
                value = scorer(output, item)
                if inspect.isawaitable(value):
                    value = await value
                scores.append(_to_score(name, value))
            except Exception as exc:
                _scorer_failed(on_error, name, item, exc)
                scores.append(None)
        _add_scores(result, scores)
    return result


def _item_context(experiment_id: str, item: ExperimentItem) -> Context:
    """The context of one item run. It drops the active span, so that the run starts a trace of
    its own, and gives the run a session id of its own: an ambient session or
    ``session_mode="process"`` would otherwise put all runs in one session trace."""
    base = trace.set_span_in_context(trace.INVALID_SPAN, otel_context.get_current())
    return context_with_propagated_attributes(
        {
            **propagated_attributes(base),
            ATTR_EXPERIMENT_ID: experiment_id,
            ATTR_DATASET_ITEM_ID: item.id,
            ATTR_SESSION_ID: f"{experiment_id}:{item.id}:{item.run_index}",
        },
        base,
    )


def _trace_id(span: SpanHandle) -> str | None:
    context = span.span.get_span_context()
    return f"{context.trace_id:032x}" if span.span.is_recording() and context.is_valid else None


def _item_result(
    item: ExperimentItem,
    trace_id: str | None,
    started: float,
    output: Any,
    error: Exception | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "itemId": item.id,
        "runIndex": item.run_index,
        "output": None if error is not None else _output_text(output),
        "latencyMs": round((perf_counter() - started) * 1000),
    }
    if trace_id is not None:
        result["traceId"] = trace_id
    if error is not None:
        result["error"] = truncate(_error_text(error), _MAX_ERROR_LENGTH)
    return result


def _add_scores(result: dict[str, Any], scores: list[dict[str, Any] | None]) -> None:
    kept = [score for score in scores if score is not None]
    if kept:
        result["scores"] = kept


def _score(
    name: str,
    scorer: Callable[[Any, ExperimentItem], Any],
    output: Any,
    item: ExperimentItem,
    on_error: Callable[[BaseException], None] | None,
) -> dict[str, Any] | None:
    # After the span of the item ends, so that the model calls of a scorer do not count in the
    # cost of the item.
    try:
        value = scorer(output, item)
        if inspect.isawaitable(value):
            _close(value)
            raise TypeError("returned an awaitable: use run_experiment_async")
        return _to_score(name, value)
    except Exception as exc:
        _scorer_failed(on_error, name, item, exc)
        return None


def _scorer_failed(
    on_error: Callable[[BaseException], None] | None,
    name: str,
    item: ExperimentItem,
    exc: Exception,
) -> None:
    error = ExperimentError(
        "scorer_failed", f"The scorer {name} failed on item {item.number}: {_error_text(exc)}"
    )
    error.__cause__ = exc
    report_error(on_error, "a scorer failed", error)


def _to_score(name: str, result: object) -> dict[str, Any] | None:
    """The score of a scorer result. It raises for a result that the server would reject."""
    if result is None:
        return None
    score: dict[str, Any] = {"name": name}
    if isinstance(result, bool):
        score.update(_pass_fail(result))
    elif isinstance(result, (int, float)):
        score["value"] = result
    elif isinstance(result, str):
        score["label"] = result.strip()
    elif isinstance(result, Mapping):
        fields = cast("Mapping[str, object]", result)
        value, label, reason = fields.get("value"), fields.get("label"), fields.get("reason")
        for key, text in (("label", label), ("reason", reason)):
            if text is not None and not isinstance(text, str):
                raise TypeError(f"returned a {key} that is not a string")
        if isinstance(value, bool):
            score.update(_pass_fail(value))
        elif value is not None:
            score["value"] = value
        if isinstance(label, str):
            score["label"] = label.strip()
        if isinstance(reason, str):
            score["reason"] = truncate(reason, MAX_REASON_LENGTH)
    else:
        raise TypeError(f"returned a {type(result).__name__}")
    problem = check_value(name, score.get("value"), score.get("label"))
    if problem is not None:
        raise TypeError(f"returned a score that is not valid: {problem}")
    return score


def _pass_fail(passed: bool) -> dict[str, Any]:
    return {"value": 1, "label": "pass"} if passed else {"value": 0, "label": "fail"}


def _output_text(output: Any) -> str | None:
    """The output as text. An output that JSON cannot hold, such as a dict that holds itself, is
    None."""
    if output is None:
        return None
    try:
        text = output if isinstance(output, str) else json_dumps(output)
    except Exception:
        return None
    return truncate(_well_formed(text), _MAX_OUTPUT_LENGTH)


def _error_text(error: BaseException) -> str:
    try:
        text = f"{type(error).__name__}: {error}"
    except Exception:
        # A __str__ that raises.
        return type(error).__name__
    return _well_formed(text)


def _well_formed(text: str) -> str:
    # truncate fails on a lone surrogate, which a str can hold, so each one becomes U+FFFD.
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")


def _close(awaitable: object) -> None:
    # A coroutine that never runs warns when it is collected.
    close = getattr(awaitable, "close", None)
    if callable(close):
        close()


def _finish(api: Api, run: _Run, created: _Created) -> ExperimentResult:
    # The spans of the items reach ingest before the server reads their cost.
    flush()
    _complete(api, created)
    result, body = _wait_for_result(api, created.id)
    write_result(os.environ, body, api.on_error)
    print(_describe(result), flush=True)
    if run.in_ci:
        _check_result(result)
    return result


def _complete(api: Api, created: _Created) -> None:
    request(
        api,
        ExperimentError,
        "POST",
        f"/v1/experiments/{urllib.parse.quote(created.id, safe='')}/complete",
        timeout_s=_REQUEST_TIMEOUT_S,
        idempotent=True,
    )


def _wait_for_result(api: Api, experiment_id: str) -> tuple[ExperimentResult, object]:
    """Wait until the server evaluators finish and the experiment ends."""
    deadline = time.monotonic() + _MAX_WAIT_S
    interval = 1.0
    while True:
        body = request(
            api,
            ExperimentError,
            "GET",
            f"/v1/experiments/{urllib.parse.quote(experiment_id, safe='')}",
            timeout_s=_REQUEST_TIMEOUT_S,
            idempotent=True,
        )
        result = _parse_result(body)
        if result.status in _FINISHED:
            return result, body
        if time.monotonic() + interval > deadline:
            raise ExperimentError(
                "timeout",
                f"The experiment {result.name} did not end in {_MAX_WAIT_S / 60:g} minutes. "
                f"{result.url}",
                result=result,
            )
        time.sleep(interval)
        interval = min(interval * 2, _MAX_POLL_INTERVAL_S)


def _check_result(result: ExperimentResult) -> None:
    """In CI mode, a failed check or an experiment that did not complete fails the run."""
    if result.status != "completed":
        reason = f": {result.error}" if result.error else ""
        raise ExperimentError(
            "experiment_failed",
            f"The experiment {result.name} {result.status}{reason}. {result.url}",
            result=result,
        )
    if result.check_status != "failed":
        return
    failed = [
        f"{score.name} is {_format(score.kind, score.value)}, under the pass mark "
        f"{_format(score.kind, score.min_score)}"
        for score in result.scores
        if score.passed is False
    ]
    if result.items_failed > 0:
        failed.insert(0, _items_failed_text(result))
    detail = f": {'; '.join(failed)}" if failed else ""
    raise ExperimentError(
        "check_failed",
        f"The check of {result.name} failed{detail}. {result.url}",
        result=result,
    )


def _items_failed_text(result: ExperimentResult) -> str:
    """As "5 of 6 items failed": a failed item has no score, so it fails the check."""
    return f"{result.items_failed} of {result.items_done} items failed"


def _describe(result: ExperimentResult) -> str:
    if result.status != "completed":
        outcome = result.status + (f": {result.error}" if result.error else "")
    elif result.check_status is None:
        outcome = "completed"
    else:
        outcome = f"completed. The check {result.check_status}"
    lines = [f"Experiment {result.name} {outcome}. {result.url}"]
    if result.items_failed > 0:
        lines.append(f"  {_items_failed_text(result)}.")
    for score in result.scores:
        mark = (
            ""
            if score.min_score is None
            else f" (pass mark {_format(score.kind, score.min_score)})"
        )
        lines.append(f"  {score.name}: {_format(score.kind, score.value)}{mark}")
    return "\n".join(lines)


def _format(kind: str | None, value: float | None) -> str:
    if value is None:
        return "no score"
    if kind in _RATE_KINDS:
        return f"{_js_round(value * 100)}%"
    rounded = _js_round(value * 100) / 100
    return str(int(rounded)) if rounded.is_integer() else repr(rounded)


def _js_round(value: float) -> int:
    # Math.round of JavaScript, which rounds a half up. round() of Python rounds it to even.
    return math.floor(value + 0.5)


def _parse_created(body: object) -> _Created:
    fields = cast("dict[str, Any]", body) if isinstance(body, dict) else {}
    created_id, name, url = fields.get("id"), fields.get("name"), fields.get("url")
    items_total = fields.get("itemsTotal")
    if (
        not isinstance(created_id, str)
        or not isinstance(name, str)
        or not isinstance(url, str)
        or not _is_int(items_total)
    ):
        raise ExperimentError(
            "invalid_response", "telemetry.dev returned an experiment it cannot read"
        )
    return _Created(created_id, name, items_total, url)


def _parse_page(body: object) -> _Page | None:
    if not isinstance(body, dict):
        return None
    fields = cast("dict[str, Any]", body)
    dataset, items = fields.get("dataset"), fields.get("items")
    next_cursor = fields.get("nextCursor")
    if not isinstance(dataset, dict) or not isinstance(items, list):
        return None
    version = cast("dict[str, Any]", dataset).get("version")
    if not _is_int(version) or version < 0:
        return None
    if next_cursor is not None and not isinstance(next_cursor, str):
        return None
    parsed: list[DatasetItem] = []
    for raw in cast("list[object]", items):
        item = _parse_item(raw)
        if item is None:
            return None
        parsed.append(item)
    return _Page(version, parsed, next_cursor)


def _parse_item(raw: object) -> DatasetItem | None:
    if not isinstance(raw, dict):
        return None
    item = cast("dict[str, Any]", raw)
    item_id, number, tags = item.get("id"), item.get("number"), item.get("tags")
    item_input, metadata = item.get("input"), item.get("metadata")
    expected, split = item.get("expectedOutput"), item.get("split")
    if (
        not isinstance(item_id, str)
        or not _is_int(number)
        or not isinstance(item_input, dict)
        or not (expected is None or isinstance(expected, str))
        or not isinstance(metadata, dict)
        or not isinstance(tags, list)
        or not all(isinstance(tag, str) for tag in cast("list[object]", tags))
        or not (split is None or isinstance(split, str))
    ):
        return None
    return DatasetItem(
        id=item_id,
        number=number,
        input=cast("dict[str, Any]", item_input),
        expected_output=expected,
        metadata=cast("dict[str, Any]", metadata),
        tags=tuple(cast("list[str]", tags)),
        split=split,
    )


def _parse_result(body: object) -> ExperimentResult:
    fields = cast("dict[str, Any]", body) if isinstance(body, dict) else {}
    raw_scores = fields.get("scores")
    scores = (
        [_parse_score(score) for score in cast("list[object]", raw_scores)]
        if isinstance(raw_scores, list)
        else None
    )
    # Required: it decides the result of CI mode, so a missing one must not read as no check.
    check_status = fields.get("checkStatus", "missing")
    items_failed = fields.get("itemsFailed", 0)
    error = fields.get("error")
    if (
        not isinstance(fields.get("id"), str)
        or not isinstance(fields.get("name"), str)
        or not isinstance(fields.get("status"), str)
        or not isinstance(fields.get("url"), str)
        or check_status not in (None, "passed", "failed")
        or not _is_number(fields.get("itemsTotal"))
        or not _is_number(fields.get("itemsDone"))
        or not _is_number(items_failed)
        or not (error is None or isinstance(error, str))
        or scores is None
        or any(score is None for score in scores)
    ):
        raise ExperimentError("invalid_response", "telemetry.dev returned a result it cannot read")
    return ExperimentResult(
        id=fields["id"],
        name=fields["name"],
        status=fields["status"],
        check_status=check_status,
        items_total=int(fields["itemsTotal"]),
        items_done=int(fields["itemsDone"]),
        items_failed=int(items_failed),
        error=error,
        scores=tuple(cast("list[ExperimentScore]", scores)),
        url=fields["url"],
    )


def _parse_score(raw: object) -> ExperimentScore | None:
    """A score of the result. A field that can be None can also be missing."""
    if not isinstance(raw, dict):
        return None
    score = cast("dict[str, Any]", raw)
    name, kind, passed = score.get("name"), score.get("kind"), score.get("passed")
    pass_count, fail_count = score.get("passCount"), score.get("failCount")
    nullable = [
        score.get(key) for key in ("value", "baselineValue", "delta", "minScore", "itemsFailed")
    ]
    if (
        not isinstance(name, str)
        or not (kind is None or isinstance(kind, str))
        or not _is_number(pass_count)
        or not _is_number(fail_count)
        or not all(value is None or _is_number(value) for value in nullable)
        or not (passed is None or isinstance(passed, bool))
    ):
        return None
    value, baseline_value, delta, min_score, items_failed = nullable
    return ExperimentScore(
        name=name,
        kind=kind,
        value=value,
        pass_count=int(cast("float", pass_count)),
        fail_count=int(cast("float", fail_count)),
        baseline_value=baseline_value,
        delta=delta,
        min_score=min_score,
        passed=passed,
        items_failed=None if items_failed is None else int(cast("float", items_failed)),
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
