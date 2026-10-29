from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from telemetry_dev import (
    ExperimentError,
    ExperimentItem,
    ExperimentResult,
    ExperimentScore,
    Scorer,
    log,
    propagate_attributes,
    run_experiment,
    run_experiment_async,
    start_span,
)

from .conftest import MakeClient
from .fake_api import (
    EXPERIMENT_ID,
    EXPERIMENT_URL,
    ApiRequest,
    FakeApi,
    Reply,
    dataset_item,
    fake_api,
    item_id,
    result,
)

ENV_VARS = (
    "TELEMETRY_DEV_EXPERIMENT_DATASET",
    "TELEMETRY_DEV_EXPERIMENT_MIN_SCORES",
    "TELEMETRY_DEV_EXPERIMENT_RESULTS",
    "CI",
    "GITHUB_ACTIONS",
    "GITHUB_REPOSITORY",
    "GITHUB_SHA",
    "GITHUB_EVENT_PATH",
    "GITHUB_RUN_ID",
    "GITHUB_SERVER_URL",
)


@pytest.fixture(autouse=True)
def clear_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    # GitHub Actions sets CI and GITHUB_* for this suite too.
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def api() -> Generator[FakeApi]:
    with fake_api() as server:
        yield server


def answer(item: ExperimentItem) -> str:
    return f"answer to {item.input['question']}"


def experiment(api: FakeApi, **options: Any) -> ExperimentResult:
    arguments: dict[str, Any] = {
        "api_key": "td_live_test",
        "base_url": api.url,
        "dataset": "support-golden",
        "task": answer,
    }
    arguments.update(options)
    return run_experiment(**arguments)


def answer_with(status: int, body: object, headers: dict[str, str] | None = None) -> Reply:
    return status, body, headers or {}


def test_runs_each_item_of_each_page_posts_batches_of_100_on_one_version_then_completes() -> None:
    with fake_api(
        items=[dataset_item(number) for number in range(1, 76)],
        page_size=50,
        results=[result(itemsTotal=150, itemsDone=146, itemsFailed=4)],
    ) as api:
        outcome = experiment(api, runs_per_item=2, evaluators=["Correctness"])

    routes = api.routes()
    assert routes[:2] == ["GET /v1/datasets/support-golden", "POST /v1/experiments"]
    # The second page and the first post of items run at the same time, in either order.
    assert sorted(routes[2:-2]) == [
        "GET /v1/datasets/support-golden",
        f"POST /v1/experiments/{EXPERIMENT_ID}/items",
        f"POST /v1/experiments/{EXPERIMENT_ID}/items",
    ]
    assert routes[-2:] == [
        f"POST /v1/experiments/{EXPERIMENT_ID}/complete",
        f"GET /v1/experiments/{EXPERIMENT_ID}",
    ]
    # The first page names no version; the next one asks for the version of the first.
    pages = [request for request in api.requests if request.path.startswith("/v1/datasets/")]
    assert pages[0].query == {}
    assert pages[1].query == {"version": "3", "cursor": "50"}
    assert api.requests[1].body == {
        "dataset": "support-golden",
        "datasetVersion": 3,
        "evaluators": ["Correctness"],
        "runsPerItem": 2,
    }
    assert api.requests[1].headers["authorization"] == "Bearer td_live_test"
    assert api.requests[1].headers["x-telemetry-dev-sdk"] == "telemetry-dev"

    posts = api.posts()
    assert [len(items) for items in posts] == [100, 50]
    runs = [f"{run['itemId']}:{run['runIndex']}" for items in posts for run in items]
    assert len(set(runs)) == 150
    assert f"{item_id(75)}:1" in runs
    first = posts[0][0]
    assert isinstance(first.pop("latencyMs"), int)
    assert first == {"itemId": item_id(1), "runIndex": 0, "output": "answer to q1"}

    assert outcome == ExperimentResult(
        id=EXPERIMENT_ID,
        name="support-golden-v3",
        status="completed",
        check_status=None,
        items_total=150,
        items_done=146,
        items_failed=4,
        error=None,
        scores=(),
        url=EXPERIMENT_URL,
    )


def test_each_item_run_is_its_own_trace_and_each_span_of_the_task_carries_the_item(
    make: MakeClient, api: FakeApi
) -> None:
    # The session parent of a root span needs the key.
    memory = make(api_key="td_live_test", environment="experiment", session_mode="process")

    def task(item: ExperimentItem) -> dict[str, str]:
        with start_span("chat gpt-4o", type="generation"):
            pass
        log("replied")
        return {"reply": f"r{item.number}"}

    # An active span, an ambient session, and the process session would each put the runs in
    # one trace.
    with (
        propagate_attributes(session_id="ambient-session", user_id="u-1"),
        start_span("eval script"),
    ):
        experiment(api, runs_per_item=2, task=task)

    finished = memory.span_exporter.get_finished_spans()
    chats = [span for span in finished if span.name == "chat gpt-4o"]
    roots = [span for span in finished if span.name == "experiment support-golden-v3"]
    assert len(chats) == 4
    assert len(roots) == 4

    def trace_of(span: Any) -> str:
        return f"{span.context.trace_id:032x}"

    assert len({trace_of(span) for span in roots}) == 4
    assert [span.parent is not None and span.parent.is_remote for span in roots] == [True] * 4
    posted = [run for items in api.posts() for run in items]
    for run in posted:
        root = next(span for span in roots if trace_of(span) == run["traceId"])
        chat = next(span for span in chats if trace_of(span) == run["traceId"])
        assert chat.parent is not None
        assert chat.parent.span_id == root.context.span_id
        for span in (root, chat):
            attributes = dict(span.attributes or {})
            assert attributes["td.experiment.id"] == EXPERIMENT_ID
            assert attributes["td.dataset.item_id"] == run["itemId"]
            assert attributes["gen_ai.conversation.id"] == (
                f"{EXPERIMENT_ID}:{run['itemId']}:{run['runIndex']}"
            )
            assert attributes["user.id"] == "u-1"
    records = memory.log_exporter.get_finished_logs()
    assert len(records) == 4
    for record in records:
        trace_id = f"{record.log_record.trace_id:032x}"
        run = next(run for run in posted if run["traceId"] == trace_id)
        attributes = dict(record.log_record.attributes or {})
        assert attributes["td.experiment.id"] == EXPERIMENT_ID
        assert attributes["td.dataset.item_id"] == run["itemId"]
    assert sorted(run["output"] for run in posted) == [
        '{"reply":"r1"}',
        '{"reply":"r1"}',
        '{"reply":"r2"}',
        '{"reply":"r2"}',
    ]


def test_runs_at_most_concurrency_tasks_at_a_time() -> None:
    lock = threading.Lock()
    running = 0
    most = 0

    def task(item: ExperimentItem) -> str:
        nonlocal running, most
        with lock:
            running += 1
            most = max(most, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return "ok"

    with fake_api(items=[dataset_item(number) for number in range(1, 13)]) as api:
        experiment(api, concurrency=3, task=task)

    assert most == 3
    assert len([run for items in api.posts() for run in items]) == 12


def test_a_task_that_raises_fails_only_its_item(api: FakeApi) -> None:
    def task(item: ExperimentItem) -> str:
        if item.number == 1:
            raise ValueError("no reply")
        return "fine"

    experiment(api, task=task)

    failed, passed = [run for items in api.posts() for run in items]
    assert failed["output"] is None
    assert failed["error"] == "ValueError: no reply"
    assert "scores" not in failed
    assert passed["output"] == "fine"
    assert "error" not in passed


def test_scorers_give_scores_and_a_bad_result_drops_only_its_score(make: MakeClient) -> None:
    errors: list[BaseException] = []
    make(environment="experiment", on_error=errors.append)

    def thrown(output: Any, item: ExperimentItem) -> None:
        raise RuntimeError("judge down")

    scorers: dict[str, Scorer] = {
        "exact": lambda output, item: output == f"answer to {item.input['question']}",
        "polite": lambda output, item: False,
        "length": lambda output, item: len(output),
        "tone": lambda output, item: " friendly ",
        "judged": lambda output, item: {"value": 0.25, "reason": "too short"},
        "none": lambda output, item: None,
        "thrown": thrown,
        "mismatched": lambda output, item: {"value": 0.5, "label": "pass"},
        # 33 emoji are 33 characters in Python but 66 UTF-16 units for the API.
        "wide": lambda output, item: "\U0001f600" * 33,
    }

    with fake_api(items=[dataset_item(1)]) as api:
        experiment(api, scorers=scorers)

    assert api.posts()[0][0]["scores"] == [
        {"name": "exact", "value": 1, "label": "pass"},
        {"name": "polite", "value": 0, "label": "fail"},
        {"name": "length", "value": 12},
        {"name": "tone", "label": "friendly"},
        {"name": "judged", "value": 0.25, "reason": "too short"},
    ]
    assert [(getattr(error, "code", None), str(error)) for error in errors] == [
        ("scorer_failed", "The scorer thrown failed on item 1: RuntimeError: judge down"),
        (
            "scorer_failed",
            "The scorer mismatched failed on item 1: TypeError: returned a score that is not "
            "valid: The label pass goes with the value 1",
        ),
        (
            "scorer_failed",
            "The scorer wide failed on item 1: TypeError: returned a score that is not valid: "
            "label must be a string of 1 to 64 characters",
        ),
    ]


def test_an_output_is_cut_to_100000_utf16_units(api: FakeApi) -> None:
    def task(item: ExperimentItem) -> str:
        return "\U0001f600" * 60_000

    experiment(api, task=task)

    output = api.posts()[0][0]["output"]
    assert len(output.encode("utf-16-le")) // 2 <= 100_000
    assert output.endswith("...[truncated]")


def test_a_failed_post_of_items_stops_new_items_never_completes_and_raises() -> None:
    started = 0

    def task(item: ExperimentItem) -> str:
        nonlocal started
        started += 1
        return "ok"

    def ended(request: ApiRequest) -> Reply | None:
        if request.path.endswith("/items"):
            return answer_with(
                409,
                {"error": {"code": "experiment_ended", "message": "support-golden-v3 has ended."}},
            )
        return None

    with fake_api(items=[dataset_item(n) for n in range(1, 301)], override=ended) as api:
        with pytest.raises(ExperimentError) as caught:
            experiment(api, concurrency=1, task=task)

    assert (caught.value.code, caught.value.status) == ("experiment_ended", 409)
    # The one worker waits for the post of the first 100, which fails, so no item starts after.
    assert started == 100
    assert f"POST /v1/experiments/{EXPERIMENT_ID}/complete" not in api.routes()


def test_an_item_post_that_fails_with_500_goes_again() -> None:
    failures = [1]

    def flaky(request: ApiRequest) -> Reply | None:
        if request.path.endswith("/items") and failures:
            failures.pop()
            return answer_with(500, {})
        return None

    with fake_api(override=flaky) as api:
        experiment(api)

    assert len(api.posts()) == 2
    assert api.posts()[1] == api.posts()[0]


def test_a_create_that_fails_with_500_does_not_go_again() -> None:
    def broken(request: ApiRequest) -> Reply | None:
        return answer_with(500, {}) if request.route == "POST /v1/experiments" else None

    with fake_api(override=broken) as api:
        with pytest.raises(ExperimentError) as caught:
            experiment(api)

    assert (caught.value.code, caught.value.status) == ("http_500", 500)
    assert api.routes().count("POST /v1/experiments") == 1


def test_a_create_that_gets_a_429_goes_again_after_its_retry_after() -> None:
    limited = [1]

    def limit(request: ApiRequest) -> Reply | None:
        if request.route == "POST /v1/experiments" and limited:
            limited.pop()
            return answer_with(
                429,
                {"error": {"code": "rate_limited", "message": "Too many requests"}},
                {"retry-after": "2"},
            )
        return None

    with fake_api(override=limit) as api:
        started = time.monotonic()
        experiment(api)

    assert api.routes().count("POST /v1/experiments") == 2
    # Without the header, the first retry waits 1 second.
    assert time.monotonic() - started >= 2


def test_waits_until_the_server_evaluators_end_the_experiment() -> None:
    with fake_api(results=[result(status="running", itemsDone=1), result()]) as api:
        outcome = experiment(api)

    assert outcome.status == "completed"
    assert api.routes().count(f"GET /v1/experiments/{EXPERIMENT_ID}") == 2


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (
            404,
            {"error": {"code": "dataset_not_found", "message": "No dataset"}},
            "dataset_not_found",
        ),
        (200, {"dataset": {"version": 1}, "items": [{"id": 1}]}, "invalid_response"),
        (
            200,
            {"dataset": {"id": "d", "name": "support-golden", "version": 0}, "items": []},
            "dataset_empty",
        ),
    ],
    ids=["the API error", "a page that cannot be read", "an empty dataset"],
)
def test_raises_before_it_creates_an_experiment(status: int, body: object, code: str) -> None:
    def page(request: ApiRequest) -> Reply | None:
        return answer_with(status, body) if request.path.startswith("/v1/datasets/") else None

    with fake_api(override=page) as api:
        with pytest.raises(ExperimentError) as caught:
            experiment(api)

    assert caught.value.code == code
    assert api.routes() == ["GET /v1/datasets/support-golden"]


def test_a_cursor_that_repeats_is_an_invalid_response() -> None:
    def same(request: ApiRequest) -> Reply | None:
        if not request.path.startswith("/v1/datasets/"):
            return None
        return answer_with(
            200,
            {
                "dataset": {"id": "d", "name": "support-golden", "version": 3},
                "items": [dataset_item(1)],
                "nextCursor": "same",
            },
        )

    with fake_api(override=same) as api:
        with pytest.raises(ExperimentError) as caught:
            experiment(api)

    assert caught.value.code == "invalid_response"
    assert api.routes().count("GET /v1/datasets/support-golden") == 2


FAILED_CHECK = result(
    checkStatus="failed",
    scores=[
        {
            "name": "correctness",
            "kind": "pass_fail",
            "value": 0.85,
            "passCount": 17,
            "failCount": 3,
            "baselineValue": 0.9,
            "delta": -5,
            "minScore": 0.9,
            "passed": False,
        },
        {
            "name": "length",
            "kind": None,
            "value": None,
            "passCount": 0,
            "failCount": 0,
            "baselineValue": None,
            "delta": None,
            "minScore": None,
            "passed": None,
        },
    ],
)


def test_in_ci_mode_a_failed_check_raises_with_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "true")
    with fake_api(results=[FAILED_CHECK]) as api:
        with pytest.raises(ExperimentError) as caught:
            experiment(api, min_scores={"correctness": 0.9})

    assert caught.value.code == "check_failed"
    assert str(caught.value) == (
        "The check of support-golden-v3 failed: correctness is 85%, under the pass mark 90%. "
        f"{EXPERIMENT_URL}"
    )
    assert caught.value.result is not None
    assert caught.value.result.scores[1] == ExperimentScore(
        name="length",
        kind=None,
        value=None,
        pass_count=0,
        fail_count=0,
        baseline_value=None,
        delta=None,
        min_score=None,
        passed=None,
    )


@pytest.mark.parametrize(
    ("options", "ci"),
    [({}, None), ({"ci": False}, "true")],
    ids=["CI is off", "ci=False wins over CI"],
)
def test_a_failed_check_returns_when(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], ci: str | None
) -> None:
    if ci is not None:
        monkeypatch.setenv("CI", ci)
    with fake_api(results=[FAILED_CHECK]) as api:
        assert experiment(api, **options).check_status == "failed"


def test_in_ci_mode_an_experiment_that_fails_raises_with_its_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CI", "1")
    with fake_api(
        results=[result(status="failed", error="0 of 2 items ran. Each one failed.")]
    ) as api:
        with pytest.raises(ExperimentError) as caught:
            experiment(api)

    assert caught.value.code == "experiment_failed"
    assert str(caught.value) == (
        "The experiment support-golden-v3 failed: 0 of 2 items ran. Each one failed.. "
        f"{EXPERIMENT_URL}"
    )


def test_in_github_actions_it_records_the_pull_request_its_head_commit_and_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api: FakeApi
) -> None:
    event = tmp_path / "event.json"
    results = tmp_path / "results.jsonl"
    event.write_text(
        json.dumps(
            {
                "pull_request": {
                    "number": 42,
                    "title": "Shorter replies",
                    "html_url": "https://github.com/acme/support/pull/42",
                    "head": {"sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"},
                }
            }
        )
    )
    for name, value in {
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "acme/support",
        "GITHUB_SHA": "f" * 40,
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_RUN_ID": "991",
        "TELEMETRY_DEV_EXPERIMENT_DATASET": "support-golden",
        "TELEMETRY_DEV_EXPERIMENT_MIN_SCORES": "correctness=0.9\nlength = 12.5",
        "TELEMETRY_DEV_EXPERIMENT_RESULTS": str(results),
    }.items():
        monkeypatch.setenv(name, value)

    experiment(api, dataset=None, min_scores={"correctness": 0.5, "tone": 2})

    create = api.requests[1].body
    assert create["dataset"] == "support-golden"
    assert create["minScores"] == {"correctness": 0.9, "length": 12.5, "tone": 2}
    assert create["ci"] == {
        "provider": "github_actions",
        "repository": "acme/support",
        "commit": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        "prNumber": 42,
        "prTitle": "Shorter replies",
        "prUrl": "https://github.com/acme/support/pull/42",
        "runUrl": "https://github.com/acme/support/actions/runs/991",
    }
    lines = results.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [result()]
    assert results.stat().st_mode & 0o777 == 0o600


def test_a_pull_request_title_is_cut_to_256_utf16_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api: FakeApi
) -> None:
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"title": "\U0001f600" * 200}}))
    for name, value in {
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "acme/support",
        "GITHUB_SHA": "a1b2c3d",
        "GITHUB_EVENT_PATH": str(event),
    }.items():
        monkeypatch.setenv(name, value)

    experiment(api)

    assert api.requests[1].body["ci"]["prTitle"] == "\U0001f600" * 128


def one(output: Any, item: ExperimentItem) -> int:
    return 1


REJECTED: list[tuple[dict[str, str], dict[str, Any], type[Exception], str]] = [
    (
        {},
        {"dataset": None},
        TypeError,
        "Pass dataset, or set the dataset input of the experiment action",
    ),
    (
        {"TELEMETRY_DEV_EXPERIMENT_DATASET": "support-v2"},
        {},
        TypeError,
        "dataset is support-golden, but the experiment action names support-v2. Name one dataset.",
    ),
    (
        {"TELEMETRY_DEV_EXPERIMENT_MIN_SCORES": "correctness=1_000"},
        {},
        TypeError,
        'TELEMETRY_DEV_EXPERIMENT_MIN_SCORES: "correctness=1_000" is not a name=number pair',
    ),
    ({}, {"runs_per_item": 6}, ValueError, "runs_per_item must be an integer from 1 to 5"),
    ({}, {"runs_per_item": True}, ValueError, "runs_per_item must be an integer from 1 to 5"),
    ({}, {"concurrency": 0}, ValueError, "concurrency must be an integer from 1 to 100"),
    (
        {},
        {"scorers": {"user_feedback": one}},
        TypeError,
        "user_feedback is the score of user feedback. Name the scorer another way.",
    ),
    (
        {},
        {"evaluators": ["correctness"], "scorers": {"Correctness": one}},
        TypeError,
        "The scorer Correctness has the name of an evaluator. Rename one of them.",
    ),
    ({}, {"evaluators": "correctness"}, TypeError, "evaluators must be a list of at most"),
]


@pytest.mark.parametrize(
    ("env", "options", "error", "message"),
    REJECTED,
    ids=[
        "no dataset",
        "a dataset that the action does not name",
        "min-scores that do not parse",
        "runs_per_item 6",
        "runs_per_item True",
        "concurrency 0",
        "a scorer named user_feedback",
        "a scorer with the name of an evaluator",
        "evaluators as one string",
    ],
)
def test_raises_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
    api: FakeApi,
    env: dict[str, str],
    options: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(error) as caught:
        experiment(api, **options)

    assert str(caught.value).startswith(message)
    assert api.requests == []


def test_without_an_api_key_it_raises_missing_api_key(api: FakeApi) -> None:
    with pytest.raises(ExperimentError) as caught:
        experiment(api, api_key="")

    assert caught.value.code == "missing_api_key"
    assert api.requests == []


def test_the_task_gets_each_run_of_an_item_of_the_split_with_its_run_index() -> None:
    seen: list[ExperimentItem] = []

    def task(item: ExperimentItem) -> str:
        seen.append(item)
        return "ok"

    with fake_api(items=[dataset_item(7, "test")]) as api:
        experiment(api, split="test", runs_per_item=2, concurrency=1, task=task)

    assert seen == [
        ExperimentItem(
            id=item_id(7),
            number=7,
            input={"question": "q7"},
            expected_output="a7",
            metadata={"source": "test"},
            tags=("golden",),
            split="test",
            run_index=index,
        )
        for index in (0, 1)
    ]
    assert api.requests[0].query["split"] == "test"
    assert api.requests[1].body["split"] == "test"


def test_run_experiment_rejects_an_async_task(api: FakeApi) -> None:
    async def task(item: ExperimentItem) -> str:
        return "ok"

    with pytest.raises(TypeError, match="task is async: use run_experiment_async"):
        experiment(api, task=task)
    assert api.requests == []


async def test_run_experiment_async_runs_async_tasks_and_scorers_at_the_same_time(
    make: MakeClient,
) -> None:
    memory = make(environment="experiment")
    running = 0
    most = 0

    async def task(item: ExperimentItem) -> str:
        nonlocal running, most
        running += 1
        most = max(most, running)
        with start_span("chat gpt-4o", type="generation"):
            await asyncio.sleep(0.05)
        running -= 1
        return f"r{item.number}"

    async def exact(output: Any, item: ExperimentItem) -> bool:
        await asyncio.sleep(0)
        return output == f"r{item.number}"

    with fake_api(items=[dataset_item(number) for number in range(1, 9)]) as api:
        outcome = await run_experiment_async(
            api_key="td_live_test",
            base_url=api.url,
            dataset="support-golden",
            task=task,
            scorers={"exact": exact},
            concurrency=4,
        )

    assert outcome.status == "completed"
    assert most == 4
    posted = sorted(
        (run for items in api.posts() for run in items), key=lambda run: str(run["itemId"])
    )
    assert [run["scores"] for run in posted] == [
        [{"name": "exact", "value": 1, "label": "pass"}]
    ] * 8
    chats = [
        span for span in memory.span_exporter.get_finished_spans() if span.name == "chat gpt-4o"
    ]
    by_trace = {f"{span.context.trace_id:032x}": span for span in chats}
    for run in posted:
        attributes = dict(by_trace[run["traceId"]].attributes or {})
        assert attributes["td.dataset.item_id"] == run["itemId"]
