from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

import telemetry_dev
from telemetry_dev import ScoreError, score, score_async

from .conftest import MakeClient
from .fake_api import ApiRequest, FakeApi, Reply, fake_api

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"


def replying(reply: Reply) -> Generator[FakeApi]:
    def answer(request: ApiRequest) -> Reply:
        return reply

    with fake_api(override=answer) as api:
        yield api


@pytest.fixture
def accepted() -> Generator[FakeApi]:
    yield from replying((202, {"id": "score-1"}, {}))


@pytest.fixture
def errors(make: MakeClient) -> list[ScoreError]:
    caught: list[ScoreError] = []
    make(on_error=caught.append)
    return caught


def test_posts_the_score_with_the_key_and_base_url_of_init_and_returns_its_id(
    make: MakeClient, accepted: FakeApi
) -> None:
    make(api_key="td_live_key", base_url=f"{accepted.url}/")

    assert (
        score(
            name=" user_feedback ",
            trace_id=TRACE_ID,
            span_id=SPAN_ID,
            value=0,
            comment="Wrong refund amount",
        )
        == "score-1"
    )

    [request] = accepted.requests
    assert request.route == "POST /v1/scores"
    assert request.headers["authorization"] == "Bearer td_live_key"
    assert request.headers["x-telemetry-dev-sdk"] == "telemetry-dev"
    assert request.headers["user-agent"] == f"telemetry-dev-python/{telemetry_dev.__version__}"
    assert request.headers["content-type"] == "application/json"
    assert request.body == {
        "name": "user_feedback",
        "traceId": TRACE_ID,
        "spanId": SPAN_ID,
        "value": 0,
        "comment": "Wrong refund amount",
    }


async def test_a_session_score_needs_no_trace_and_a_label_goes_trimmed(accepted: FakeApi) -> None:
    returned = await score_async(
        api_key="td_live_key",
        base_url=accepted.url,
        name="resolved",
        session_id="chat-9",
        label=" yes ",
    )

    assert returned == "score-1"
    assert accepted.requests[0].body == {"name": "resolved", "sessionId": "chat-9", "label": "yes"}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"trace_id": None}, "Pass trace_id or session_id"),
        (
            {"trace_id": "0" * 32},
            "trace_id must be 32 lowercase hex digits and not all zeros",
        ),
        (
            {"trace_id": TRACE_ID.upper()},
            "trace_id must be 32 lowercase hex digits and not all zeros",
        ),
        (
            {"trace_id": None, "session_id": "s", "span_id": SPAN_ID},
            "Pass the trace_id of the span",
        ),
        ({"value": None}, "Pass a value or a label"),
        ({"value": float("nan")}, "value must be a finite number"),
        ({"value": True}, "value must be a finite number"),
        ({"value": 10**400}, "value must be a finite number"),
        (
            {"name": "user_feedback", "value": 1, "label": "up"},
            "user_feedback takes the value 1 for up or 0 for down, and no label",
        ),
        (
            {"name": "user_feedback", "value": 0.5},
            "user_feedback takes the value 1 for up or 0 for down, and no label",
        ),
        ({"value": 0, "label": "pass"}, "The label pass goes with the value 1"),
        ({"value": 1, "label": "fail"}, "The label fail goes with the value 0"),
        ({"name": "  "}, "name must be a string of 1 to 64 characters"),
        # 33 emoji are 33 characters in Python but 66 UTF-16 units for the API.
        ({"name": "\U0001f600" * 33}, "name must be a string of 1 to 64 characters"),
        ({"comment": "x" * 4001}, "comment must be a string of at most 4000 characters"),
    ],
    ids=[
        "no trace and no session",
        "the all-zero trace id of a span that did not record",
        "an uppercase trace id",
        "a span without its trace",
        "no value and no label",
        "a value that is not finite",
        "a bool value",
        "an int too large for a float",
        "a label for user feedback",
        "user feedback of 0.5",
        "a pass of 0",
        "a fail of 1",
        "a blank name",
        "a name of 66 UTF-16 units",
        "a comment of 4001 characters",
    ],
)
def test_reports_invalid_score_and_sends_nothing(
    errors: list[ScoreError], accepted: FakeApi, options: dict[str, Any], message: str
) -> None:
    arguments: dict[str, Any] = {
        "api_key": "td_live_key",
        "base_url": accepted.url,
        "name": "helpful",
        "trace_id": TRACE_ID,
        "value": 1,
    }
    arguments.update(options)

    assert score(**arguments) is None
    assert accepted.requests == []
    assert [(error.code, error.message) for error in errors] == [("invalid_score", message)]


FAILURES: list[tuple[Reply, str, int | None]] = [
    (
        (403, {"error": {"code": "insufficient_scope", "message": "No scores"}}, {}),
        "insufficient_scope",
        403,
    ),
    ((503, {}, {"retry-after": "1"}), "http_503", 503),
    ((202, {}, {}), "invalid_response", None),
]


@pytest.mark.parametrize(
    ("reply", "code", "status"),
    FAILURES,
    ids=["the API error", "a status with no API error", "a body with no id"],
)
def test_reports_the_failure_tries_once_and_returns_none(
    errors: list[ScoreError], reply: Reply, code: str, status: int | None
) -> None:
    for api in replying(reply):
        assert (
            score(
                api_key="td_live_key", base_url=api.url, name="helpful", trace_id=TRACE_ID, value=1
            )
            is None
        )
        assert len(api.requests) == 1

    assert [(error.code, error.status) for error in errors] == [(code, status)]


def test_without_an_api_key_it_reports_missing_api_key(
    errors: list[ScoreError], accepted: FakeApi
) -> None:
    score(api_key="", base_url=accepted.url, name="helpful", trace_id=TRACE_ID, value=1)

    assert accepted.requests == []
    assert [error.code for error in errors] == ["missing_api_key"]


def test_does_not_follow_a_redirect_so_the_key_stays_with_the_api_host(
    errors: list[ScoreError],
) -> None:
    with fake_api() as other:

        def redirect(request: ApiRequest) -> Reply:
            return 307, {}, {"location": f"{other.url}/v1/scores"}

        with fake_api(override=redirect) as api:
            score(
                api_key="td_live_key", base_url=api.url, name="helpful", trace_id=TRACE_ID, value=1
            )

    assert other.requests == []
    assert [(error.code, error.status) for error in errors] == [("http_307", 307)]
