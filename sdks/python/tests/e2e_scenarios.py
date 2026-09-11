"""E2E scenario driver for the cross-SDK conformance suite.

This script EMITS the shared conformance scenario against a real ingest endpoint; the
ClickHouse row assertions live in apps/ingest/src/sdk_conformance.ts (C1-C19), keyed by the
session id below. Keep the scenario in lockstep with apps/ingest/src/sdk_otlp_e2e_test.ts and
docs/sdk-conformance.md.

Run:

    TELEMETRY_DEV_API_KEY=td_live_... \
    TELEMETRY_DEV_BASE_URL=http://localhost:4318 \
    TD_E2E_SESSION_ID=conv-<unique> \
    uv run python tests/e2e_scenarios.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import telemetry_dev
from telemetry_dev import (
    MaskContext,
    get_traceparent,
    log,
    observe,
    propagate_attributes,
    start_span,
)


@observe(name="format-output")  # C5: observed plain function -> operation "function"
def format_output(text: str) -> dict[str, bool]:
    return {"formatted": True}


def mask(value: Any, ctx: MaskContext) -> Any:
    """C16b: redact any string input containing SECRET (shared with the TS scenario)."""
    if ctx.key == "gen_ai.input.messages" and isinstance(value, str) and "SECRET" in value:
        return {"masked": True}
    return value


def main() -> int:
    api_key = os.environ.get("TELEMETRY_DEV_API_KEY")
    base_url = os.environ.get("TELEMETRY_DEV_BASE_URL")
    session_id = os.environ.get("TD_E2E_SESSION_ID")
    if not api_key or not session_id:
        print(
            "e2e_scenarios: TELEMETRY_DEV_API_KEY and TD_E2E_SESSION_ID are required",
            file=sys.stderr,
        )
        return 1

    # service_name/environment come from OTEL_SERVICE_NAME / TELEMETRY_DEV_ENVIRONMENT.
    telemetry_dev.init(
        api_key=api_key,
        base_url=base_url,
        export_mode="immediate",
        mask=mask,
        disable_atexit=True,
    )

    # C6: stamped on every span and log record emitted inside this block.
    agent_traceparent: str | None = None
    with propagate_attributes(user_id="user_e2e", session_id=session_id, metadata={"plan": "pro"}):
        # C2: agent root span.
        with start_span("support-agent", type="agent", agent_name="support"):
            agent_traceparent = get_traceparent()
            # C3: generation span with model/provider/usage (server computes cost).
            with start_span(
                "chat-completion",
                type="generation",
                model="gpt-4o",
                provider="openai",
                input=[{"role": "user", "content": "What is the weather?"}],
            ) as generation:
                # C8: log inside the generation span (trace-correlated otlp-log row).
                log("inside generation", event_name="e2e.inside")
                generation.update(
                    output={"role": "assistant", "content": "It is sunny."},
                    usage={"input_tokens": 11, "output_tokens": 7},
                    finish_reason="stop",
                )

            # C4: tool span -> gen_ai.tool.call.arguments/result.
            with start_span(
                "web-search",
                type="tool",
                tool_name="web-search",
                tool_call_id="call_1",
                input={"q": "weather"},
            ) as tool:
                tool.update(output={"hits": 1})

            # C5: observed plain function (input binds to {"text": ...} in both SDKs).
            format_output(text="It is sunny.")

            # C7/C9: error span (status ERROR + error.type + exception event -> error log).
            try:
                with start_span("broken-step"):
                    raise ValueError("boom")
            except ValueError:
                pass

            # C12: embedding span.
            start_span(
                "embed-query",
                type="embedding",
                model="text-embedding-4",
                usage={"input_tokens": 3},
            ).end()
            # C14: per-call capture_input=False.
            start_span("private-step", input="should-not-appear", capture_input=False).end()
            # C16: oversized content gets truncated at the shared UTF-16 cap with the marker.
            start_span("big-payload", input="x" * 10_000 + "\U0001f916" * 30_000).end()
            # C15: client-side cost override.
            start_span(
                "priced-call",
                type="generation",
                model="custom-model-x",
                provider="custom",
                usage={"input_tokens": 5, "output_tokens": 2},
                cost_usd=0.5,
            ).end()
            start_span(
                "openrouter-call",
                type="generation",
                provider="openrouter",
            ).end()
            # C16b: mask hook redaction.
            start_span("masked-step", type="generation", input="SECRET stuff").end()
            streamed = start_span(
                "streamed-chat",
                type="generation",
                model="gpt-4o",
                provider="openai",
                input=[{"role": "user", "content": "Stream please"}],
            )
            for timestamp in (1000, 1010, 1050, 1210):
                streamed.record_output_chunk(timestamp)
            streamed.end(
                time_to_first_chunk_ms=250,
                output={"role": "assistant", "content": "chunk..."},
            )

        # C13: continue the trace from a serialized W3C traceparent.
        start_span("background-job", parent=agent_traceparent).end()
        # C20: a second root call of the same session joins the session trace.
        start_span("follow-up-turn", type="agent", agent_name="support").end()

    # C10: log outside any span (standalone record, no trace correlation).
    log("outside spans", level="warn", event_name="e2e.outside")

    # C11: auto-metrics histograms flush on shutdown.
    telemetry_dev.flush()
    telemetry_dev.shutdown()

    telemetry_dev.init(
        api_key=api_key,
        base_url=base_url,
        export_mode="immediate",
        session_mode="process",
        disable_atexit=True,
    )
    start_span("process-root-a").end()
    start_span("process-root-b").end()
    with start_span(
        "explicit-process-root",
        attributes={"gen_ai.conversation.id": f"{session_id}_explicit"},
    ):
        start_span("explicit-process-child").end()
        log("explicit process log")
    log("process standalone log")
    telemetry_dev.flush()
    telemetry_dev.shutdown()

    # C17: fail-open — without a key the SDK emits nothing and never raises.
    os.environ.pop("TELEMETRY_DEV_API_KEY", None)
    telemetry_dev.init(log_level="silent", disable_atexit=True)
    format_output(text="ignored-no-key")
    log("ignored-no-key")
    telemetry_dev.flush()
    telemetry_dev.shutdown()
    print(f"e2e_scenarios: emitted conformance scenario for session {session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
