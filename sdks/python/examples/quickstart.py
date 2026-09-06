"""telemetry.dev Python SDK quickstart.

Run with your ingest credentials in the environment:

    TELEMETRY_DEV_API_KEY=td_live_... uv run examples/quickstart.py

Optional:
    TELEMETRY_DEV_BASE_URL    (default https://ingest.telemetry.dev)
    TELEMETRY_DEV_ENVIRONMENT (default production)
    OTEL_SERVICE_NAME         (default unknown_service)
"""

from __future__ import annotations

import telemetry_dev
from telemetry_dev import log, observe, propagate_attributes, start_span, update_current_span


@observe
def plan_route(city: str) -> dict[str, str]:
    log("planning route", attributes={"city": city})
    return {"city": city, "route": "scenic"}


def main() -> None:
    telemetry_dev.init(log_level="info")

    with propagate_attributes(
        user_id="user_123", session_id="session_456", metadata={"plan": "pro"}
    ):
        with start_span("trip-assistant", type="agent"):
            # A generation span with model + usage (cost is computed server-side from tokens).
            with start_span(
                "chat gpt-4o",
                type="generation",
                model="gpt-4o",
                provider="openai",
                input=[{"role": "user", "content": "Plan a day trip to Kyoto"}],
            ):
                log("calling the model")
                update_current_span(
                    output=[{"role": "assistant", "content": "Here is your itinerary..."}],
                    usage={"input_tokens": 11, "output_tokens": 7},
                    finish_reason="stop",
                )

            # A tool span: input/output land on gen_ai.tool.call.arguments/result.
            with start_span("weather-lookup", type="tool", input={"city": "Kyoto"}) as tool:
                tool.update(output={"forecast": "sunny"})

            # A plain observed function (operation "function").
            plan_route("Kyoto")

    log("quickstart finished", level="info")
    telemetry_dev.flush()
    telemetry_dev.shutdown()


if __name__ == "__main__":
    main()
