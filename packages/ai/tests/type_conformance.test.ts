import type { Telemetry } from "ai";
import type { TelemetryIntegration } from "ai-v6";
import { expectTypeOf, test } from "vitest";

import { telemetryDev } from "../src/index.ts";
import { telemetryDev as telemetryDevV6 } from "../src/v6.ts";

// The emitted types never import from `ai`, so each entry's integration object must stay
// structurally assignable to its major's interface (v6 `TelemetryIntegration` was removed in v7;
// v7 `Telemetry` does not exist in v6). A break fails `vp check`, not the test run.
test("each entry's telemetryDev is assignable to its ai major's integration type (compile-time)", () => {
  expectTypeOf(telemetryDev).returns.toExtend<Telemetry>();
  expectTypeOf(telemetryDevV6).returns.toExtend<TelemetryIntegration>();
});
