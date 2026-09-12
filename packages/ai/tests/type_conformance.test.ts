import type { Telemetry } from "ai";
import type { TelemetryIntegration } from "ai-v6";
import { expect, test } from "vitest";

import { telemetryDev } from "../src/index.ts";
import { telemetryDev as telemetryDevV6 } from "../src/v6.ts";

// Compile-time contract: the emitted types never import from `ai`, so each entry's integration
// object must stay structurally assignable to its major's interface (v6 `TelemetryIntegration`
// was removed in v7; v7 `Telemetry` does not exist in v6). A break here fails typecheck, not
// runtime.
const integ = telemetryDev({ apiKey: "td_live_t" });

void (integ satisfies Telemetry);

const integV6 = telemetryDevV6({ apiKey: "td_live_t" });

void (integV6 satisfies TelemetryIntegration);

test("each entry's telemetryDev satisfies its ai major's integration type", () => {
  expect(integ).toBeDefined();
  expect(integV6).toBeDefined();
});
