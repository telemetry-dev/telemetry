import { getCurrentSpan } from "@openai/agents";
import type { SandboxClient } from "@openai/agents/sandbox";

/**
 * Key under which `instrumentSandboxClient` writes the provider's sandbox id into the data of the
 * `sandbox.create_session` or `sandbox.resume_session` span. The Agents SDK records only its
 * backend id there; the provider id lives in the session state.
 */
export const SANDBOX_ID_KEY = "sandbox_id";

type SessionFactory = (...args: never[]) => Promise<{ state?: { sandboxId?: unknown } }>;

/**
 * Records the provider's sandbox id (for example E2B's `sandboxId`) on the Agents SDK span that
 * creates or resumes the session, so telemetry.dev can join the provider's own sandbox events to
 * the trace. Patches `create` and `resume` on this client instance and returns the same instance.
 */
export function instrumentSandboxClient<T extends SandboxClient>(client: T): T {
  // `create` and `resume` are generic over each client's options and state. The wrapper only
  // forwards its arguments and reads `state.sandboxId`, so one loose signature is enough.
  const target = client as unknown as Record<"create" | "resume", SessionFactory | undefined>;

  for (const method of ["create", "resume"] as const) {
    const original = target[method];

    if (original === undefined) continue;

    target[method] = async (...args) => {
      const session = await original.apply(client, args);

      try {
        const sandboxId = session?.state?.sandboxId;
        const span = getCurrentSpan();

        if (typeof sandboxId === "string" && span?.spanData.type === "custom") {
          span.spanData.data[SANDBOX_ID_KEY] = sandboxId;
        }
      } catch {
        // Telemetry must never fail sandbox creation.
      }

      return session;
    };
  }

  return client;
}
