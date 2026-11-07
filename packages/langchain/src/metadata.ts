type Metadata = Record<string, unknown>;

// LangGraph copies `configurable.thread_id` into the metadata of each run of the graph.
const SESSION_KEYS = ["sessionId", "session_id", "thread_id", "conversation_id"];
const USER_KEYS = ["userId", "user_id"];
const IDENTITY_KEYS = new Set([...SESSION_KEYS, ...USER_KEYS]);

// Keys that LangChain and LangGraph add to the metadata of a run.
const INTERNAL_PREFIXES = ["ls_", "langgraph_", "checkpoint_", "__"];
const INTERNAL_KEYS = new Set(["versions"]);

/** The session and user of a run, from the keys of its metadata. */
export function identityAttributes(metadata: Metadata | undefined): Record<string, string> {
  const attributes: Record<string, string> = {};
  const session = firstId(metadata, SESSION_KEYS);
  const user = firstId(metadata, USER_KEYS);

  if (session !== undefined) attributes["gen_ai.conversation.id"] = session;

  if (user !== undefined) attributes["user.id"] = user;

  return attributes;
}

/** The metadata that the app gave the run, without the keys of LangChain and LangGraph. */
export function userMetadata(metadata: Metadata | undefined): Metadata | undefined {
  if (!metadata) return undefined;

  const entries = Object.entries(metadata).filter(
    ([key, value]) =>
      value !== undefined &&
      !IDENTITY_KEYS.has(key) &&
      !INTERNAL_KEYS.has(key) &&
      !INTERNAL_PREFIXES.some((prefix) => key.startsWith(prefix)),
  );

  return entries.length > 0 ? Object.fromEntries(entries) : undefined;
}

function firstId(metadata: Metadata | undefined, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = metadata?.[key];

    if (typeof value === "string" && value.length > 0) return value;

    if (typeof value === "number" || typeof value === "bigint") return value.toString();
  }

  return undefined;
}
