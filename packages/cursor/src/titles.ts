import { readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Looks up Cursor chat titles from `~/.cursor/chats/<hash>/<conversationId>/meta.json`.
 * Cursor writes titles asynchronously after the first response, so misses retry
 * on the next lookup; only found titles are cached.
 */
export function createTitleLookup(
  chatsDir = join(homedir(), ".cursor", "chats"),
): (conversationId: string) => string | undefined {
  const cache = new Map<string, string>();

  return (conversationId) => {
    if (
      conversationId === "." ||
      conversationId === ".." ||
      conversationId.includes("/") ||
      conversationId.includes("\\")
    ) {
      return undefined;
    }

    const cached = cache.get(conversationId);

    if (cached !== undefined) return cached;
    let hashes: string[];

    try {
      hashes = readdirSync(chatsDir);
    } catch {
      // Missing chats dir (tests, CI) means no titles; telemetry keeps default names.
      return undefined;
    }

    for (const hash of hashes) {
      let raw: string;

      try {
        raw = readFileSync(join(chatsDir, hash, conversationId, "meta.json"), "utf8");
      } catch {
        continue;
      }

      try {
        const meta = JSON.parse(raw) as { title?: unknown };
        const title = meta.title;

        if (String(title) === title && title.length > 0) {
          cache.set(conversationId, title);

          return title;
        }
      } catch {
        // Malformed meta must never break telemetry.
      }

      return undefined;
    }

    return undefined;
  };
}
