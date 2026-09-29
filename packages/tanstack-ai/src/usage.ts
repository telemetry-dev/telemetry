import type { TokenUsage } from "@tanstack/ai";

// These adapters report promptTokens without cache reads and writes; span input tokens include them.
const cacheExclusiveProviders = new Set(["anthropic", "bedrock-converse"]);

export function inputTokens(
  provider: string | undefined,
  usage: TokenUsage | null | undefined,
): number | undefined {
  const promptTokens = usage?.promptTokens;

  if (promptTokens === undefined || provider === undefined) return promptTokens;

  if (!cacheExclusiveProviders.has(provider)) return promptTokens;
  const details = usage?.promptTokensDetails;

  return promptTokens + (details?.cachedTokens ?? 0) + (details?.cacheWriteTokens ?? 0);
}
