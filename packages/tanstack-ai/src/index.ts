import type {
  AbortInfo,
  ChatMiddleware,
  ChatMiddlewareContext,
  ErrorInfo,
  FinishInfo,
  TokenUsage,
} from "@tanstack/ai";
import type { GenerationEmitterOverrides } from "@telemetry-dev/otel";

import type { TelemetryDevOptions } from "./config.ts";
import {
  generationTelemetryDev,
  type GenerationMiddlewareCompat,
  type GenerationMiddlewareContextCompat,
} from "./generation-middleware.ts";
import { chatTelemetryDev } from "./middleware.ts";

export type { GenerationMiddlewareCompat, GenerationMiddlewareContextCompat };

export function telemetryDev(
  options?: TelemetryDevOptions,
  overrides?: GenerationEmitterOverrides,
): ChatMiddleware & GenerationMiddlewareCompat {
  const chat = chatTelemetryDev(options, overrides);
  const generation = generationTelemetryDev(options, overrides);

  const isChat = (
    ctx: ChatMiddlewareContext | GenerationMiddlewareContextCompat,
  ): ctx is ChatMiddlewareContext => !("activity" in ctx) || ctx.activity === "chat";

  return {
    ...chat,
    onStart(ctx: ChatMiddlewareContext | GenerationMiddlewareContextCompat) {
      return isChat(ctx) ? chat.onStart?.(ctx) : generation.onStart?.(ctx);
    },
    onUsage(ctx: ChatMiddlewareContext | GenerationMiddlewareContextCompat, usage: TokenUsage) {
      return isChat(ctx) ? chat.onUsage?.(ctx, usage) : generation.onUsage?.(ctx, usage);
    },
    onFinish(
      ctx: ChatMiddlewareContext | GenerationMiddlewareContextCompat,
      info: { duration: number; usage?: TokenUsage },
    ) {
      return isChat(ctx)
        ? chat.onFinish?.(ctx, info as FinishInfo)
        : generation.onFinish?.(ctx, info);
    },
    onError(ctx: ChatMiddlewareContext | GenerationMiddlewareContextCompat, info: ErrorInfo) {
      return isChat(ctx) ? chat.onError?.(ctx, info) : generation.onError?.(ctx, info);
    },
    onAbort(ctx: ChatMiddlewareContext | GenerationMiddlewareContextCompat, info: AbortInfo) {
      return isChat(ctx) ? chat.onAbort?.(ctx, info) : generation.onAbort?.(ctx, info);
    },
  };
}

export type { TelemetryDevOptions } from "./config.ts";
