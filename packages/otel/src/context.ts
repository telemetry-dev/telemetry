import {
  type Attributes,
  type Context,
  context as apiContext,
  type ContextManager,
  createContextKey,
  ROOT_CONTEXT,
} from "@opentelemetry/api";
import type { AsyncLocalStorage } from "node:async_hooks";

import { jsonAttr } from "./attrs.ts";
import { diag } from "./debug.ts";

type AlsConstructor = new <T>() => AsyncLocalStorage<T>;
type RuntimeGlobal = typeof globalThis & {
  AsyncLocalStorage?: AlsConstructor;
  process?: {
    getBuiltinModule?: (id: string) => { AsyncLocalStorage?: AlsConstructor } | undefined;
  };
};

// Synchronous, no `node:` static import, no top-level await — the dist must stay
// require(esm)-compatible and runtime-neutral. Vercel Edge exposes AsyncLocalStorage as a global;
// Node >= 20.19 and Workers with nodejs_compat expose process.getBuiltinModule.
function loadAls(): AlsConstructor | undefined {
  const g: RuntimeGlobal = globalThis;
  if (g.AsyncLocalStorage) return g.AsyncLocalStorage;
  try {
    return g.process?.getBuiltinModule?.("node:async_hooks")?.AsyncLocalStorage;
  } catch {}
  return undefined;
}

const AlsCtor = loadAls();

export const als: AsyncLocalStorage<Context> | undefined = AlsCtor
  ? new AlsCtor<Context>()
  : undefined;

/**
 * The active context: our AsyncLocalStorage when available, falling back to the OTel global
 * context (which joins a host app's own OTel setup when one is registered).
 */
export function activeContext(): Context {
  return als?.getStore() ?? apiContext.active();
}

/** Run `fn` with `ctx` active in both our ALS and the global OTel context manager. */
export function withContext<T>(ctx: Context, fn: () => T): T {
  const run = () => apiContext.with(ctx, fn);
  return als ? als.run(ctx, run) : run();
}

export const PROPAGATED_KEY = createContextKey("telemetry.dev propagated attributes");

export interface PropagatedAttributes<MetadataValue = unknown> {
  /** Stamped as user.id on every span and log record in scope. */
  userId?: string;
  /** Stamped as gen_ai.conversation.id on every span and log record in scope. */
  sessionId?: string;
  /** Stamped as td.metadata.<key> on every span and log record in scope. */
  metadata?: Record<string, MetadataValue>;
}

const RESERVED_METADATA_KEYS = new Set(["userId", "sessionId", "user_id", "session_id"]);

export function buildPropagatedAttributes(attrs: PropagatedAttributes): Attributes {
  const out: Attributes = {};
  if (attrs.userId !== undefined) out["user.id"] = attrs.userId;
  if (attrs.sessionId !== undefined) out["gen_ai.conversation.id"] = attrs.sessionId;
  if (attrs.metadata) {
    for (const [key, value] of Object.entries(attrs.metadata)) {
      if (RESERVED_METADATA_KEYS.has(key)) {
        diag.debug(
          `metadata key "${key}" is reserved; use the userId/sessionId fields of propagateAttributes`,
        );
        continue;
      }
      const attr = typeof value === "string" ? value : jsonAttr(value);
      if (attr !== undefined) out[`td.metadata.${key}`] = attr;
    }
  }
  return out;
}

export function propagatedFromContext(ctx: Context): Attributes | undefined {
  return ctx.getValue(PROPAGATED_KEY) as Attributes | undefined;
}

/**
 * Stamp correlation attributes (user, session/conversation, metadata) on every span and log
 * record created inside `fn`. Inner scopes merge over outer ones per key. Works before init().
 */
export function propagateAttributes<T>(attributes: PropagatedAttributes, fn: () => T): T {
  const base = activeContext();
  const merged = { ...propagatedFromContext(base), ...buildPropagatedAttributes(attributes) };
  return withContext(base.setValue(PROPAGATED_KEY, merged), fn);
}

/** Minimal ContextManager over our AsyncLocalStorage, registered only for registerGlobal. */
export class AlsContextManager implements ContextManager {
  constructor(private readonly storage: AsyncLocalStorage<Context>) {}

  active(): Context {
    return this.storage.getStore() ?? ROOT_CONTEXT;
  }

  with<A extends unknown[], F extends (...args: A) => ReturnType<F>>(
    context: Context,
    fn: F,
    thisArg?: ThisParameterType<F>,
    ...args: A
  ): ReturnType<F> {
    const cb = thisArg == null ? fn : fn.bind(thisArg);
    return this.storage.run(context, cb as (...args: A) => ReturnType<F>, ...args);
  }

  bind<T>(context: Context, target: T): T {
    if (typeof target === "function") {
      const storage = this.storage;
      const bound = function (this: unknown, ...args: unknown[]) {
        return storage.run(context, () =>
          (target as (...a: unknown[]) => unknown).apply(this, args),
        );
      };
      return bound as T;
    }
    return target;
  }

  enable(): this {
    return this;
  }

  disable(): this {
    this.storage.disable();
    return this;
  }
}
