import { BedrockAgentRuntimeClient } from "@aws-sdk/client-bedrock-agent-runtime";

import { AGENT_HANDLERS } from "./internal-handlers.ts";
import {
  createInstrumentedSend,
  type BedrockInstrumentationOptions,
  type SendFn,
} from "./internal.ts";

export type { BedrockInstrumentationOptions } from "./internal.ts";

export interface WrappableBedrockAgentClient {
  send: (...args: never[]) => any;
}

const WRAPPED = new WeakSet<object>();
let ORIGINAL: { proto: object; hadOwn: boolean; value: unknown; wrapper: SendFn } | undefined;

export function wrapBedrockAgents<T extends WrappableBedrockAgentClient>(
  client: T,
  options: BedrockInstrumentationOptions = {},
): T {
  if (WRAPPED.has(client as object)) return client;
  const original = client.send as SendFn;
  Object.defineProperty(client, "send", {
    value: createInstrumentedSend(original, options, AGENT_HANDLERS),
    configurable: true,
    writable: true,
  });
  WRAPPED.add(client as object);
  return client;
}

export function instrumentBedrockAgents(options: BedrockInstrumentationOptions = {}): void {
  if (ORIGINAL) return;
  const proto = BedrockAgentRuntimeClient.prototype as { send: SendFn };
  const hadOwn = Object.prototype.hasOwnProperty.call(proto, "send");
  const original = proto.send;
  const instrumentedSend = createInstrumentedSend(original, options, AGENT_HANDLERS);
  const wrapper = function telemetryDevBedrockAgentSend(
    this: WrappableBedrockAgentClient,
    ...args: unknown[]
  ) {
    if (WRAPPED.has(this as object)) return original.apply(this, args);
    return instrumentedSend.apply(this, args);
  };
  ORIGINAL = { proto, hadOwn, value: original, wrapper };
  Object.defineProperty(proto, "send", {
    configurable: true,
    writable: true,
    value: wrapper,
  });
}

export function uninstrumentBedrockAgents(): void {
  const original = ORIGINAL;
  if (!original) return;
  if (Object.getOwnPropertyDescriptor(original.proto, "send")?.value !== original.wrapper) return;
  ORIGINAL = undefined;
  if (original.hadOwn) {
    Object.defineProperty(original.proto, "send", {
      configurable: true,
      writable: true,
      value: original.value,
    });
  } else {
    Reflect.deleteProperty(original.proto, "send");
  }
}
