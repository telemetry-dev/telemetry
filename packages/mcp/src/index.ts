import { propagation, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import {
  activeContext,
  extractW3cContext,
  injectW3cContext,
  type SpanHandle,
  startSpan,
  withContext,
} from "@telemetry-dev/sdk";

interface McpTransport {
  send: (...args: never[]) => Promise<void>;
  onmessage?: (...args: never[]) => void;
  onclose?: () => void;
  sessionId?: string;
  readonly protocolVersion?: string;
  setProtocolVersion?: (version: string) => void;
}

export interface InstrumentMcpTransportOptions {
  capturePayloads?: boolean;
  propagateBaggage?: boolean;
}

type JsonRpcId = string | number;
type JsonRecord = Record<string, unknown>;
type MessageHandler = (message: unknown, extra?: unknown) => void;
type Send = (message: unknown, options?: unknown) => Promise<void>;

interface SendOptions extends JsonRecord {
  requestSignal?: AbortSignal;
  onRequestStreamEnd?: () => void;
  resumptionToken?: string;
}

interface InstrumentableTransport {
  send: Send;
  onmessage?: MessageHandler;
  onclose?: () => void;
  sessionId?: string;
  protocolVersion?: string;
  setProtocolVersion?: (version: string) => void;
}

interface PendingRequest {
  handle: SpanHandle;
  method: string;
  receiver: boolean;
  capturePayloads: boolean;
  completed: boolean;
  removeAbortListener?: () => void;
}

interface ParsedMessage {
  message: JsonRecord;
  method?: string;
  id?: JsonRpcId;
  params?: JsonRecord;
  meta?: JsonRecord;
}

interface RequestSpanAttributes extends Record<string, string> {
  "gen_ai.operation.name": string;
  "mcp.method.name": string;
}

interface InjectedRequest extends JsonRecord {
  params: JsonRecord;
}

const INSTRUMENTED = new WeakSet<object>();
const PROTOCOL_VERSIONS = new WeakMap<object, string>();
const CALLER_ERROR_CODES = new Set([-32700, -32600, -32601, -32602, -32002, -32021, -32022]);
const PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion";

function asRecord(value: unknown): JsonRecord | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : undefined;
}

function parseMessage(value: unknown): ParsedMessage | undefined {
  const message = asRecord(value);
  if (message === undefined) return undefined;
  const params = asRecord(message.params);
  return {
    message,
    method: requestMethod(message),
    id: requestId(message),
    params,
    meta: asRecord(params?._meta),
  };
}

function requestId(message: JsonRecord): JsonRpcId | undefined {
  if (!("id" in message)) return undefined;
  const id = message.id;
  return typeof id === "string" || typeof id === "number" ? id : undefined;
}

function idKey(id: JsonRpcId): string {
  return `${typeof id}:${String(id)}`;
}

function requestMethod(message: JsonRecord): string | undefined {
  return typeof message.method === "string" ? message.method : undefined;
}

function isRequest(
  message: ParsedMessage | undefined,
): message is ParsedMessage & { method: string; id: JsonRpcId } {
  return (
    message?.method !== undefined &&
    message.id !== undefined &&
    message.method !== "notifications/cancelled"
  );
}

function targetFor(method: string, params: JsonRecord | undefined): string | undefined {
  if (method !== "tools/call" && method !== "prompts/get") return undefined;
  return typeof params?.name === "string" ? params.name : undefined;
}

function resourceUri(method: string, params: JsonRecord | undefined): string | undefined {
  if (
    method !== "resources/read" &&
    method !== "resources/subscribe" &&
    method !== "resources/unsubscribe" &&
    method !== "notifications/resources/updated"
  ) {
    return undefined;
  }
  return typeof params?.uri === "string" ? params.uri : undefined;
}

function metaProtocolVersion(meta: JsonRecord | undefined): string | undefined {
  const version = meta?.[PROTOCOL_VERSION_META_KEY];
  return typeof version === "string" ? version : undefined;
}

function protocolVersion(
  transport: InstrumentableTransport,
  message: ParsedMessage,
): string | undefined {
  const modern = metaProtocolVersion(message.meta);
  if (modern !== undefined) return modern;
  if (message.method === "initialize" && typeof message.params?.protocolVersion === "string") {
    PROTOCOL_VERSIONS.set(transport, message.params.protocolVersion);
    return message.params.protocolVersion;
  }
  return PROTOCOL_VERSIONS.get(transport) ?? transport.protocolVersion;
}

function responseProtocolVersion(
  transport: InstrumentableTransport,
  method: string,
  message: ParsedMessage,
): string | undefined {
  const modern = metaProtocolVersion(message.meta);
  if (modern !== undefined) return modern;
  const result = asRecord(message.message.result);
  if (method === "initialize" && typeof result?.protocolVersion === "string") {
    PROTOCOL_VERSIONS.set(transport, result.protocolVersion);
    return result.protocolVersion;
  }
  return PROTOCOL_VERSIONS.get(transport) ?? transport.protocolVersion;
}

function requestAttributes(
  transport: InstrumentableTransport,
  message: ParsedMessage,
): RequestSpanAttributes {
  const method = message.method!;
  const attributes: RequestSpanAttributes = {
    "gen_ai.operation.name": method === "tools/call" ? "execute_tool" : "mcp",
    "mcp.method.name": method,
  };
  if (message.id !== undefined) {
    attributes["jsonrpc.request.id"] = String(message.id);
  }
  if (typeof transport.sessionId === "string") attributes["mcp.session.id"] = transport.sessionId;
  const version = protocolVersion(transport, message);
  if (version !== undefined) attributes["mcp.protocol.version"] = version;
  const target = targetFor(method, message.params);
  if (method === "tools/call" && target !== undefined) attributes["gen_ai.tool.name"] = target;
  if (method === "prompts/get" && target !== undefined) attributes["gen_ai.prompt.name"] = target;
  const uri = resourceUri(method, message.params);
  if (uri !== undefined) attributes["mcp.resource.uri"] = uri;
  return attributes;
}

function startRequest(
  transport: InstrumentableTransport,
  message: ParsedMessage,
  receiver: boolean,
  capturePayloads: boolean,
  propagateBaggage: boolean,
): PendingRequest | undefined {
  const method = message.method;
  if (method === undefined) return undefined;
  const target = targetFor(method, message.params);
  const ambient = activeContext();
  const extracted = receiver
    ? extractW3cContext(message.meta ?? {}, { includeBaggage: propagateBaggage })
    : undefined;
  const ambientSpan = trace.getSpanContext(ambient);
  const remoteSpan = extracted === undefined ? undefined : trace.getSpanContext(extracted);
  const remoteIsValid = remoteSpan !== undefined && trace.isSpanContextValid(remoteSpan);
  const remoteBaggage = extracted === undefined ? undefined : propagation.getBaggage(extracted);
  const parent = remoteIsValid
    ? extracted
    : remoteBaggage === undefined
      ? undefined
      : propagation.setBaggage(ambient, remoteBaggage);
  const links =
    ambientSpan !== undefined &&
    trace.isSpanContextValid(ambientSpan) &&
    remoteIsValid &&
    (ambientSpan.traceId !== remoteSpan.traceId || ambientSpan.spanId !== remoteSpan.spanId)
      ? [{ context: ambientSpan }]
      : undefined;
  const handle = startSpan(target === undefined ? method : `${method} ${target}`, {
    type: method === "tools/call" ? "tool" : "span",
    kind: receiver ? SpanKind.SERVER : SpanKind.CLIENT,
    parent,
    links,
    attributes: requestAttributes(transport, message),
    input: capturePayloads && method === "tools/call" ? message.params?.arguments : undefined,
  });
  return { handle, method, receiver, capturePayloads, completed: false };
}

function updateObservableAttributes(
  transport: InstrumentableTransport,
  pending: PendingRequest,
  message: ParsedMessage,
): void {
  const attributes: Record<string, string> = {};
  if (typeof transport.sessionId === "string") attributes["mcp.session.id"] = transport.sessionId;
  const version = responseProtocolVersion(transport, pending.method, message);
  if (version !== undefined) attributes["mcp.protocol.version"] = version;
  pending.handle.update({ attributes });
}

function setFailure(pending: PendingRequest, type: string, description?: string): void {
  pending.handle.span.setAttribute("error.type", type);
  pending.handle.span.setStatus({ code: SpanStatusCode.ERROR, message: description });
}

function finishResponse(
  transport: InstrumentableTransport,
  pending: PendingRequest,
  message: ParsedMessage,
): void {
  updateObservableAttributes(transport, pending, message);
  const error = asRecord(message.message.error);
  if (error !== undefined) {
    const code = typeof error.code === "number" ? error.code : undefined;
    const codeText = code === undefined ? "_OTHER" : String(code);
    pending.handle.span.setAttribute("rpc.response.status_code", codeText);
    if (!pending.receiver || code === undefined || !CALLER_ERROR_CODES.has(code)) {
      setFailure(pending, codeText, typeof error.message === "string" ? error.message : undefined);
    }
  } else {
    const result = asRecord(message.message.result);
    if (pending.method === "tools/call" && result?.isError === true) {
      setFailure(pending, "tool_error");
    }
    if (pending.capturePayloads && pending.method === "tools/call" && result?.isError !== true) {
      pending.handle.update({ output: message.message.result });
    }
  }
}

function completePending(
  requests: Map<string, PendingRequest>,
  key: string,
  pending: PendingRequest,
  update?: () => void,
): void {
  if (pending.completed) return;
  pending.completed = true;
  if (requests.get(key) === pending) requests.delete(key);
  pending.removeAbortListener?.();
  try {
    update?.();
  } catch {}
  try {
    pending.handle.end();
  } catch {}
}

function completeAs(
  requests: Map<string, PendingRequest>,
  id: JsonRpcId,
  type: "cancelled" | "connection_error",
): void {
  const key = idKey(id);
  const pending = requests.get(key);
  if (pending !== undefined) {
    completePending(requests, key, pending, () => setFailure(pending, type));
  }
}

function cancellationId(message: ParsedMessage): JsonRpcId | undefined {
  if (message.method !== "notifications/cancelled") return undefined;
  if (message.params === undefined || !("requestId" in message.params)) return undefined;
  const id = message.params.requestId;
  return typeof id === "string" || typeof id === "number" ? id : undefined;
}

function prepareInjectedRequest(message: ParsedMessage) {
  const params = { ...message.params };
  const meta = { ...message.meta };
  params._meta = meta;
  return { message: { ...message.message, params }, meta };
}

export function instrumentMcpTransport<T extends McpTransport>(
  transport: T,
  options: InstrumentMcpTransportOptions = {},
): T {
  if (INSTRUMENTED.has(transport)) return transport;
  INSTRUMENTED.add(transport);

  const target = transport as T & InstrumentableTransport;
  const originalSend = target.send.bind(target);
  const originalSetProtocolVersion = target.setProtocolVersion?.bind(target);
  const outgoing = new Map<string, PendingRequest>();
  const incoming = new Map<string, PendingRequest>();
  const sendingResponses = new Set<PendingRequest>();
  const dispatching = new WeakSet<object>();
  const capturePayloads = options.capturePayloads === true;
  const propagateBaggage = options.propagateBaggage === true;
  let onmessage = target.onmessage;
  const wrapClose = (handler: (() => void) | undefined) =>
    function () {
      for (const [key, pending] of outgoing) {
        completePending(outgoing, key, pending, () => setFailure(pending, "connection_error"));
      }
      for (const [key, pending] of incoming) {
        if (sendingResponses.has(pending)) continue;
        completePending(incoming, key, pending, () => setFailure(pending, "connection_error"));
      }
      handler?.call(target);
    };
  let onclose = wrapClose(target.onclose);

  target.send = async (value, sendOptions) => {
    const values = Array.isArray(value) ? value : [value];
    let messages: Array<ParsedMessage | undefined> = [];
    let cancellations: Array<JsonRpcId | undefined> = [];
    let started: Array<{ index: number; key: string; pending: PendingRequest }> = [];
    let requestSignal: AbortSignal | undefined;
    let sent = value;
    let forwardedOptions = sendOptions;

    try {
      messages = values.map(parseMessage);
      cancellations = messages.map((message) =>
        message === undefined ? undefined : cancellationId(message),
      );
      const rawOptions = asRecord(sendOptions) as SendOptions | undefined;
      const copiedOptions = rawOptions === undefined ? undefined : { ...rawOptions };
      const resumeOnly =
        typeof copiedOptions?.resumptionToken === "string" &&
        copiedOptions.resumptionToken.length > 0;
      requestSignal = copiedOptions?.requestSignal;
      const originalStreamEnd = copiedOptions?.onRequestStreamEnd;
      const requestCounts = new Map<string, number>();
      if (!resumeOnly) {
        for (const message of messages) {
          if (!isRequest(message)) continue;
          const key = idKey(message.id);
          requestCounts.set(key, (requestCounts.get(key) ?? 0) + 1);
        }
      }

      const transformed = [...values];
      const prepared: Array<{
        index: number;
        key: string;
        parsed: ParsedMessage;
        message: InjectedRequest;
        meta: JsonRecord;
      }> = [];
      const cancelledKeys = new Set<string>();
      for (const [index, message] of messages.entries()) {
        const cancelled = cancellations[index];
        if (cancelled !== undefined) {
          const key = idKey(cancelled);
          cancelledKeys.add(key);
        }
        if (!isRequest(message) || resumeOnly) continue;
        const key = idKey(message.id);
        if (requestCounts.get(key) !== 1 || (outgoing.has(key) && !cancelledKeys.has(key))) {
          continue;
        }
        const injected = prepareInjectedRequest(message);
        prepared.push({ index, key, parsed: message, ...injected });
      }

      for (const request of prepared) {
        const pending = startRequest(
          target,
          request.parsed,
          false,
          capturePayloads,
          propagateBaggage,
        );
        if (pending !== undefined) {
          started.push({ index: request.index, key: request.key, pending });
          injectW3cContext(pending.handle.context, request.meta, {
            includeBaggage: propagateBaggage,
          });
          transformed[request.index] = request.message;
        }
      }

      if (started.length > 0) sent = Array.isArray(value) ? transformed : transformed[0];
      if (started.length > 0 && copiedOptions !== undefined) {
        forwardedOptions = {
          ...copiedOptions,
          onRequestStreamEnd: function (this: unknown) {
            try {
              originalStreamEnd?.call(this);
            } finally {
              for (const { key, pending } of started) {
                completePending(outgoing, key, pending, () =>
                  setFailure(pending, "connection_error"),
                );
              }
            }
          },
        } satisfies SendOptions;
      }
    } catch {
      for (const { pending } of started) {
        try {
          pending.handle.end();
        } catch {}
      }
      messages = [];
      cancellations = [];
      started = [];
      requestSignal = undefined;
      sent = value;
      forwardedOptions = sendOptions;
    }

    const startedByIndex = new Map(started.map((request) => [request.index, request]));
    const signal = requestSignal;
    for (const [index, cancelled] of cancellations.entries()) {
      if (cancelled !== undefined) completeAs(outgoing, cancelled, "cancelled");
      const request = startedByIndex.get(index);
      if (request !== undefined) {
        const { key, pending } = request;
        if (outgoing.has(key)) {
          completePending(outgoing, key, pending, () =>
            setFailure(pending, "duplicate_request_id"),
          );
          continue;
        }
        outgoing.set(key, pending);
        if (signal !== undefined) {
          const onAbort = () =>
            completePending(outgoing, key, pending, () => setFailure(pending, "cancelled"));
          try {
            signal.addEventListener("abort", onAbort, { once: true });
            pending.removeAbortListener = () => signal.removeEventListener("abort", onAbort);
            if (signal.aborted) onAbort();
          } catch {}
        }
      }
    }

    const responses: Array<{ key: string; pending: PendingRequest; message: ParsedMessage }> = [];
    for (const message of messages) {
      if (message === undefined || message.method !== undefined || message.id === undefined)
        continue;
      const key = idKey(message.id);
      const pending = incoming.get(key);
      if (pending === undefined || sendingResponses.has(pending)) continue;
      sendingResponses.add(pending);
      responses.push({ key, pending, message });
    }

    try {
      try {
        await originalSend(sent, forwardedOptions);
      } catch (error) {
        for (const { key, pending } of started) {
          completePending(outgoing, key, pending, () => pending.handle.update({ error }));
        }
        for (const { key, pending } of responses) {
          completePending(incoming, key, pending, () => pending.handle.update({ error }));
        }
        throw error;
      }

      try {
        for (const { key, pending, message } of responses) {
          completePending(incoming, key, pending, () => finishResponse(target, pending, message));
        }
      } catch {}
    } finally {
      for (const { pending } of responses) sendingResponses.delete(pending);
    }
  };

  if (originalSetProtocolVersion !== undefined) {
    target.setProtocolVersion = (version) => {
      PROTOCOL_VERSIONS.set(target, version);
      originalSetProtocolVersion(version);
    };
  }

  Object.defineProperty(target, "onmessage", {
    configurable: true,
    enumerable: true,
    get: () => onmessage,
    set: (handler: MessageHandler | undefined) => {
      onmessage =
        handler === undefined
          ? undefined
          : function (value, extra) {
              const object = value !== null && typeof value === "object" ? value : undefined;
              if (object !== undefined && dispatching.has(object)) {
                handler.call(target, value, extra);
                return;
              }
              if (object !== undefined) dispatching.add(object);

              let invoked = false;
              const invoke = () => {
                invoked = true;
                return handler.call(target, value, extra);
              };
              const received: Array<{ key: string; pending: PendingRequest }> = [];
              const completeFailure = (error: unknown) => {
                for (const { key, pending } of received) {
                  completePending(incoming, key, pending, () => pending.handle.update({ error }));
                }
              };
              const completeAsyncFailure = (result: unknown): unknown => {
                if (
                  result === null ||
                  (typeof result !== "object" && typeof result !== "function") ||
                  !("then" in result) ||
                  typeof result.then !== "function"
                ) {
                  return result;
                }
                return Promise.resolve(result).catch((error: unknown) => {
                  completeFailure(error);
                  throw error;
                });
              };

              try {
                try {
                  const values = Array.isArray(value) ? value : [value];
                  const messages = values.map(parseMessage);
                  const requestCounts = new Map<string, number>();
                  for (const message of messages) {
                    if (!isRequest(message)) continue;
                    const key = idKey(message.id);
                    requestCounts.set(key, (requestCounts.get(key) ?? 0) + 1);
                  }

                  for (const message of messages) {
                    if (message === undefined) continue;
                    const cancelled = cancellationId(message);
                    if (cancelled !== undefined) completeAs(incoming, cancelled, "cancelled");

                    if (isRequest(message)) {
                      const key = idKey(message.id);
                      if (requestCounts.get(key) !== 1 || incoming.has(key)) continue;
                      const pending = startRequest(
                        target,
                        message,
                        true,
                        capturePayloads,
                        propagateBaggage,
                      );
                      if (pending !== undefined) {
                        incoming.set(key, pending);
                        received.push({ key, pending });
                      }
                    } else if (message.method === undefined && message.id !== undefined) {
                      const key = idKey(message.id);
                      const pending = outgoing.get(key);
                      if (pending !== undefined) {
                        completePending(outgoing, key, pending, () =>
                          finishResponse(target, pending, message),
                        );
                      }
                    }
                  }
                } catch {
                  for (const { key, pending } of received) {
                    completePending(incoming, key, pending);
                  }
                  received.length = 0;
                }

                const requestContext =
                  received.length === 1 ? received[0]?.pending.handle.context : undefined;
                try {
                  return completeAsyncFailure(
                    requestContext === undefined ? invoke() : withContext(requestContext, invoke),
                  );
                } catch (error) {
                  if (!invoked) {
                    try {
                      return completeAsyncFailure(invoke());
                    } catch (handlerError) {
                      completeFailure(handlerError);
                      throw handlerError;
                    }
                  }
                  completeFailure(error);
                  throw error;
                }
              } finally {
                if (object !== undefined) dispatching.delete(object);
              }
            };
    },
  });

  Object.defineProperty(target, "onclose", {
    configurable: true,
    enumerable: true,
    get: () => onclose,
    set: (handler: (() => void) | undefined) => {
      onclose = wrapClose(handler);
    },
  });

  if (onmessage !== undefined) target.onmessage = onmessage;
  return transport;
}
