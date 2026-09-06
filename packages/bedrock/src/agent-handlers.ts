import type { SpanFields } from "@telemetry-dev/sdk";

import {
  arrayValue,
  awsMetadataFields,
  basenameArn,
  bytesToString,
  isRecord,
  mergeFields,
  modeledStreamError,
  numberValue,
  omitUndefined,
  PROVIDER,
  stringValue,
  type BedrockInstrumentationOptions,
  type JsonRecord,
  type JsonValue,
  type StreamState,
} from "./internal.ts";

export function agentInputMessages<T>(inputText: T): JsonValue {
  const text = stringValue(inputText);
  return text ? [{ role: "user", parts: [{ type: "text", content: text }] }] : undefined;
}

export function agentRequestFields(input: JsonRecord): SpanFields {
  return omitUndefined({
    provider: PROVIDER,
    input: agentInputMessages(input.inputText),
    agentId: stringValue(input.agentId),
    metadata: omitUndefined({
      bedrock_agent_alias_id: stringValue(input.agentAliasId),
      bedrock_session_id: stringValue(input.sessionId),
      bedrock_memory_id: stringValue(input.memoryId),
    }),
  });
}

export function inlineAgentRequestFields(input: JsonRecord): SpanFields {
  return omitUndefined({
    input: agentInputMessages(input.inputText),
    model: stringValue(input.foundationModel),
    provider: PROVIDER,
    agentName: stringValue(input.agentName),
    metadata: omitUndefined({ bedrock_session_id: stringValue(input.sessionId) }),
  });
}

export class AgentStreamState implements StreamState {
  private text = "";
  private usage: NonNullable<SpanFields["usage"]> = {};
  private traceEventCount = 0;
  private failureReason: string | undefined;
  private guardrailAction: string | undefined;
  private returnControl: unknown;
  private readonly traces: unknown[] = [];
  private error: Error | undefined;
  private errorFields: SpanFields | undefined;
  private citationCount = 0;
  private readonly outputFiles: unknown[] = [];

  constructor(private readonly options: BedrockInstrumentationOptions) {}

  feed<T>(event: T): void {
    const raw: unknown = event;
    if (!isRecord(raw)) return;
    const streamError = modeledStreamError(raw);
    if (streamError) {
      this.error = streamError.error;
      this.errorFields = streamError.fields;
      return;
    }
    const chunk = isRecord(raw.chunk) ? raw.chunk : undefined;
    const bytes = chunk?.bytes;
    if (bytes !== undefined) this.text += bytesToString(bytes) ?? "";
    const attribution = isRecord(chunk?.attribution) ? chunk.attribution : undefined;
    const citations = arrayValue<unknown>(attribution?.citations);
    if (citations) this.citationCount += citations.length;

    const files = isRecord(raw.files) ? arrayValue<JsonValue>(raw.files.files) : undefined;
    if (files) this.outputFiles.push(...files.map(sanitizeOutputFile));

    const returnControl = isRecord(raw.returnControl) ? raw.returnControl : undefined;
    if (returnControl) this.returnControl = returnControl;

    const traceEvent = isRecord(raw.trace) ? raw.trace : undefined;
    if (traceEvent) {
      this.traceEventCount += 1;
      if (this.options.captureAgentTrace) this.traces.push(traceEvent);
      collectTrace(traceEvent, (usage) => this.addUsage(usage));
      const failure = findKey(traceEvent, "failureTrace");
      if (isRecord(failure))
        this.failureReason = stringValue(failure.failureReason) ?? this.failureReason;
      const guardrail = findKey(traceEvent, "guardrailTrace");
      if (isRecord(guardrail))
        this.guardrailAction = stringValue(guardrail.action) ?? this.guardrailAction;
    }
  }

  finish(): SpanFields {
    const metadata = omitUndefined({
      trace_event_count: this.traceEventCount || undefined,
      failure_reason: this.failureReason,
      guardrail_action: this.guardrailAction,
      return_control: this.returnControl ? true : undefined,
      citation_count: this.citationCount || undefined,
      output_files: this.outputFiles.length > 0 ? this.outputFiles : undefined,
      agent_trace:
        this.options.captureAgentTrace && this.traces.length > 0 ? this.traces : undefined,
    });
    return mergeFields(
      omitUndefined({
        output: this.returnControl
          ? { returnControl: this.returnControl }
          : this.text
            ? [{ role: "assistant", parts: [{ type: "text", content: this.text }] }]
            : undefined,
        usage: Object.keys(this.usage).length > 0 ? this.usage : undefined,
        metadata,
        error: this.error,
      }),
      this.errorFields,
    );
  }
  private addUsage<T>(value: T): void {
    const raw: unknown = value;
    if (!isRecord(raw)) return;
    const inputTokens = numberValue(raw.inputTokens);
    const outputTokens = numberValue(raw.outputTokens);
    const totalTokens = numberValue(raw.totalTokens);
    if (inputTokens !== undefined)
      this.usage.inputTokens = (this.usage.inputTokens ?? 0) + inputTokens;
    if (outputTokens !== undefined)
      this.usage.outputTokens = (this.usage.outputTokens ?? 0) + outputTokens;
    if (totalTokens !== undefined)
      this.usage.totalTokens = (this.usage.totalTokens ?? 0) + totalTokens;
    else if (inputTokens !== undefined || outputTokens !== undefined) {
      this.usage.totalTokens = (this.usage.inputTokens ?? 0) + (this.usage.outputTokens ?? 0);
    }
  }
}

function sanitizeOutputFile<T>(value: T): JsonValue {
  const raw: unknown = value;
  if (!isRecord(raw)) return undefined;
  return omitUndefined({
    name: stringValue(raw.name),
    type: stringValue(raw.type),
  });
}

export function retrieveRequestFields(input: JsonRecord): SpanFields {
  return omitUndefined({
    provider: PROVIDER,
    input: isRecord(input.retrievalQuery) ? input.retrievalQuery : undefined,
    metadata: { knowledge_base_id: stringValue(input.knowledgeBaseId) },
  });
}

export function retrieveResponseFields<T>(output: T): SpanFields {
  const raw: unknown = output;
  const result = isRecord(raw) ? raw : {};
  const retrievalResults = arrayValue<unknown>(result.retrievalResults);
  return mergeFields(awsMetadataFields(result.$metadata), {
    output: retrievalResults,
    metadata: omitUndefined({
      citation_count: retrievalResults?.length,
      guardrail_action: stringValue(result.guardrailAction),
    }),
  });
}

export function ragModel(input: JsonRecord): string | undefined {
  const config = isRecord(input.retrieveAndGenerateConfiguration)
    ? input.retrieveAndGenerateConfiguration
    : undefined;
  const kb = isRecord(config?.knowledgeBaseConfiguration)
    ? config?.knowledgeBaseConfiguration
    : undefined;
  const external = isRecord(config?.externalSourcesConfiguration)
    ? config?.externalSourcesConfiguration
    : undefined;
  return stringValue(kb?.modelArn) ?? stringValue(external?.modelArn);
}

export function ragRequestFields(input: JsonRecord): SpanFields {
  const text = isRecord(input.input) ? stringValue(input.input.text) : undefined;
  return omitUndefined({
    provider: PROVIDER,
    model: ragModel(input),
    input: text ? [{ role: "user", parts: [{ type: "text", content: text }] }] : undefined,
    metadata: omitUndefined({ bedrock_session_id: stringValue(input.sessionId) }),
  });
}

export function ragResponseFields<T>(output: T): SpanFields {
  const raw: unknown = output;
  const result = isRecord(raw) ? raw : {};
  const text = isRecord(result.output) ? stringValue(result.output.text) : undefined;
  return mergeFields(awsMetadataFields(result.$metadata), {
    output: text ? [{ role: "assistant", parts: [{ type: "text", content: text }] }] : undefined,
    metadata: omitUndefined({
      citation_count: arrayValue<unknown>(result.citations)?.length,
      guardrail_action: stringValue(result.guardrailAction),
      bedrock_session_id: stringValue(result.sessionId),
    }),
  });
}

export class RagStreamState implements StreamState {
  private text = "";
  private citationCount = 0;
  private guardrailAction: string | undefined;
  private error: Error | undefined;
  private errorFields: SpanFields | undefined;

  feed<T>(event: T): void {
    const raw: unknown = event;
    if (!isRecord(raw)) return;
    const streamError = modeledStreamError(raw);
    if (streamError) {
      this.error = streamError.error;
      this.errorFields = streamError.fields;
      return;
    }
    const output = isRecord(raw.output) ? raw.output : undefined;
    this.text += stringValue(output?.text) ?? "";
    if (raw.citation) this.citationCount += 1;
    const guardrail = isRecord(raw.guardrail) ? raw.guardrail : undefined;
    this.guardrailAction = stringValue(guardrail?.action) ?? this.guardrailAction;
  }

  finish(): SpanFields {
    return mergeFields(
      omitUndefined({
        output: this.text
          ? [{ role: "assistant", parts: [{ type: "text", content: this.text }] }]
          : undefined,
        metadata: omitUndefined({
          citation_count: this.citationCount || undefined,
          guardrail_action: this.guardrailAction,
        }),
        error: this.error,
      }),
      this.errorFields,
    );
  }
}

export function flowRequestFields(input: JsonRecord): SpanFields {
  return omitUndefined({
    provider: PROVIDER,
    input: input.inputs,
    agentId: stringValue(input.flowIdentifier),
  });
}

export class FlowStreamState implements StreamState {
  private outputs: unknown[] = [];
  private completionReason: string | undefined;
  private error: Error | undefined;
  private errorFields: SpanFields | undefined;

  feed<T>(event: T): void {
    const raw: unknown = event;
    if (!isRecord(raw)) return;
    const streamError = modeledStreamError(raw);
    if (streamError) {
      this.error = streamError.error;
      this.errorFields = streamError.fields;
      return;
    }
    const output = isRecord(raw.flowOutputEvent) ? raw.flowOutputEvent : undefined;
    if (output) this.outputs.push(output.content);
    const inputRequest = isRecord(raw.flowMultiTurnInputRequestEvent)
      ? raw.flowMultiTurnInputRequestEvent
      : undefined;
    if (inputRequest) this.outputs.push(inputRequest.content);
    const completion = isRecord(raw.flowCompletionEvent) ? raw.flowCompletionEvent : undefined;
    this.completionReason = stringValue(completion?.completionReason) ?? this.completionReason;
  }

  finish(): SpanFields {
    return mergeFields(
      omitUndefined({
        output: this.outputs.length > 0 ? this.outputs : undefined,
        finishReason: this.completionReason,
        error: this.error,
      }),
      this.errorFields,
    );
  }
}

export function flowName(input: JsonRecord): string {
  return `invoke_flow ${stringValue(input.flowIdentifier) ?? "unknown"}`;
}

export function ragName(input: JsonRecord): string {
  return `retrieve_and_generate ${basenameArn(ragModel(input)) ?? "knowledge-base"}`;
}

function collectTrace<T>(value: T, onUsage: <TUsage>(usage: TUsage) => void): void {
  const raw: unknown = value;
  if (!isRecord(raw)) return;
  const output = isRecord(raw.modelInvocationOutput) ? raw.modelInvocationOutput : undefined;
  if (output) {
    const metadata = isRecord(output.metadata) ? output.metadata : undefined;
    onUsage(metadata?.usage);
  }
  for (const child of Object.values(raw)) {
    if (isRecord(child)) collectTrace(child, onUsage);
    else if (Array.isArray(child)) child.forEach((item) => collectTrace(item, onUsage));
  }
}

function findKey<T>(value: T, key: string): JsonValue {
  const raw: unknown = value;
  if (!isRecord(raw)) return undefined;
  if (raw[key] !== undefined) return raw[key];
  for (const child of Object.values(raw)) {
    const found = findKey(child, key);
    if (found !== undefined) return found;
  }
  return undefined;
}
