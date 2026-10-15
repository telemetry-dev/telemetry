import { boundedCapture, captureEnabled, type SpanFields } from "@telemetry-dev/sdk";

import {
  arrayValue,
  awsMetadataFields,
  basenameArn,
  bytesToString,
  captureMaskedBeforeStart,
  isRecord,
  mergeFields,
  modeledStreamError,
  numberValue,
  omitUndefined,
  PROVIDER,
  stringValue,
  StreamRetention,
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
  readonly retention = new StreamRetention();
  private text = "";
  // Trace and output-file metadata share the budget but do not make the text incomplete.
  private textRejected = false;
  private usage: NonNullable<SpanFields["usage"]> = {};
  private traceEventCount = 0;
  private failed = false;
  private failureCode: number | undefined;
  private failureReason: string | undefined;
  private guardrailAction: string | undefined;
  private returnControl: unknown;
  private sawReturnControl = false;
  private returnControlRejected = false;
  private readonly traces: unknown[] = [];
  private error: Error | undefined;
  private errorFields: SpanFields | undefined;
  private citationCount = 0;
  private readonly outputFiles: unknown[] = [];
  // Metadata bypasses the capture flags and the mask, so trace and file content is filtered here,
  // and a mask that cannot see it withholds it.
  private readonly contentCapture: AgentTraceCapture = {
    input: captureEnabled("input") && !captureMaskedBeforeStart(),
    output: captureEnabled("output") && !captureMaskedBeforeStart(),
  };

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

    const piece = bytes === undefined ? undefined : bytesToString(bytes);

    if (piece) {
      if (this.retention.accept(piece)) this.text += piece;
      else this.textRejected = true;
    }

    const attribution = isRecord(chunk?.attribution) ? chunk.attribution : undefined;
    const citations = arrayValue<unknown>(attribution?.citations);

    if (citations) this.citationCount += citations.length;

    const files = isRecord(raw.files) ? arrayValue<JsonValue>(raw.files.files) : undefined;

    const sanitizedFiles = files?.map((file) =>
      sanitizeOutputFile(file, this.contentCapture.output),
    );

    if (sanitizedFiles && this.retention.accept(sanitizedFiles)) {
      this.outputFiles.push(...sanitizedFiles);
    }

    const returnControl = isRecord(raw.returnControl) ? raw.returnControl : undefined;

    if (returnControl) {
      this.sawReturnControl = true;

      // Return control replaces the text output, so it gets its own fresh bound and survives
      // text that already spent the shared budget; one that alone does not fit is dropped and
      // the capture is marked incomplete.
      if (new StreamRetention().accept(returnControl)) {
        this.returnControl = returnControl;
      } else {
        this.returnControl = undefined;
        this.returnControlRejected = true;
        this.retention.truncated = true;
      }
    }

    const traceEvent = isRecord(raw.trace) ? raw.trace : undefined;

    if (traceEvent) {
      this.traceEventCount += 1;

      if (this.options.captureAgentTrace) {
        const retained = projectTrace(traceEvent, TRACE_PART, this.contentCapture);

        if (this.retention.accept(retained)) this.traces.push(retained);
      }

      // Summaries are read only where the API puts them, so content and unknown branches can't
      // feed them.
      const trace = isRecord(traceEvent.trace) ? traceEvent.trace : {};

      for (const step of MODEL_INVOCATION_STEPS) {
        const member = isRecord(trace[step]) ? trace[step] : undefined;
        const output = isRecord(member?.modelInvocationOutput) ? member.modelInvocationOutput : {};
        this.addUsage(isRecord(output.metadata) ? output.metadata.usage : undefined);
      }

      const failure = trace.failureTrace;

      if (isRecord(failure)) {
        this.failed = true;
        this.failureCode = numberValue(failure.failureCode) ?? this.failureCode;

        // The reason is free-form text that can quote the prompt or a tool result.
        if (this.contentCapture.input && this.contentCapture.output) {
          this.failureReason = stringValue(failure.failureReason) ?? this.failureReason;
        }
      }

      const guardrail = trace.guardrailTrace;

      if (isRecord(guardrail))
        this.guardrailAction = stringValue(guardrail.action) ?? this.guardrailAction;
    }
  }

  outputTruncated(): boolean {
    // Return control replaces the text output and is retained whole when it fits its bound.
    if (this.returnControl !== undefined) return false;

    return this.textRejected || this.returnControlRejected;
  }

  finish(): SpanFields {
    const metadata = omitUndefined({
      trace_event_count: this.traceEventCount || undefined,
      failure: this.failed ? true : undefined,
      failure_code: this.failureCode,
      failure_reason: this.failureReason,
      guardrail_action: this.guardrailAction,
      return_control: this.sawReturnControl ? true : undefined,
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

function sanitizeOutputFile<T>(value: T, keepName: boolean): JsonValue {
  const raw: unknown = value;

  if (!isRecord(raw)) return undefined;

  return omitUndefined({
    name: keepName ? stringValue(raw.name) : undefined,
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

function rerankConfiguration(input: JsonRecord) {
  const configuration = isRecord(input.rerankingConfiguration)
    ? input.rerankingConfiguration
    : undefined;

  const bedrock = isRecord(configuration?.bedrockRerankingConfiguration)
    ? configuration.bedrockRerankingConfiguration
    : undefined;

  return {
    bedrock,
    model: isRecord(bedrock?.modelConfiguration) ? bedrock.modelConfiguration : undefined,
  };
}

export function rerankModel(input: JsonRecord): string | undefined {
  return stringValue(rerankConfiguration(input).model?.modelArn);
}

export function rerankName(input: JsonRecord): string {
  const model = rerankModel(input);

  return `rerank ${model?.split("/").at(-1) ?? "unknown"}`;
}

export function rerankRequestFields(input: JsonRecord): SpanFields {
  const { bedrock, model } = rerankConfiguration(input);

  const capture = captureEnabled("input")
    ? boundedCapture(
        omitUndefined({
          queries: input.queries,
          sources: input.sources,
          additionalModelRequestFields: model?.additionalModelRequestFields,
          nextToken: input.nextToken,
        }),
      )
    : undefined;

  return omitUndefined({
    provider: PROVIDER,
    model: stringValue(model?.modelArn),
    input: capture?.value,
    attributes: {
      "gen_ai.operation.name": "rerank",
      ...(capture?.truncated ? { "telemetry.dev.capture.truncated": true } : undefined),
    },
    metadata: omitUndefined({
      requested_result_count: numberValue(bedrock?.numberOfResults),
    }),
  });
}

export function rerankResponseFields<T>(output: T): SpanFields {
  const raw: unknown = output;
  const result = isRecord(raw) ? raw : {};
  const results = arrayValue<unknown>(result.results);
  const capture = captureEnabled("output") ? boundedCapture(results) : undefined;

  return mergeFields(awsMetadataFields(result.$metadata), {
    output: capture?.value,
    attributes: capture?.truncated ? { "telemetry.dev.capture.truncated": true } : undefined,
    metadata: omitUndefined({
      result_count: results?.length,
      has_next_token: stringValue(result.nextToken) ? true : undefined,
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
  readonly retention = new StreamRetention();
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
    const text = stringValue(output?.text);

    if (text && this.retention.accept(text)) this.text += text;

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
  readonly retention = new StreamRetention();
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

    if (output && this.retention.accept(output.content)) this.outputs.push(output.content);

    const inputRequest = isRecord(raw.flowMultiTurnInputRequestEvent)
      ? raw.flowMultiTurnInputRequestEvent
      : undefined;

    if (inputRequest && this.retention.accept(inputRequest.content)) {
      this.outputs.push(inputRequest.content);
    }

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

type AgentTraceCapture = Record<"input" | "output", boolean>;

/** Content kept as received when every listed capture is on; otherwise reduced to `withheld`. */
class TraceContent {
  constructor(
    readonly needs: ReadonlyArray<keyof AgentTraceCapture>,
    readonly withheld?: TraceShape,
  ) {}
}

const SCALAR = "scalar";

// A record shape lists the only fields kept; a one-item array shape is a list of that shape.
type TraceShape =
  | typeof SCALAR
  | TraceContent
  | readonly [TraceShape]
  | { readonly [field: string]: TraceShape };

const TRACE_METADATA: TraceShape = {
  usage: { inputTokens: SCALAR, outputTokens: SCALAR },
  startTime: SCALAR,
  endTime: SCALAR,
  totalTimeMs: SCALAR,
  operationTotalTimeMs: SCALAR,
  clientRequestId: SCALAR,
};

// What the agent sends to a model, tool, or guardrail is input content; what comes back is output.
const MODEL_INVOCATION_INPUT = new TraceContent(["input"], {
  traceId: SCALAR,
  type: SCALAR,
  foundationModel: SCALAR,
});

const INVOCATION_INPUT = new TraceContent(["input"], { traceId: SCALAR, invocationType: SCALAR });

const MODEL_INVOCATION_OUTPUT = new TraceContent(["output"], {
  traceId: SCALAR,
  metadata: TRACE_METADATA,
});

const OBSERVATION = new TraceContent(["output"], { traceId: SCALAR, type: SCALAR });

const MODEL_INVOCATION_STEPS = [
  "preProcessingTrace",
  "orchestrationTrace",
  "postProcessingTrace",
  "routingClassifierTrace",
];

const PROCESSING_TRACE: TraceShape = {
  modelInvocationInput: MODEL_INVOCATION_INPUT,
  modelInvocationOutput: MODEL_INVOCATION_OUTPUT,
};

// Mirrors TracePart and the Trace union in the Bedrock Agent Runtime API; anything else, including
// the AWS SDK's `$unknown` union member, is dropped because nothing says what it holds.
const TRACE_PART: TraceShape = {
  agentId: SCALAR,
  agentAliasId: SCALAR,
  agentVersion: SCALAR,
  sessionId: SCALAR,
  collaboratorName: SCALAR,
  eventTime: SCALAR,
  callerChain: [{ agentAliasArn: SCALAR }],
  trace: {
    preProcessingTrace: PROCESSING_TRACE,
    postProcessingTrace: PROCESSING_TRACE,
    orchestrationTrace: {
      modelInvocationInput: MODEL_INVOCATION_INPUT,
      invocationInput: INVOCATION_INPUT,
      modelInvocationOutput: MODEL_INVOCATION_OUTPUT,
      rationale: new TraceContent(["output"], { traceId: SCALAR }),
      observation: OBSERVATION,
    },
    routingClassifierTrace: {
      modelInvocationInput: MODEL_INVOCATION_INPUT,
      invocationInput: INVOCATION_INPUT,
      modelInvocationOutput: MODEL_INVOCATION_OUTPUT,
      observation: OBSERVATION,
    },
    customOrchestrationTrace: { traceId: SCALAR, event: new TraceContent(["output"]) },
    guardrailTrace: {
      traceId: SCALAR,
      action: SCALAR,
      metadata: TRACE_METADATA,
      inputAssessments: new TraceContent(["input"]),
      outputAssessments: new TraceContent(["output"]),
    },
    failureTrace: {
      traceId: SCALAR,
      failureCode: SCALAR,
      metadata: TRACE_METADATA,
      // Free-form text that can quote either the prompt or a tool result.
      failureReason: new TraceContent(["input", "output"]),
    },
  },
};

function projectTrace(value: JsonValue, shape: TraceShape, capture: AgentTraceCapture): JsonValue {
  if (shape instanceof TraceContent) {
    if (shape.needs.every((kind) => capture[kind])) return value ?? undefined;

    return shape.withheld === undefined ? undefined : projectTrace(value, shape.withheld, capture);
  }

  if (shape === SCALAR) {
    const scalar = ["string", "number", "boolean"].includes(typeof value) || value instanceof Date;

    return scalar ? value : undefined;
  }

  if (Array.isArray(shape)) {
    if (!Array.isArray(value)) return undefined;

    return value
      .map((item) => projectTrace(item, shape[0], capture))
      .filter((item) => item !== undefined);
  }

  if (!isRecord(value)) return undefined;

  return omitUndefined(
    Object.fromEntries(
      Object.entries(shape).map(([field, child]) => [
        field,
        projectTrace(value[field], child, capture),
      ]),
    ),
  );
}
