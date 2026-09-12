import type { JsonRecord, JsonValue, OpHandler } from "./internal.ts";
import {
  AgentStreamState,
  FlowStreamState,
  RagStreamState,
  agentRequestFields,
  flowName,
  flowRequestFields,
  inlineAgentRequestFields,
  ragName,
  ragRequestFields,
  ragResponseFields,
  retrieveRequestFields,
  retrieveResponseFields,
} from "./agent-handlers.ts";
import { ConverseStreamState, converseRequestFields, converseResponseFields } from "./converse.ts";
import {
  invokeModelRequestFields,
  invokeModelResponseFields,
  invokeModelSpanName,
  InvokeModelStreamState,
  isEmbeddingModel,
} from "./invoke-model.ts";
import {
  awsMetadataFields,
  mergeFields,
  omitUndefined,
  PROVIDER,
  stringValue,
  wrapAsyncIterable,
  endSpan,
  type BedrockInstrumentationOptions,
} from "./internal.ts";

export const RUNTIME_HANDLERS = {
  ConverseCommand: {
    spanName: (input) => `chat ${stringValue(input.modelId) ?? "unknown"}`,
    spanType: () => "generation",
    requestFields: converseRequestFields,
    onResult(result, span) {
      endSpan(span, converseResponseFields(result));

      return result;
    },
  },
  ConverseStreamCommand: {
    spanName: (input) => `chat ${stringValue(input.modelId) ?? "unknown"}`,
    spanType: () => "generation",
    requestFields: converseRequestFields,
    onResult(result, span, t0) {
      const response = result as { stream?: unknown; $metadata?: unknown };

      if (!response.stream) {
        endSpan(span, awsMetadataFields(response.$metadata));

        return result;
      }

      return {
        ...response,
        stream: wrapAsyncIterable(
          response.stream,
          new ConverseStreamState(),
          span,
          t0,
          awsMetadataFields(response.$metadata),
        ),
      };
    },
  },
  InvokeModelCommand: {
    spanName: invokeModelSpanName,
    spanType: (input) => (isEmbeddingModel(input.modelId) ? "embedding" : "generation"),
    requestFields: invokeModelRequestFields,
    onResult(result, span, _t0, input) {
      endSpan(span, invokeModelResponseFields(input, result));

      return result;
    },
  },
  InvokeModelWithResponseStreamCommand: {
    spanName: invokeModelSpanName,
    spanType: (input) => (isEmbeddingModel(input.modelId) ? "embedding" : "generation"),
    requestFields: invokeModelRequestFields,
    onResult(result, span, t0) {
      const response = result as { body?: unknown; $metadata?: unknown };

      if (!response.body) {
        endSpan(span, awsMetadataFields(response.$metadata));

        return result;
      }

      return {
        ...response,
        body: wrapAsyncIterable(
          response.body,
          new InvokeModelStreamState(),
          span,
          t0,
          awsMetadataFields(response.$metadata),
        ),
      };
    },
  },
  ApplyGuardrailCommand: {
    spanName: (input) => `apply_guardrail ${stringValue(input.guardrailIdentifier) ?? "unknown"}`,
    spanType: () => "span",
    requestFields: (input) => ({
      provider: PROVIDER,
      input: omitUndefined({
        source: input.source,
        content: input.content,
        guardrailIdentifier: input.guardrailIdentifier,
        guardrailVersion: input.guardrailVersion,
      }),
    }),
    onResult(result, span) {
      const response = result as JsonRecord;
      endSpan(
        span,
        mergeFields(awsMetadataFields(response.$metadata), {
          output: omitUndefined({ outputs: response.outputs, assessments: response.assessments }),
          metadata: omitUndefined({
            guardrail_action: stringValue(response.action),
            guardrail_action_reason: stringValue(response.actionReason),
          }),
        }),
      );

      return result;
    },
  },
} satisfies Record<string, OpHandler>;

export const AGENT_HANDLERS = {
  InvokeAgentCommand: {
    spanName: (input) => `invoke_agent ${stringValue(input.agentId) ?? "unknown"}`,
    spanType: () => "agent",
    requestFields: agentRequestFields,
    onResult: streamAgentResult("completion"),
  },
  InvokeInlineAgentCommand: {
    spanName: (input) => `invoke_agent ${stringValue(input.agentName) ?? "inline-agent"}`,
    spanType: () => "agent",
    requestFields: inlineAgentRequestFields,
    onResult: streamAgentResult("completion"),
  },
  RetrieveCommand: {
    spanName: (input) => `retrieve ${stringValue(input.knowledgeBaseId) ?? "unknown"}`,
    spanType: () => "span",
    requestFields: retrieveRequestFields,
    onResult(result, span) {
      endSpan(span, retrieveResponseFields(result));

      return result;
    },
  },
  RetrieveAndGenerateCommand: {
    spanName: ragName,
    spanType: () => "generation",
    requestFields: ragRequestFields,
    onResult(result, span) {
      endSpan(span, ragResponseFields(result));

      return result;
    },
  },
  RetrieveAndGenerateStreamCommand: {
    spanName: ragName,
    spanType: () => "generation",
    requestFields: ragRequestFields,
    onResult(result, span, t0) {
      const response = result as { stream?: unknown; $metadata?: unknown; sessionId?: string };

      const base = mergeFields(awsMetadataFields(response.$metadata), {
        metadata: omitUndefined({ bedrock_session_id: stringValue(response.sessionId) }),
      });

      if (!response.stream) {
        endSpan(span, base);

        return result;
      }

      return {
        ...response,
        stream: wrapAsyncIterable(response.stream, new RagStreamState(), span, t0, base, false),
      };
    },
  },
  InvokeFlowCommand: {
    spanName: flowName,
    spanType: () => "agent",
    requestFields: flowRequestFields,
    onResult(result, span, t0) {
      const response = result as { responseStream?: unknown; $metadata?: unknown };

      if (!response.responseStream) {
        endSpan(span, awsMetadataFields(response.$metadata));

        return result;
      }

      return {
        ...response,
        responseStream: wrapAsyncIterable(
          response.responseStream,
          new FlowStreamState(),
          span,
          t0,
          awsMetadataFields(response.$metadata),
          false,
        ),
      };
    },
  },
} satisfies Record<string, OpHandler>;

function streamAgentResult(key: "completion") {
  return (
    result: JsonValue,
    span: Parameters<OpHandler["onResult"]>[1],
    t0: number,
    _input: JsonRecord,
    options: BedrockInstrumentationOptions,
  ) => {
    const response = result as JsonRecord & {
      completion?: JsonValue;
      $metadata?: JsonValue;
      sessionId?: string;
      memoryId?: string;
    };

    const base = mergeFields(awsMetadataFields(response.$metadata), {
      metadata: omitUndefined({
        bedrock_session_id: stringValue(response.sessionId),
        bedrock_memory_id: stringValue(response.memoryId),
      }),
    });

    if (!response[key]) {
      endSpan(span, base);

      return result;
    }

    return {
      ...response,
      [key]: wrapAsyncIterable(response[key], new AgentStreamState(options), span, t0, base, false),
    };
  };
}
