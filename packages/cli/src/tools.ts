/**
 * Results of the telemetry.dev MCP tools, as the server documents them. `--json` prints
 * these as returned, so their fields are the CLI's machine-readable contract too.
 */

export type ProjectList = {
  organization_id: string;
  projects: { id: string; name: string; retention_days: number; created_at: string }[];
};

export type TraceSummary = {
  id: string;
  name: string;
  status: "success" | "error";
  error_type: string | null;
  started_at: string;
  duration_ms: number | null;
  environment: string | null;
  service_name: string | null;
  session_id: string | null;
  user_id: string | null;
  models: string[];
  span_count: number;
  error_span_count: number;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
};

export type TracePage = { traces: TraceSummary[]; next_cursor: string | null };

export type SessionSummary = {
  id: string;
  last_active_at: string;
  user_id: string | null;
  agent_name: string | null;
  models: string[];
  trace_count: number;
  error_trace_count: number;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
};

export type SessionPage = { sessions: SessionSummary[]; next_cursor: string | null };

export type Span = {
  id: string;
  parent_span_id: string | null;
  name: string;
  operation: string | null;
  kind: string | null;
  agent_name: string | null;
  tool_name: string | null;
  provider: string | null;
  model: string | null;
  status: string;
  error_type: string | null;
  error: string | null;
  finish_reason: string | null;
  started_at: string;
  duration_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
};

export type TraceLog = {
  timestamp: string;
  level: string;
  span_id: string | null;
  message: string;
};

export type TraceDetailPage = {
  trace: Omit<TraceSummary, "models" | "span_count" | "error_span_count"> & {
    framework: string | null;
    /** The trace's dashboard page. Servers before this field omit it. */
    url?: string;
  };
  span_count: number;
  log_count: number;
  spans: Span[];
  logs: TraceLog[];
  next_cursor: string | null;
};

export type PayloadPage = {
  field: "input" | "output" | "error";
  total_characters: number;
  stored_text_truncated: boolean;
  offset: number;
  next_offset: number | null;
  text: string;
};

export type LogList = {
  range_days: number;
  counts: {
    debugCount: number;
    infoCount: number;
    warnCount: number;
    errorCount: number;
    totalCount: number;
    traceCount: number;
  };
  has_more: boolean;
  logs: {
    timestamp: string;
    level: string;
    message: string;
    service_name: string | null;
    environment: string | null;
    trace_id: string | null;
    span_id: string | null;
  }[];
};

export type UsageTotals = {
  totalRequests: number;
  errorCount: number;
  errorRate: number;
  p95DurationMs: number;
  avgDurationMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCostUsd: number;
  activeUsers: number;
};

export type UsageSummary = {
  range_days: number;
  since: string;
  data_incomplete: boolean;
  current: UsageTotals;
  previous: UsageTotals | null;
  group_by: "model" | "provider" | "user" | "operation";
  breakdown: {
    category: string | null;
    errorCount: number;
    errorRate: number;
    p95DurationMs: number;
    totalCostUsd: number | null;
    totalTokens: number | null;
    totalRequests: number;
  }[];
  breakdown_limit: number;
};
