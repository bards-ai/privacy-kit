// Shapes returned by the privacy-kit gateway's /api/v1 endpoints.

export interface Interaction {
  id: number;
  created_at: string;
  source: string;
  wire_format: string;
  kind: string; // call purpose: "main" | "safety" | "helper"
  model: string;
  policy: string;
  language: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  entity_total: number;
  entity_counts: Record<string, number>;
}

export interface InteractionListItem extends Interaction {
  text_count: number;
  detection_types: string[];
}

export interface InteractionList {
  items: InteractionListItem[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface Detection {
  id: number;
  entity_type: string;
  count: number;
}

// Origin of a saved segment: text the human typed, data a local tool/file
// produced, or the agent's response (stored only when the turn had PII).
export type TextCategory = "user" | "tool" | "assistant";

export interface TextSegment {
  id: number;
  seq: number;
  category: TextCategory;
  original: string | null;
  anonymized: string;
}

export interface InteractionDetail {
  interaction: Interaction;
  detections: Detection[];
  texts: TextSegment[];
  texts_redacted: boolean;
}

export interface Summary {
  interactions: number;
  entities_total: number;
  entities_by_type: Record<string, number>;
  by_source: Record<string, number>;
  by_wire_format: Record<string, number>;
  by_policy: Record<string, number>;
  by_model: Record<string, number>;
  tokens: { input: number; output: number };
  timeseries: { date: string; interactions: number; entities: number }[];
}

export interface FilterValues {
  sources: string[];
  wire_formats: string[];
  kinds: string[];
  models: string[];
  policies: string[];
  languages: string[];
  entity_types: string[];
}

export interface TextRow {
  interaction_id: number;
  when: string;
  source: string;
  model: string;
  seq: number;
  category: TextCategory;
  original: string | null;
  anonymized: string;
}

export interface TextsResponse {
  texts: TextRow[];
  redacted: boolean;
}

export interface ConversationListItem {
  conversation_id: string;
  first_seen: string;
  last_seen: string;
  turn_count: number;
  entity_total: number;
  entity_counts: Record<string, number>;
  sources: string[];
  models: string[];
  background_count: number;
  input_tokens: number;
  output_tokens: number;
}

export interface ConversationList {
  items: ConversationListItem[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface ConversationTurn {
  interaction: Interaction;
  detections: Detection[];
  texts: TextSegment[];
}

export interface ConversationDetail {
  conversation_id: string;
  summary: ConversationListItem;
  turns: ConversationTurn[];
  background_count: number;
  texts_redacted: boolean;
}

export type Policy = "monitor" | "pseudonymize";
export type SaveTexts = "anonymized" | "all";

export interface AppConfig {
  version: string;
  policy: Policy;
  save_texts: SaveTexts;
  expose_plaintext: boolean;
  model_id: string;
  threshold: number;
  db_path: string;
  host: string;
  port: number;
  anthropic_upstream: string;
  openai_upstream: string;
  chatgpt_upstream: string;
  otel_downstream: string | null;
}

export interface PreviewResult {
  spans: { start: number; end: number; label: string; score: number }[];
  anonymized: string;
  counts: Record<string, number>;
}

export type ImportSource = "claude-code" | "codex";

export interface ImportPreview {
  sources: Record<ImportSource, { found: number; new: number; imported: number }>;
}

export interface ImportSessionItem {
  source: ImportSource;
  id: string | null;
  title: string | null;
  project: string | null;
  modified_at: string;
  imported: boolean;
}

export interface ImportSessionsPreview {
  total: number;
  titles_redacted: boolean;
  sessions: ImportSessionItem[];
}

export interface ImportRequest {
  sources: ImportSource[];
  since?: string;
  until?: string;
  project?: string;
  dry_run?: boolean;
}

export interface ImportStatus {
  state: "idle" | "running" | "done" | "error";
  sources?: string[];
  dry_run?: boolean;
  since?: string | null;
  until?: string | null;
  project?: string | null;
  found?: number;
  skipped?: number;
  imported?: number;
  failed?: number;
  turns?: number;
  entities?: number;
  current?: string | null;
  error?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}
