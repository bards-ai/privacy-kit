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

export interface TextSegment {
  id: number;
  seq: number;
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
  original: string | null;
  anonymized: string;
}

export interface TextsResponse {
  texts: TextRow[];
  redacted: boolean;
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
