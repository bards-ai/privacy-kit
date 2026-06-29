import type { ReactNode } from "react";

import { ClearLog } from "@/components/clear-log";
import { PolicySelect, SaveTextsSelect, ThresholdInput } from "@/components/config-controls";
import { ExportMenu } from "@/components/interactions-controls";
import { Card, CardContent, CardHeader, CardTitle, ConnectionError, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { AppConfig, Summary } from "@/lib/types";

export const dynamic = "force-dynamic";

const EMPTY_SUMMARY: Summary = {
  interactions: 0,
  entities_total: 0,
  entities_by_type: {},
  by_source: {},
  by_wire_format: {},
  by_policy: {},
  by_model: {},
  tokens: { input: 0, output: 0 },
  timeseries: [],
};

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b py-2 last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="break-all text-right text-sm font-medium">{children}</span>
    </div>
  );
}

export default async function SettingsPage() {
  const { data: config, error } = await apiGetOr<AppConfig | null>("/config", null);
  const { data: summary } = await apiGetOr<Summary>("/summary", EMPTY_SUMMARY);

  if (error || !config) {
    return (
      <>
        <PageHeader title="Settings" />
        <ConnectionError message={error ?? "no configuration returned"} />
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Settings"
        description={`privacy-kit v${config.version} · current gateway configuration`}
        actions={<ExportMenu />}
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Policy &amp; detection</CardTitle>
          </CardHeader>
          <CardContent>
            <Row label="Policy">
              <PolicySelect policy={config.policy} />
            </Row>
            <Row label="Save texts">
              <SaveTextsSelect value={config.save_texts} />
            </Row>
            <Row label="Threshold">
              <ThresholdInput value={config.threshold} />
            </Row>
            <Row label="Expose plaintext">{config.expose_plaintext ? "yes" : "no (redacted)"}</Row>
            <Row label="Model">{config.model_id}</Row>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Storage</CardTitle>
          </CardHeader>
          <CardContent>
            <Row label="Database">{config.db_path}</Row>
            <Row label="Interactions">{formatNumber(summary.interactions)}</Row>
            <Row label="PII entities">{formatNumber(summary.entities_total)}</Row>
            <Row label="Bind">{`${config.host}:${config.port}`}</Row>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Upstreams</CardTitle>
          </CardHeader>
          <CardContent>
            <Row label="Anthropic">{config.anthropic_upstream}</Row>
            <Row label="OpenAI">{config.openai_upstream}</Row>
            <Row label="ChatGPT (Codex)">{config.chatgpt_upstream}</Row>
            <Row label="OTLP downstream">{config.otel_downstream ?? "—"}</Row>
          </CardContent>
        </Card>

        <Card className="border-red-500/30">
          <CardHeader>
            <CardTitle className="text-red-500">Danger zone</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-3 text-sm text-muted-foreground">
              Permanently delete all recorded interactions, detections, and saved text. Export first
              if you want a copy.
            </p>
            <ClearLog />
          </CardContent>
        </Card>
      </div>

      <p className="mt-6 text-xs text-muted-foreground">
        These values come from the gateway&apos;s environment (PII_* variables / .env). Policy, save
        texts, and threshold can be changed here at runtime — they apply to subsequent traffic but
        reset to their PII_* values on restart. The remaining values require editing the environment
        and restarting the gateway.
      </p>
    </>
  );
}
