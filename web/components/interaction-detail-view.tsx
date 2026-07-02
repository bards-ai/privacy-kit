import type { ReactNode } from "react";

import { EntityChips } from "@/components/entity-chips";
import { InteractionDetailBody } from "@/components/interaction-detail-body";
import { PolicyBadge } from "@/components/policy-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui";
import { formatDateTime } from "@/lib/format";
import { kindMeta } from "@/lib/kind";
import type { InteractionDetail } from "@/lib/types";

// Metadata card + detections/text body for one interaction (turn). Shared by
// the standalone /interactions/[id] page and the conversation-scoped nested
// route so both render identically.
export function InteractionDetailView({ data }: { data: InteractionDetail }) {
  const it = data.interaction;
  const meta: [string, ReactNode][] = [
    ["ID", `#${it.id}`],
    ["When", formatDateTime(it.created_at)],
    ["Source", it.source],
    ["Kind", kindMeta(it.kind).label],
    ["Policy", <PolicyBadge key="p" policy={it.policy} />],
    ["Wire format", it.wire_format],
    ["Model", it.model],
    ["Language", it.language ?? "—"],
    ["Input tokens", it.input_tokens ?? "—"],
    ["Output tokens", it.output_tokens ?? "—"],
    ["Entities total", it.entity_total],
  ];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-5">
            {meta.map(([k, v]) => (
              <div key={k}>
                <dt className="text-xs uppercase tracking-wide text-muted-foreground">{k}</dt>
                <dd className="mt-1 text-sm font-medium">{v}</dd>
              </div>
            ))}
          </dl>
          <div className="mt-4">
            <span className="text-xs uppercase tracking-wide text-muted-foreground">
              Entity counts
            </span>
            <div className="mt-2">
              <EntityChips counts={it.entity_counts} />
            </div>
          </div>
        </CardContent>
      </Card>

      <InteractionDetailBody
        detections={data.detections}
        texts={data.texts}
        redacted={data.texts_redacted}
      />
    </div>
  );
}
