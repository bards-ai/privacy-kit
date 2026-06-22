import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { DeleteInteraction } from "@/components/delete-interaction";
import { EntityChips } from "@/components/entity-chips";
import { PolicyBadge } from "@/components/policy-badge";
import { Card, CardContent, CardHeader, CardTitle, EmptyState, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import { entityColor } from "@/lib/colors";
import { formatDateTime } from "@/lib/format";
import type { InteractionDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

function BackLink() {
  return (
    <Link
      href="/interactions"
      className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
    >
      <ArrowLeft className="h-4 w-4" /> Back to interactions
    </Link>
  );
}

export default async function InteractionDetailPage({ params }: { params: { id: string } }) {
  const { data } = await apiGetOr<InteractionDetail | null>(`/interactions/${params.id}`, null);

  if (!data) {
    return (
      <>
        <BackLink />
        <EmptyState
          title="Interaction not found"
          description="It may have been deleted, or the gateway is unreachable."
        />
      </>
    );
  }

  const it = data.interaction;
  const meta: [string, ReactNode][] = [
    ["ID", `#${it.id}`],
    ["When", formatDateTime(it.created_at)],
    ["Source", it.source],
    ["Policy", <PolicyBadge key="p" policy={it.policy} />],
    ["Wire format", it.wire_format],
    ["Model", it.model],
    ["Language", it.language ?? "—"],
    ["Input tokens", it.input_tokens ?? "—"],
    ["Output tokens", it.output_tokens ?? "—"],
    ["Entities total", it.entity_total],
  ];

  return (
    <>
      <BackLink />
      <PageHeader
        title={`Interaction #${it.id}`}
        description={`${it.source} · ${formatDateTime(it.created_at)}`}
        actions={<DeleteInteraction id={it.id} />}
      />

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

        <Card>
          <CardHeader>
            <CardTitle>Detections</CardTitle>
          </CardHeader>
          <CardContent>
            {data.detections.length === 0 ? (
              <p className="text-sm text-muted-foreground">No PII detected.</p>
            ) : (
              <table className="w-full max-w-md text-sm">
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <th className="py-2">Entity type</th>
                    <th className="py-2">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {data.detections.map((d) => (
                    <tr key={d.id} className="border-b last:border-0">
                      <td className="py-2">
                        <span className="inline-flex items-center gap-2">
                          <span
                            className="h-2 w-2 rounded-full"
                            style={{ backgroundColor: entityColor(d.entity_type) }}
                          />
                          {d.entity_type}
                        </span>
                      </td>
                      <td className="py-2 tabular-nums">{d.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Saved text — before &amp; after</CardTitle>
          </CardHeader>
          <CardContent>
            {data.texts_redacted ? (
              <p className="mb-3 text-xs text-amber-500">
                Originals are redacted (PII_EXPOSE_PLAINTEXT=false).
              </p>
            ) : null}
            {data.texts.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No text segments saved for this interaction.
              </p>
            ) : (
              <div className="space-y-4">
                {data.texts.map((t) => (
                  <div key={t.id} className="grid gap-3 lg:grid-cols-2">
                    <div>
                      <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                        Original
                      </div>
                      <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm">
                        {t.original ?? <span className="text-muted-foreground">[redacted]</span>}
                      </div>
                    </div>
                    <div>
                      <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                        Anonymized
                      </div>
                      <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 font-mono text-sm">
                        {t.anonymized}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}
