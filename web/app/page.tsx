import { Activity, ArrowRight, Cpu, Database, Hash, ShieldAlert } from "lucide-react";
import Link from "next/link";

import { ActivityAreaChart, EntityBarChart, SourcePie } from "@/components/charts";
import { EntityChips } from "@/components/entity-chips";
import { PolicyBadge } from "@/components/policy-badge";
import { StatCard } from "@/components/stat-card";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  ConnectionError,
  EmptyState,
  PageHeader,
} from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import { CHART_PALETTE } from "@/lib/colors";
import { formatDateTime, formatNumber } from "@/lib/format";
import type { InteractionList, Summary } from "@/lib/types";

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
const EMPTY_LIST: InteractionList = { items: [], page: 1, page_size: 8, total: 0, total_pages: 0 };

export default async function OverviewPage() {
  const { data: summary, error } = await apiGetOr<Summary>("/summary", EMPTY_SUMMARY);
  const { data: recent } = await apiGetOr<InteractionList>(
    "/interactions?page_size=8&sort=created_at&order=desc",
    EMPTY_LIST,
  );

  if (error) {
    return (
      <>
        <PageHeader title="Overview" />
        <ConnectionError message={error} />
      </>
    );
  }

  const entityData = Object.entries(summary.entities_by_type)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
  const sourceData = Object.entries(summary.by_source)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
  const monitor = summary.by_policy["monitor"] ?? 0;

  return (
    <>
      <PageHeader
        title="Overview"
        description="What the on-device PII gateway has seen across all proxied traffic."
      />

      {summary.interactions === 0 ? (
        <EmptyState
          icon={<Activity className="h-8 w-8" />}
          title="No interactions recorded yet"
          description="Route a tool through the gateway (see Settings) and send a prompt — it will show up here. Until then, try the Live preview to see PII detection in action."
        />
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatCard
              label="Interactions"
              value={formatNumber(summary.interactions)}
              icon={<Activity className="h-4 w-4" />}
            />
            <StatCard
              label="PII entities"
              value={formatNumber(summary.entities_total)}
              sub={`${entityData.length} distinct type(s)`}
              icon={<ShieldAlert className="h-4 w-4" />}
            />
            <StatCard
              label="Sources / models"
              value={`${Object.keys(summary.by_source).length} / ${Object.keys(summary.by_model).length}`}
              icon={<Database className="h-4 w-4" />}
            />
            <StatCard
              label="Tokens in / out"
              value={`${formatNumber(summary.tokens.input)} / ${formatNumber(summary.tokens.output)}`}
              icon={<Hash className="h-4 w-4" />}
            />
          </div>

          {monitor > 0 ? (
            <div className="flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-sm">
              <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
              <p className="text-amber-700 dark:text-amber-300">
                <span className="font-medium">{monitor}</span> interaction(s) ran under the{" "}
                <span className="font-mono">monitor</span> policy — real PII was forwarded to the
                upstream and only logged here. Switch to <span className="font-mono">pseudonymize</span>{" "}
                to replace PII before it leaves the machine.
              </p>
            </div>
          ) : null}

          <div className="grid gap-4 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Activity over time</CardTitle>
              </CardHeader>
              <CardContent>
                {summary.timeseries.length ? (
                  <ActivityAreaChart data={summary.timeseries} />
                ) : (
                  <p className="py-12 text-center text-sm text-muted-foreground">No data</p>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>By source</CardTitle>
              </CardHeader>
              <CardContent>
                <SourcePie data={sourceData} />
                <div className="mt-2 space-y-1">
                  {sourceData.map((s, i) => (
                    <div key={s.name} className="flex items-center justify-between text-xs">
                      <span className="flex items-center gap-2">
                        <span
                          className="h-2 w-2 rounded-full"
                          style={{ backgroundColor: CHART_PALETTE[i % CHART_PALETTE.length] }}
                        />
                        {s.name}
                      </span>
                      <span className="tabular-nums text-muted-foreground">{s.value}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Entities by type</CardTitle>
              </CardHeader>
              <CardContent>
                {entityData.length ? (
                  <EntityBarChart data={entityData} />
                ) : (
                  <p className="py-12 text-center text-sm text-muted-foreground">No PII detected</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Recent interactions</CardTitle>
                <Link
                  href="/interactions"
                  className="flex items-center gap-1 text-xs text-primary hover:underline"
                >
                  View all <ArrowRight className="h-3 w-3" />
                </Link>
              </CardHeader>
              <CardContent className="space-y-1">
                {recent.items.map((row) => (
                  <Link
                    key={row.id}
                    href={`/interactions/${row.id}`}
                    className="flex items-center justify-between gap-3 rounded-md px-2 py-2 text-sm hover:bg-accent"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <Cpu className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                        <span className="truncate font-medium">{row.model}</span>
                        <PolicyBadge policy={row.policy} />
                      </div>
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        {formatDateTime(row.created_at)} · {row.source}
                      </div>
                    </div>
                    <EntityChips counts={row.entity_counts} max={2} />
                  </Link>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
