"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, Cpu, Database, Download, Hash, ShieldAlert } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { ActivityAreaChart, EntityBarChart, SourcePie } from "@/components/charts";
import { EntityChips } from "@/components/entity-chips";
import { DateRangeFilter } from "@/components/interactions-controls";
import { PolicyBadge } from "@/components/policy-badge";
import { StatCard } from "@/components/stat-card";
import { Card, CardContent, CardHeader, CardTitle, EmptyState } from "@/components/ui";
import { clientGet } from "@/lib/client-api";
import { CHART_PALETTE } from "@/lib/colors";
import { formatDateTime, formatNumber } from "@/lib/format";
import type { InteractionList, Summary } from "@/lib/types";

const RECENT_PATH = "/interactions?page_size=8&sort=created_at&order=desc";

export function OverviewLive({
  initialSummary,
  initialRecent,
}: {
  initialSummary: Summary;
  initialRecent: InteractionList;
}) {
  const searchParams = useSearchParams();
  const qs = new URLSearchParams();
  for (const k of ["date_from", "date_to"] as const) {
    const v = searchParams.get(k);
    if (v) qs.set(k, v);
  }
  const query = qs.toString();

  const { data: summary } = useQuery({
    queryKey: ["summary", query],
    queryFn: () => clientGet<Summary>(`/summary${query ? `?${query}` : ""}`),
    initialData: initialSummary,
    refetchInterval: 5000,
  });
  const { data: recent } = useQuery({
    queryKey: ["interactions", "recent", query],
    queryFn: () => clientGet<InteractionList>(`${RECENT_PATH}${query ? `&${query}` : ""}`),
    initialData: initialRecent,
    refetchInterval: 5000,
  });

  if (summary.interactions === 0) {
    if (query) {
      return (
        <div className="space-y-4">
          <DateRangeFilter />
          <EmptyState
            icon={<Activity className="h-8 w-8" />}
            title="No interactions in this range"
            description="Nothing was recorded between the selected dates. Widen the range or reset the filter to see all activity."
          />
        </div>
      );
    }
    return (
      <div className="space-y-4">
        <div className="flex items-start gap-3 rounded-lg border border-primary/30 bg-primary/5 p-3 text-sm">
          <Download className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          <p>
            <span className="font-medium">Already have conversations?</span> You can import your
            existing Claude Code and Codex history from{" "}
            <Link href="/settings" className="text-primary hover:underline">
              Settings → Import history
            </Link>
            .
          </p>
        </div>
        <EmptyState
          icon={<Activity className="h-8 w-8" />}
          title="No interactions recorded yet"
          description="Route a tool through the gateway (see Settings) and send a prompt — it will show up here. Until then, try the Live preview to see PII detection in action."
        />
      </div>
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
    <div className="space-y-6">
      <DateRangeFilter />
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
              href="/conversations"
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
  );
}
