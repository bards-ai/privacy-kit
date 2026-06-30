"use client";

import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";

import { EntityChips } from "@/components/entity-chips";
import { Pagination, SortHeader } from "@/components/interactions-controls";
import { PolicyBadge } from "@/components/policy-badge";
import { Card, ConnectionError, EmptyState } from "@/components/ui";
import { clientGet } from "@/lib/client-api";
import { cn } from "@/lib/cn";
import { formatDateTime, formatTokens } from "@/lib/format";
import { isBackground, kindMeta } from "@/lib/kind";
import type { InteractionListItem, InteractionList } from "@/lib/types";

const COLUMNS = 12; // keep in sync with the <thead> below (for group-row colSpan)

function KindBadge({ kind }: { kind: string }) {
  const m = kindMeta(kind);
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium"
      style={{ backgroundColor: `${m.color}1a`, color: m.color }}
    >
      {m.short}
    </span>
  );
}

function DataRow({ row, dimmed }: { row: InteractionListItem; dimmed?: boolean }) {
  return (
    <tr className={cn("border-b last:border-0 hover:bg-accent/50", dimmed && "opacity-65")}>
      <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
        {formatDateTime(row.created_at)}
      </td>
      <td className="px-4 py-3">{row.source}</td>
      <td className="px-4 py-3">
        <KindBadge kind={row.kind} />
      </td>
      <td className="px-4 py-3">
        <PolicyBadge policy={row.policy} />
      </td>
      <td className="px-4 py-3 text-muted-foreground">{row.wire_format}</td>
      <td className="px-4 py-3 font-medium">{row.model}</td>
      <td className="px-4 py-3 text-muted-foreground">{row.language ?? "—"}</td>
      <td className="px-4 py-3">
        <EntityChips counts={row.entity_counts} max={3} />
      </td>
      <td className="px-4 py-3 tabular-nums">{row.entity_total}</td>
      <td className="whitespace-nowrap px-4 py-3 tabular-nums text-muted-foreground">
        {formatTokens(row.input_tokens, row.output_tokens)}
      </td>
      <td className="px-4 py-3 tabular-nums text-muted-foreground">{row.text_count}</td>
      <td className="px-4 py-3 text-right">
        <Link href={`/interactions/${row.id}`} className="text-primary hover:underline">
          View
        </Link>
      </td>
    </tr>
  );
}

// Collapsible header for one background kind, plus its rows when expanded.
function BackgroundGroup({ kind, rows }: { kind: string; rows: InteractionListItem[] }) {
  const [open, setOpen] = useState(false);
  const m = kindMeta(kind);
  const entities = rows.reduce((n, r) => n + r.entity_total, 0);
  return (
    <>
      <tr
        className="cursor-pointer border-b bg-muted/30 hover:bg-muted/50"
        onClick={() => setOpen((o) => !o)}
      >
        <td colSpan={COLUMNS} className="px-4 py-2.5">
          <div className="flex items-center gap-2 text-sm">
            {open ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
            <KindBadge kind={kind} />
            <span className="font-medium">{m.label}</span>
            <span className="rounded-full bg-muted px-2 py-0.5 text-xs tabular-nums text-muted-foreground">
              {rows.length}
            </span>
            {!open ? (
              <span className="text-xs text-muted-foreground">
                background calls · {entities} entities — click to preview
              </span>
            ) : null}
          </div>
        </td>
      </tr>
      {open ? rows.map((r) => <DataRow key={r.id} row={r} dimmed />) : null}
    </>
  );
}

export function InteractionsLive({ initialData }: { initialData: InteractionList }) {
  const searchParams = useSearchParams();
  const qs = new URLSearchParams();
  for (const [k, v] of searchParams.entries()) {
    if (v !== "") qs.set(k, v);
  }
  const query = qs.toString();
  const path = `/interactions${query ? `?${query}` : ""}`;

  const {
    data: list,
    isError,
    error,
  } = useQuery({
    queryKey: ["interactions", query],
    queryFn: () => clientGet<InteractionList>(path),
    initialData,
    refetchInterval: 5000,
  });

  if (isError) {
    return <ConnectionError message={error instanceof Error ? error.message : String(error)} />;
  }

  if (list.total === 0) {
    return (
      <EmptyState
        title="No interactions match"
        description="Try clearing the filters, or route a tool through the gateway to generate traffic."
      />
    );
  }

  // Real conversation rows stay inline in their sorted order; background calls
  // (safety/helper) on this page fold into one collapsible group per kind.
  const mains = list.items.filter((r: InteractionListItem) => !isBackground(r.kind));
  const groups = new Map<string, InteractionListItem[]>();
  for (const r of list.items as InteractionListItem[]) {
    if (!isBackground(r.kind)) continue;
    const g = groups.get(r.kind) ?? [];
    g.push(r);
    groups.set(r.kind, g);
  }

  return (
    <>
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1040px] text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th className="px-4 py-3">
                  <SortHeader column="created_at" label="When" />
                </th>
                <th className="px-4 py-3">
                  <SortHeader column="source" label="Source" />
                </th>
                <th className="px-4 py-3">
                  <SortHeader column="kind" label="Kind" />
                </th>
                <th className="px-4 py-3">Policy</th>
                <th className="px-4 py-3">
                  <SortHeader column="wire_format" label="Wire" />
                </th>
                <th className="px-4 py-3">
                  <SortHeader column="model" label="Model" />
                </th>
                <th className="px-4 py-3">Lang</th>
                <th className="px-4 py-3">Entities</th>
                <th className="px-4 py-3">
                  <SortHeader column="entity_total" label="Total" />
                </th>
                <th className="px-4 py-3">
                  <SortHeader column="input_tokens" label="Tokens" />
                </th>
                <th className="px-4 py-3">Text</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {mains.map((row: InteractionListItem) => (
                <DataRow key={row.id} row={row} />
              ))}
              {[...groups.entries()].map(([kind, rows]) => (
                <BackgroundGroup key={kind} kind={kind} rows={rows} />
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Pagination
        page={list.page}
        totalPages={list.total_pages}
        total={list.total}
        pageSize={list.page_size}
      />
    </>
  );
}
