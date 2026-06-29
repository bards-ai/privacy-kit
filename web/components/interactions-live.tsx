"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { EntityChips } from "@/components/entity-chips";
import { Pagination, SortHeader } from "@/components/interactions-controls";
import { PolicyBadge } from "@/components/policy-badge";
import { Card, ConnectionError, EmptyState } from "@/components/ui";
import { clientGet } from "@/lib/client-api";
import { formatDateTime, formatTokens } from "@/lib/format";
import type { InteractionList } from "@/lib/types";

export function InteractionsLive({ initialData }: { initialData: InteractionList }) {
  const searchParams = useSearchParams();
  const qs = new URLSearchParams();
  for (const [k, v] of searchParams.entries()) {
    if (v !== "") qs.set(k, v);
  }
  const query = qs.toString();
  const path = `/interactions${query ? `?${query}` : ""}`;

  const { data: list, isError, error } = useQuery({
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

  return (
    <>
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[940px] text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th className="px-4 py-3">
                  <SortHeader column="created_at" label="When" />
                </th>
                <th className="px-4 py-3">
                  <SortHeader column="source" label="Source" />
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
              {list.items.map((row) => (
                <tr key={row.id} className="border-b last:border-0 hover:bg-accent/50">
                  <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
                    {formatDateTime(row.created_at)}
                  </td>
                  <td className="px-4 py-3">{row.source}</td>
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
