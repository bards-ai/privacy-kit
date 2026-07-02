"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { EntityChips } from "@/components/entity-chips";
import { Pagination, SortHeader } from "@/components/interactions-controls";
import { Card, ConnectionError, EmptyState } from "@/components/ui";
import { clientGet } from "@/lib/client-api";
import { formatDateTime } from "@/lib/format";
import type { ConversationList, ConversationListItem } from "@/lib/types";

function DataRow({ row }: { row: ConversationListItem }) {
  return (
    <tr className="border-b last:border-0 hover:bg-accent/50">
      <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
        {formatDateTime(row.first_seen)}
      </td>
      <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
        {formatDateTime(row.last_seen)}
      </td>
      <td className="px-4 py-3 text-center">{row.turn_count}</td>
      <td className="px-4 py-3">{row.sources.join(", ")}</td>
      <td className="px-4 py-3">{row.models.join(", ")}</td>
      <td className="px-4 py-3">
        <EntityChips counts={row.entity_counts} max={3} />
      </td>
      <td className="px-4 py-3 tabular-nums">{row.entity_total}</td>
      <td className="px-4 py-3 text-center">
        <Link
          href={`/conversations/${encodeURIComponent(row.conversation_id)}`}
          className="text-primary hover:underline"
        >
          View
        </Link>
      </td>
    </tr>
  );
}

export function ConversationsLive({ initialData }: { initialData: ConversationList }) {
  const searchParams = useSearchParams();
  const qs = new URLSearchParams();
  for (const [k, v] of searchParams.entries()) {
    if (v !== "") qs.set(k, v);
  }
  const query = qs.toString();
  const path = `/conversations${query ? `?${query}` : ""}`;

  const {
    data: list,
    isError,
    error,
  } = useQuery({
    queryKey: ["conversations", query],
    queryFn: () => clientGet<ConversationList>(path),
    initialData,
    refetchInterval: 5000,
  });

  if (isError) {
    return <ConnectionError message={error instanceof Error ? error.message : String(error)} />;
  }

  if (list.total === 0) {
    return (
      <EmptyState
        title="No conversations"
        description="Start a multi-turn conversation with Claude Code, Codex, or Cursor to see it here."
      />
    );
  }

  return (
    <>
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1040px] text-sm">
            <thead>
              <tr className="border-b text-left text-xs text-muted-foreground">
                <th className="px-4 py-3">
                  <SortHeader column="first_seen" label="Started" />
                </th>
                <th className="px-4 py-3">
                  <SortHeader column="last_seen" label="Ended" />
                </th>
                <th className="px-4 py-3 text-center">
                  <SortHeader column="turn_count" label="Turns" />
                </th>
                <th className="px-4 py-3">Sources</th>
                <th className="px-4 py-3">Models</th>
                <th className="px-4 py-3">Entities</th>
                <th className="px-4 py-3">
                  <SortHeader column="entity_total" label="Total" />
                </th>
                <th className="px-4 py-3 text-center">Details</th>
              </tr>
            </thead>
            <tbody>
              {list.items.map((row: ConversationListItem) => (
                <DataRow key={row.conversation_id} row={row} />
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
