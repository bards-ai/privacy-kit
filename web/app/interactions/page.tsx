import Link from "next/link";

import { EntityChips } from "@/components/entity-chips";
import {
  ExportMenu,
  FilterBar,
  Pagination,
  SortHeader,
} from "@/components/interactions-controls";
import { PolicyBadge } from "@/components/policy-badge";
import { Card, ConnectionError, EmptyState, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import { formatDateTime, formatTokens } from "@/lib/format";
import type { FilterValues, InteractionList } from "@/lib/types";

export const dynamic = "force-dynamic";

const EMPTY_LIST: InteractionList = { items: [], page: 1, page_size: 50, total: 0, total_pages: 0 };
const EMPTY_FILTERS: FilterValues = {
  sources: [],
  wire_formats: [],
  models: [],
  policies: [],
  languages: [],
  entity_types: [],
};

export default async function InteractionsPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(searchParams)) {
    if (typeof v === "string" && v !== "") qs.set(k, v);
  }
  const query = qs.toString();
  const { data: list, error } = await apiGetOr<InteractionList>(
    `/interactions${query ? `?${query}` : ""}`,
    EMPTY_LIST,
  );
  const { data: filters } = await apiGetOr<FilterValues>("/filters", EMPTY_FILTERS);

  return (
    <>
      <PageHeader
        title="Interactions"
        description="Every request that passed through the gateway."
        actions={<ExportMenu />}
      />
      <FilterBar filters={filters} />

      {error ? (
        <ConnectionError message={error} />
      ) : list.total === 0 ? (
        <EmptyState
          title="No interactions match"
          description="Try clearing the filters, or route a tool through the gateway to generate traffic."
        />
      ) : (
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
                      <td className="px-4 py-3 tabular-nums text-muted-foreground">
                        {row.text_count}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <Link
                          href={`/interactions/${row.id}`}
                          className="text-primary hover:underline"
                        >
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
      )}
    </>
  );
}
