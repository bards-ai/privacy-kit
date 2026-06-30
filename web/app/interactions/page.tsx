import { InteractionsLive } from "@/components/interactions-live";
import { ExportMenu, FilterBar } from "@/components/interactions-controls";
import { ConnectionError, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import type { FilterValues, InteractionList } from "@/lib/types";

export const dynamic = "force-dynamic";

const EMPTY_LIST: InteractionList = { items: [], page: 1, page_size: 50, total: 0, total_pages: 0 };
const EMPTY_FILTERS: FilterValues = {
  sources: [],
  wire_formats: [],
  kinds: [],
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

      {error ? <ConnectionError message={error} /> : <InteractionsLive initialData={list} />}
    </>
  );
}
