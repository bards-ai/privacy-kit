import { ConversationsLive } from "@/components/conversations-live";
import { FilterBar } from "@/components/interactions-controls";
import { ConnectionError, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import type { ConversationList, FilterValues } from "@/lib/types";

export const dynamic = "force-dynamic";

const EMPTY_LIST: ConversationList = {
  items: [],
  page: 1,
  page_size: 50,
  total: 0,
  total_pages: 0,
};
const EMPTY_FILTERS: FilterValues = {
  sources: [],
  wire_formats: [],
  kinds: [],
  models: [],
  policies: [],
  languages: [],
  entity_types: [],
};

export default async function ConversationsPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(searchParams)) {
    if (typeof v === "string" && v !== "") qs.set(k, v);
  }
  const query = qs.toString();
  const { data: list, error } = await apiGetOr<ConversationList>(
    `/conversations${query ? `?${query}` : ""}`,
    EMPTY_LIST,
  );
  const { data: filters } = await apiGetOr<FilterValues>("/filters", EMPTY_FILTERS);

  return (
    <>
      <PageHeader
        title="Conversations"
        description="Multi-turn conversations, grouped by their opening message."
      />
      <FilterBar filters={filters} />

      {error ? <ConnectionError message={error} /> : <ConversationsLive initialData={list} />}
    </>
  );
}
