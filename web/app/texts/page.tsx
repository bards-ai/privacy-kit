import { FilterBar } from "@/components/interactions-controls";
import { TextsLive } from "@/components/texts-live";
import { ConnectionError, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import type { FilterValues, TextsResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

const EMPTY_TEXTS: TextsResponse = { texts: [], redacted: false };
const EMPTY_FILTERS: FilterValues = {
  sources: [],
  wire_formats: [],
  kinds: [],
  models: [],
  policies: [],
  languages: [],
  entity_types: [],
};

export default async function TextsPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(searchParams)) {
    if (typeof v === "string" && v !== "") qs.set(k, v);
  }
  const query = qs.toString();
  const { data, error } = await apiGetOr<TextsResponse>(
    `/texts${query ? `?${query}` : ""}`,
    EMPTY_TEXTS,
  );
  const { data: filters } = await apiGetOr<FilterValues>("/filters", EMPTY_FILTERS);

  return (
    <>
      <PageHeader
        title="Text segments"
        description="User-authored text the gateway saved, with PII before and after anonymization."
      />
      <FilterBar filters={filters} />

      {error ? <ConnectionError message={error} /> : <TextsLive initialData={data} />}
    </>
  );
}
