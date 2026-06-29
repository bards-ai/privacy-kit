import { OverviewLive } from "@/components/overview-live";
import { ConnectionError, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
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

  return (
    <>
      <PageHeader
        title="Overview"
        description="What the on-device PII gateway has seen across all proxied traffic."
      />
      {error ? (
        <ConnectionError message={error} />
      ) : (
        <OverviewLive initialSummary={summary} initialRecent={recent} />
      )}
    </>
  );
}
