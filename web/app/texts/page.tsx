import Link from "next/link";

import { FilterBar } from "@/components/interactions-controls";
import { Card, CardContent, ConnectionError, EmptyState, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import type { FilterValues, TextsResponse } from "@/lib/types";

export const dynamic = "force-dynamic";

const EMPTY_TEXTS: TextsResponse = { texts: [], redacted: false };
const EMPTY_FILTERS: FilterValues = {
  sources: [],
  wire_formats: [],
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

      {data.redacted ? (
        <p className="mb-4 text-xs text-amber-500">
          Originals are redacted (PII_EXPOSE_PLAINTEXT=false). Showing anonymized text only.
        </p>
      ) : null}

      {error ? (
        <ConnectionError message={error} />
      ) : data.texts.length === 0 ? (
        <EmptyState
          title="No saved text segments"
          description="Segments are saved per PII_SAVE_TEXTS. Send a prompt with PII through the gateway to populate this view."
        />
      ) : (
        <div className="space-y-3">
          {data.texts.map((t) => (
            <Card key={`${t.interaction_id}-${t.seq}`}>
              <CardContent className="pt-4">
                <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">{t.source}</span>
                  <span>·</span>
                  <span>{t.model}</span>
                  <span>·</span>
                  <span>{formatDateTime(t.when)}</span>
                  <Link
                    href={`/interactions/${t.interaction_id}`}
                    className="ml-auto text-primary hover:underline"
                  >
                    Interaction #{t.interaction_id}
                  </Link>
                </div>
                <div className="grid gap-3 lg:grid-cols-2">
                  <div>
                    <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                      Original
                    </div>
                    <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm">
                      {t.original ?? <span className="text-muted-foreground">[redacted]</span>}
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                      Anonymized
                    </div>
                    <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 font-mono text-sm">
                      {t.anonymized}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </>
  );
}
