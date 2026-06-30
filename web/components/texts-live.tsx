"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useMemo } from "react";

import { HighlightedText } from "@/components/interaction-detail-body";
import { Card, CardContent, ConnectionError, EmptyState } from "@/components/ui";
import { clientGet } from "@/lib/client-api";
import { formatDateTime } from "@/lib/format";
import { alignPii } from "@/lib/pii";
import type { TextsResponse } from "@/lib/types";

export function TextsLive({ initialData }: { initialData: TextsResponse }) {
  const searchParams = useSearchParams();
  const qs = new URLSearchParams();
  for (const [k, v] of searchParams.entries()) {
    if (v !== "") qs.set(k, v);
  }
  const query = qs.toString();
  const path = `/texts${query ? `?${query}` : ""}`;

  const { data, isError, error } = useQuery({
    queryKey: ["texts", query],
    queryFn: () => clientGet<TextsResponse>(path),
    initialData,
    refetchInterval: 5000,
  });

  const aligned = useMemo(
    () => data.texts.map((t) => alignPii(t.original, t.anonymized)),
    [data.texts],
  );

  if (isError) {
    return <ConnectionError message={error instanceof Error ? error.message : String(error)} />;
  }

  return (
    <>
      {data.redacted ? (
        <p className="mb-4 text-xs text-amber-500">
          Originals are redacted (PII_EXPOSE_PLAINTEXT=false). Showing anonymized text only.
        </p>
      ) : null}

      {data.texts.length === 0 ? (
        <EmptyState
          title="No saved text segments"
          description="Segments are saved per PII_SAVE_TEXTS. Send a prompt with PII through the gateway to populate this view."
        />
      ) : (
        <div className="space-y-3">
          {data.texts.map((t, i) => (
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
                      {t.original === null ? (
                        <span className="text-muted-foreground">[redacted]</span>
                      ) : (
                        <HighlightedText aligned={aligned[i]} view="original" />
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">
                      Anonymized
                    </div>
                    <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 font-mono text-sm">
                      <HighlightedText aligned={aligned[i]} view="masked" />
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
