"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useMemo, useState } from "react";

import { TextSegmentsBeforeAfter } from "@/components/text-highlight";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui";
import { entityColor } from "@/lib/colors";
import { alignPii, valuesByType, type PiiSpan } from "@/lib/pii";
import type { Detection, TextSegment } from "@/lib/types";

// One detection row: entity type + count, expandable to the values identified.
function DetectionRow({ det, values }: { det: Detection; values: PiiSpan[] }) {
  const [open, setOpen] = useState(false);
  const color = entityColor(det.entity_type);
  const hidden = det.count - values.length; // values present only in non-saved text
  return (
    <div className="border-b last:border-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 py-2 text-left text-sm hover:bg-muted/40"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        )}
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
        <span className="font-medium">{det.entity_type}</span>
        <span className="ml-auto tabular-nums text-muted-foreground">×{det.count}</span>
      </button>
      {open ? (
        <div className="px-7 pb-3">
          {values.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              No values stored for this type (found only in system/assistant text, which is never
              saved).
            </p>
          ) : (
            <>
              <div className="flex flex-wrap gap-1.5">
                {values.map((v) => (
                  <span
                    key={v.placeholder}
                    title={v.placeholder}
                    className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs"
                    style={{ backgroundColor: `${color}1a`, color }}
                  >
                    <span className="font-mono opacity-70">{v.placeholder}</span>
                    {v.value ? <span className="font-medium text-foreground">{v.value}</span> : null}
                  </span>
                ))}
              </div>
              {hidden > 0 ? (
                <p className="mt-2 text-xs text-muted-foreground">
                  +{hidden} more not shown (present only in unsaved text).
                </p>
              ) : null}
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}

export function InteractionDetailBody({
  detections,
  texts,
  redacted,
}: {
  detections: Detection[];
  texts: TextSegment[];
  redacted: boolean;
}) {
  const aligned = useMemo(() => texts.map((t) => alignPii(t.original, t.anonymized)), [texts]);
  const byType = useMemo(() => valuesByType(aligned), [aligned]);

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Detections</CardTitle>
        </CardHeader>
        <CardContent>
          {detections.length === 0 ? (
            <p className="text-sm text-muted-foreground">No PII detected.</p>
          ) : (
            <div className="max-w-xl">
              {detections.map((d) => (
                <DetectionRow key={d.id} det={d} values={byType.get(d.entity_type) ?? []} />
              ))}
              <p className="mt-3 text-xs text-muted-foreground">
                Expand a type to preview the values identified. Highlighted inline below.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Saved text — before &amp; after</CardTitle>
        </CardHeader>
        <CardContent>
          {redacted ? (
            <p className="mb-3 text-xs text-amber-500">
              Originals are redacted (PII_EXPOSE_PLAINTEXT=false). Showing masked text only.
            </p>
          ) : null}
          {texts.length === 0 ? (
            <p className="text-sm text-muted-foreground">No text segments saved for this interaction.</p>
          ) : (
            <TextSegmentsBeforeAfter texts={texts} />
          )}
        </CardContent>
      </Card>
    </>
  );
}
