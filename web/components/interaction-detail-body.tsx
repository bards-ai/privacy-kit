"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui";
import { cn } from "@/lib/cn";
import { entityColor } from "@/lib/colors";
import { alignPii, valuesByType, type Aligned, type PiiPart, type PiiSpan } from "@/lib/pii";
import type { Detection, TextSegment } from "@/lib/types";

// Above this many characters a saved-text block is clamped behind "Show more".
const COLLAPSE_CHARS = 600;

function highlightStyle(type: string) {
  const color = entityColor(type);
  return { backgroundColor: `${color}22`, color, boxShadow: `inset 0 -1px 0 ${color}` };
}

// One inline PII token, colored by entity type. Hover reveals the counterpart
// (original value under the placeholder, or vice-versa) when it's known.
function PiiToken({ span, show }: { span: PiiSpan; show: "value" | "placeholder" }) {
  const text = show === "value" ? (span.value ?? span.placeholder) : span.placeholder;
  const title =
    show === "value" ? span.placeholder : span.value ? `${span.type}: ${span.value}` : span.type;
  return (
    <mark
      title={title}
      style={highlightStyle(span.type)}
      className="rounded-[3px] px-0.5 font-medium"
    >
      {text}
    </mark>
  );
}

// Render aligned parts with PII spans highlighted. `view="masked"` shows the
// `[TYPE_N]` placeholders; `view="original"` shows the recovered real values.
export function HighlightedText({ aligned, view }: { aligned: Aligned; view: "masked" | "original" }) {
  return (
    <>
      {aligned.parts.map((part: PiiPart, i: number) =>
        part.kind === "lit" ? (
          <span key={i}>{part.text}</span>
        ) : (
          <PiiToken key={i} span={part.span} show={view === "masked" ? "placeholder" : "value"} />
        ),
      )}
    </>
  );
}

function Collapsible({ enabled, children }: { enabled: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  if (!enabled) return <>{children}</>;
  return (
    <div>
      <div className={cn("relative overflow-hidden", !open && "max-h-32")}>
        {children}
        {!open ? (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-background to-transparent" />
        ) : null}
      </div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="mt-1 inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {open ? "Show less" : "Show more"}
      </button>
    </div>
  );
}

function TextBox({
  label,
  aligned,
  view,
  redacted,
}: {
  label: string;
  aligned: Aligned;
  view: "masked" | "original";
  redacted?: boolean;
}) {
  const len = aligned.parts.reduce(
    (n, p) => n + (p.kind === "lit" ? p.text.length : p.span.placeholder.length),
    0,
  );
  return (
    <div>
      <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <Collapsible enabled={len > COLLAPSE_CHARS}>
        <div
          className={cn(
            "whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm",
            view === "masked" && "font-mono",
          )}
        >
          {redacted ? (
            <span className="text-muted-foreground">[redacted]</span>
          ) : (
            <HighlightedText aligned={aligned} view={view} />
          )}
        </div>
      </Collapsible>
    </div>
  );
}

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
            <div className="space-y-4">
              {texts.map((t, i) => (
                <div key={t.id} className="grid gap-3 lg:grid-cols-2">
                  <TextBox
                    label="Original"
                    aligned={aligned[i]}
                    view="original"
                    redacted={t.original === null}
                  />
                  <TextBox label="Anonymized" aligned={aligned[i]} view="masked" />
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}
