"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { cn } from "@/lib/cn";
import { entityColor } from "@/lib/colors";
import { alignPii, type Aligned, type PiiPart, type PiiSpan } from "@/lib/pii";
import type { TextSegment } from "@/lib/types";

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
export function HighlightedText({
  aligned,
  view,
}: {
  aligned: Aligned;
  view: "masked" | "original";
}) {
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

export function TextBox({
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

// The before/after grid for a list of saved text segments, shared by the
// single-interaction detail view and the conversation thread view. Each segment
// is aligned once (original ↔ anonymized) and shown as recovered originals on
// the left, `[TYPE_N]` placeholders on the right.
export function TextSegmentsBeforeAfter({ texts }: { texts: TextSegment[] }) {
  const aligned = useMemo(() => texts.map((t) => alignPii(t.original, t.anonymized)), [texts]);
  return (
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
  );
}

// Plain single-column rendering for text that was never anonymized (model
// responses only ever get de-anonymized for display, never scrubbed — see
// transform.py `anthropic_response`/`openai_chat_response` — so there is no
// before/after to show).
export function TextSegmentsPlain({ texts }: { texts: TextSegment[] }) {
  const aligned = useMemo(() => texts.map((t) => alignPii(t.original, t.anonymized)), [texts]);
  return (
    <div className="space-y-4">
      {texts.map((t, i) => {
        const len = aligned[i].parts.reduce(
          (n, p) => n + (p.kind === "lit" ? p.text.length : p.span.placeholder.length),
          0,
        );
        return (
          <Collapsible key={t.id} enabled={len > COLLAPSE_CHARS}>
            <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm">
              {t.original === null ? (
                <span className="text-muted-foreground">[redacted]</span>
              ) : (
                <HighlightedText aligned={aligned[i]} view="original" />
              )}
            </div>
          </Collapsible>
        );
      })}
    </div>
  );
}
