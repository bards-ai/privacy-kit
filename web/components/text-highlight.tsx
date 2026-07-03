"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { Fragment, useMemo, useState, type ReactNode } from "react";

import { cn } from "@/lib/cn";
import { entityColor } from "@/lib/colors";
import {
  alignPii,
  hunkify,
  type Aligned,
  type AlignedLine,
  type HunkBlock,
  type PiiPart,
  type PiiSpan,
} from "@/lib/pii";
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
        <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm">
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

// Render a run of aligned lines, joining them with newlines so a
// `whitespace-pre-wrap` container lays them out as real lines. PII spans are
// shown as placeholders (`view="masked"`) or recovered values (`view="original"`).
function renderLines(lines: AlignedLine[], view: "masked" | "original") {
  return lines.map((line, i) => (
    <Fragment key={i}>
      {i > 0 ? "\n" : null}
      {line.parts.map((part, j) =>
        part.kind === "lit" ? (
          <span key={j}>{part.text}</span>
        ) : (
          <PiiToken key={j} span={part.span} show={view === "masked" ? "placeholder" : "value"} />
        ),
      )}
    </Fragment>
  ));
}

// One collapsed run of PII-free lines, git-diff style. Shared expansion state is
// keyed on the block's index so both columns of a before/after grid toggle in
// lock-step.
function GapStrip({
  count,
  open,
  onToggle,
}: {
  count: number;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="my-1 flex w-full items-center gap-1.5 rounded border border-dashed bg-muted/30 px-2 py-1 font-sans text-xs text-muted-foreground hover:bg-muted/50"
    >
      {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      {count} line{count === 1 ? "" : "s"} {open ? "" : "hidden"}
    </button>
  );
}

// One column of the hunked before/after view: only PII neighborhoods are shown;
// PII-free stretches collapse to GapStrips that expand in place. `expanded`
// holds the indices of expanded gap blocks (shared across both columns).
function PiiHunkColumn({
  label,
  blocks,
  view,
  expanded,
  onToggle,
  redacted,
}: {
  label: string;
  blocks: HunkBlock[];
  view: "masked" | "original";
  expanded: Set<number>;
  onToggle: (i: number) => void;
  redacted?: boolean;
}) {
  return (
    <div>
      <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm">
        {redacted ? (
          <span className="text-muted-foreground">[redacted]</span>
        ) : (
          blocks.map((block, i) =>
            block.kind === "lines" ? (
              <Fragment key={i}>{renderLines(block.lines, view)}</Fragment>
            ) : (
              <Fragment key={i}>
                <GapStrip
                  count={block.count}
                  open={expanded.has(i)}
                  onToggle={() => onToggle(i)}
                />
                {expanded.has(i) ? renderLines(block.lines, view) : null}
              </Fragment>
            ),
          )
        )}
      </div>
    </div>
  );
}

// PII-focused before/after view of one segment: the two columns share a single
// line-based hunking so gaps line up, and expanding a gap reveals its hidden
// lines in both columns at once.
function PiiHunks({ aligned, redacted }: { aligned: Aligned; redacted?: boolean }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const { blocks } = useMemo(() => hunkify(aligned), [aligned]);
  const toggle = (i: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <PiiHunkColumn
        label="Original"
        blocks={blocks}
        view="original"
        expanded={expanded}
        onToggle={toggle}
        redacted={redacted}
      />
      <PiiHunkColumn
        label="Anonymized"
        blocks={blocks}
        view="masked"
        expanded={expanded}
        onToggle={toggle}
      />
    </div>
  );
}

// A PII-free internal segment collapsed to a single strip (like an unchanged
// file in a PR). Expanding reveals the full before/after grid.
function NoPiiSegment({
  category,
  lineCount,
  aligned,
  redacted,
}: {
  category: string;
  lineCount: number;
  aligned: Aligned;
  redacted?: boolean;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 rounded-md border border-dashed bg-muted/30 px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted/50"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <span className="font-medium">{category}</span>
        <span>
          · {lineCount} line{lineCount === 1 ? "" : "s"} · no PII detected
        </span>
      </button>
      {open ? (
        <div className="mt-2 grid gap-3 lg:grid-cols-2">
          <TextBox label="Original" aligned={aligned} view="original" redacted={redacted} />
          <TextBox label="Anonymized" aligned={aligned} view="masked" />
        </div>
      ) : null}
    </div>
  );
}

// The before/after grid for a list of saved text segments, shared by the
// single-interaction detail view and the conversation thread view. Each segment
// is aligned once (original ↔ anonymized) and shown as recovered originals on
// the left, `[TYPE_N]` placeholders on the right.
//
// `variant="full"` (default) shows every segment in full (with the long-text
// clamp). `variant="diff"` is the PII-focused view for internal messages: only
// the neighborhoods around detected PII are shown, PII-free stretches collapse
// to expandable gaps, and a segment with no PII shrinks to one expandable strip.
export function TextSegmentsBeforeAfter({
  texts,
  variant = "full",
}: {
  texts: TextSegment[];
  variant?: "full" | "diff";
}) {
  const aligned = useMemo(() => texts.map((t) => alignPii(t.original, t.anonymized)), [texts]);
  const hunks = useMemo(
    () => (variant === "diff" ? aligned.map((a) => hunkify(a)) : null),
    [aligned, variant],
  );
  return (
    <div className="space-y-4">
      {texts.map((t, i) => {
        if (variant === "diff" && hunks) {
          if (hunks[i].piiCount === 0) {
            return (
              <NoPiiSegment
                key={t.id}
                category={t.category}
                lineCount={hunks[i].lineCount}
                aligned={aligned[i]}
                redacted={t.original === null}
              />
            );
          }
          return <PiiHunks key={t.id} aligned={aligned[i]} redacted={t.original === null} />;
        }
        return (
          <div key={t.id} className="grid gap-3 lg:grid-cols-2">
            <TextBox
              label="Original"
              aligned={aligned[i]}
              view="original"
              redacted={t.original === null}
            />
            <TextBox label="Anonymized" aligned={aligned[i]} view="masked" />
          </div>
        );
      })}
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
