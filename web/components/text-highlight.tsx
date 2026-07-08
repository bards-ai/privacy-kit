"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { Fragment, useMemo, useState, type ReactNode } from "react";

import { cn } from "@/lib/cn";
import { entityColor } from "@/lib/colors";
import {
  alignPii,
  hunkify,
  type Aligned,
  type HunkGap,
  type HunkPiece,
  type PiiPart,
  type PiiSpan,
} from "@/lib/pii";
import type { TextSegment } from "@/lib/types";

// Above this many characters a saved-text block switches from full display to
// the PII-focused hunk view (git-diff style: PII regions shown, rest collapsed).
const COLLAPSE_CHARS = 600;

function highlightStyle(type: string) {
  const color = entityColor(type);
  return { backgroundColor: `${color}22`, color, boxShadow: `inset 0 -1px 0 ${color}` };
}

// Character length of an aligned segment (placeholders counted at token width),
// used to decide when a segment is long enough for the hunk view.
function segLen(aligned: Aligned): number {
  return aligned.parts.reduce(
    (n, p) => n + (p.kind === "lit" ? p.text.length : p.span.placeholder.length),
    0,
  );
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
  return (
    <div>
      <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <Collapsible enabled={segLen(aligned) > COLLAPSE_CHARS}>
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

// Render aligned parts inline; literals may span newlines (laid out as real
// lines by the `whitespace-pre-wrap` container). PII spans are shown as
// placeholders (`view="masked"`) or recovered values (`view="original"`).
function renderParts(parts: PiiPart[], view: "masked" | "original") {
  return parts.map((part, j) =>
    part.kind === "lit" ? (
      <span key={j}>{part.text}</span>
    ) : (
      <PiiToken key={j} span={part.span} show={view === "masked" ? "placeholder" : "value"} />
    ),
  );
}

// The expander for one collapsed PII-free stretch. A line-shaped gap (short
// lines collapsed on line boundaries) reads "N lines hidden"; a mid-line cut
// through one long line/paragraph reads "N characters hidden".
function GapStrip({ gap, open, onToggle }: { gap: HunkGap; open: boolean; onToggle: () => void }) {
  const label =
    gap.lineShaped && gap.lines > 0
      ? `${gap.lines} line${gap.lines === 1 ? "" : "s"} hidden`
      : `${gap.chars.toLocaleString()} characters hidden`;
  return (
    <button
      type="button"
      onClick={onToggle}
      className="my-1 flex w-full items-center gap-1.5 rounded border border-dashed bg-muted/30 px-2 py-1 font-sans text-xs text-muted-foreground hover:bg-muted/50"
    >
      {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      {label}
    </button>
  );
}

// The sequence of hunk pieces: visible runs highlighted inline, PII-free
// stretches collapsed to GapStrips that expand in place.
function HunkBody({
  pieces,
  view,
  expanded,
  onToggle,
}: {
  pieces: HunkPiece[];
  view: "masked" | "original";
  expanded: Set<number>;
  onToggle: (i: number) => void;
}) {
  return (
    <>
      {pieces.map((piece, i) =>
        piece.kind === "text" ? (
          <Fragment key={i}>{renderParts(piece.parts, view)}</Fragment>
        ) : (
          <Fragment key={i}>
            <GapStrip gap={piece} open={expanded.has(i)} onToggle={() => onToggle(i)} />
            {expanded.has(i) ? renderParts(piece.parts, view) : null}
          </Fragment>
        ),
      )}
    </>
  );
}

// Track which gap blocks are expanded. Shared across both columns of a
// before/after grid so they toggle in lock-step.
function useGapToggle() {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const toggle = (i: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  return { expanded, toggle };
}

// One labeled column of the hunked before/after view.
function PiiHunkColumn({
  label,
  pieces,
  view,
  expanded,
  onToggle,
  redacted,
}: {
  label: string;
  pieces: HunkPiece[];
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
          <HunkBody pieces={pieces} view={view} expanded={expanded} onToggle={onToggle} />
        )}
      </div>
    </div>
  );
}

// PII-focused before/after view of one segment: the two columns share a single
// hunking so gaps line up, and expanding a gap reveals its hidden content in
// both columns at once.
function PiiHunks({ aligned, redacted }: { aligned: Aligned; redacted?: boolean }) {
  const { expanded, toggle } = useGapToggle();
  const { pieces } = useMemo(() => hunkify(aligned), [aligned]);
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <PiiHunkColumn
        label="Original"
        pieces={pieces}
        view="original"
        expanded={expanded}
        onToggle={toggle}
        redacted={redacted}
      />
      <PiiHunkColumn
        label="Anonymized"
        pieces={pieces}
        view="masked"
        expanded={expanded}
        onToggle={toggle}
      />
    </div>
  );
}

// Single-column hunk view for text that has no before/after (model responses):
// only the PII neighborhoods are shown, PII-free stretches collapse to gaps.
function PiiHunksPlain({ aligned }: { aligned: Aligned }) {
  const { expanded, toggle } = useGapToggle();
  const { pieces } = useMemo(() => hunkify(aligned), [aligned]);
  return (
    <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm">
      <HunkBody pieces={pieces} view="original" expanded={expanded} onToggle={toggle} />
    </div>
  );
}

// For long text with no detected PII (user prompts / model responses): show the
// start of the text and collapse the remainder behind an expander, so the reader
// starts at the top rather than at a PII-centered window. Single column, since
// with no PII the original and anonymized text are identical.
function HeadCollapse({ aligned }: { aligned: Aligned }) {
  const { expanded, toggle } = useGapToggle();
  const pieces = useMemo<HunkPiece[]>(() => {
    const text = aligned.parts
      .map((p) => (p.kind === "lit" ? p.text : p.span.placeholder))
      .join("");
    if (text.length <= COLLAPSE_CHARS) {
      return [{ kind: "text", parts: [{ kind: "lit", text }] }];
    }
    // Keep the first COLLAPSE_CHARS, extended to the end of that line.
    const nl = text.indexOf("\n", COLLAPSE_CHARS);
    const cut = nl === -1 ? COLLAPSE_CHARS : nl + 1;
    const rest = text.slice(cut);
    return [
      { kind: "text", parts: [{ kind: "lit", text: text.slice(0, cut) }] },
      {
        kind: "gap",
        parts: [{ kind: "lit", text: rest }],
        chars: rest.length,
        lines: rest.split("\n").length - 1,
        lineShaped: nl !== -1,
      },
    ];
  }, [aligned]);
  return (
    <div className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm">
      <HunkBody pieces={pieces} view="original" expanded={expanded} onToggle={toggle} />
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
// `variant="diff"` is the PII-focused view for internal messages: every segment
// uses the hunk view. `variant="full"` (default, for user prompts) shows short
// segments in full and only switches a segment to the hunk view once it grows
// past COLLAPSE_CHARS. Either way the hunk view shows only the neighborhoods
// around detected PII, collapses PII-free stretches to expandable gaps, and
// shrinks a segment with no PII to one expandable strip.
export function TextSegmentsBeforeAfter({
  texts,
  variant = "full",
}: {
  texts: TextSegment[];
  variant?: "full" | "diff";
}) {
  const aligned = useMemo(() => texts.map((t) => alignPii(t.original, t.anonymized)), [texts]);
  const hunks = useMemo(() => aligned.map((a) => hunkify(a)), [aligned]);
  return (
    <div className="space-y-4">
      {texts.map((t, i) => {
        const useHunk = variant === "diff" || segLen(aligned[i]) > COLLAPSE_CHARS;
        if (useHunk) {
          if (hunks[i].piiCount === 0) {
            // Internal messages keep the collapsed "no PII" strip; a long user
            // prompt with no PII instead shows its start and collapses the rest.
            if (variant === "diff" || t.original === null) {
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
            return <HeadCollapse key={t.id} aligned={aligned[i]} />;
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
// before/after to show). Short segments render in full; once a segment grows
// past COLLAPSE_CHARS it switches to the single-column hunk view so the PII
// stays visible and the bulk collapses git-style instead of clamping.
export function TextSegmentsPlain({ texts }: { texts: TextSegment[] }) {
  const aligned = useMemo(() => texts.map((t) => alignPii(t.original, t.anonymized)), [texts]);
  return (
    <div className="space-y-4">
      {texts.map((t, i) => {
        if (t.original === null) {
          return (
            <div
              key={t.id}
              className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm"
            >
              <span className="text-muted-foreground">[redacted]</span>
            </div>
          );
        }
        if (segLen(aligned[i]) > COLLAPSE_CHARS) {
          const hasPii = aligned[i].parts.some((p) => p.kind === "pii");
          return hasPii ? (
            <PiiHunksPlain key={t.id} aligned={aligned[i]} />
          ) : (
            <HeadCollapse key={t.id} aligned={aligned[i]} />
          );
        }
        return (
          <div
            key={t.id}
            className="whitespace-pre-wrap break-words rounded-md border bg-background p-3 text-sm"
          >
            <HighlightedText aligned={aligned[i]} view="original" />
          </div>
        );
      })}
    </div>
  );
}
