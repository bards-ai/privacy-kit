"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import Link from "next/link";
import { useState, type ReactNode } from "react";

import { CategoryBadge } from "@/components/category-badge";
import { EntityChips } from "@/components/entity-chips";
import { TextSegmentsBeforeAfter } from "@/components/text-highlight";
import { Card, CardContent } from "@/components/ui";
import { cn } from "@/lib/cn";
import { formatDateTime, formatTokens } from "@/lib/format";
import { isBackground, kindMeta } from "@/lib/kind";
import type { ConversationTurn, TextSegment } from "@/lib/types";

function KindBadge({ kind }: { kind: string }) {
  if (kind === "main") return null;

  const m = kindMeta(kind);
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium"
      style={{ backgroundColor: `${m.color}1a`, color: m.color }}
    >
      {m.short}
    </span>
  );
}

// One labeled section of a turn (prompt or response) with its before/after text.
function SegmentGroup({ heading, segments }: { heading: ReactNode; segments: TextSegment[] }) {
  if (segments.length === 0) return null;
  return (
    <div className="space-y-2">
      {heading}
      <TextSegmentsBeforeAfter texts={segments} />
    </div>
  );
}

// One real (main) turn rendered as a thread entry: metadata header, the saved
// prompt (user/tool) text, then the agent's response, and a link to the raw
// per-turn view within this conversation.
function TurnCard({ turn, conversationId }: { turn: ConversationTurn; conversationId: string }) {
  const it = turn.interaction;
  const prompt = turn.texts.filter((t) => t.category !== "assistant");
  const response = turn.texts.filter((t) => t.category === "assistant");
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
          <KindBadge kind={it.kind} />
          <span className="font-medium">{it.source}</span>
          <span className="text-muted-foreground">{it.model}</span>
          <span className="text-muted-foreground">· {formatDateTime(it.created_at)}</span>
          <span className="text-muted-foreground">
            · {formatTokens(it.input_tokens, it.output_tokens)}
          </span>
          <div className="ml-auto flex items-center gap-3">
            <EntityChips counts={it.entity_counts} max={4} />
            <Link
              href={`/conversations/${encodeURIComponent(conversationId)}/interactions/${it.id}`}
              className="text-primary hover:underline"
            >
              View
            </Link>
          </div>
        </div>
        {turn.texts.length === 0 ? (
          <p className="text-sm text-muted-foreground">No text saved for this turn.</p>
        ) : (
          <div className="space-y-4">
            <SegmentGroup
              heading={
                <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Prompt
                </span>
              }
              segments={prompt}
            />
            <SegmentGroup heading={<CategoryBadge category="assistant" />} segments={response} />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// A run of consecutive background (safety/helper) turns, collapsed inline at
// their position in the thread — same affordance the interactions list uses.
function BackgroundRun({
  turns,
  conversationId,
}: {
  turns: ConversationTurn[];
  conversationId: string;
}) {
  const [open, setOpen] = useState(false);
  const entities = turns.reduce((n, t) => n + t.interaction.entity_total, 0);
  return (
    <div className="rounded-md border bg-muted/30">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-4 py-2.5 text-sm hover:bg-muted/50"
      >
        {open ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
        )}
        <span className="font-medium">{turns.length} background call{turns.length > 1 ? "s" : ""}</span>
        <span className="text-xs text-muted-foreground">
          · {entities} entities{open ? "" : " — click to expand"}
        </span>
      </button>
      {open ? (
        <div className="space-y-3 px-4 pb-4">
          {turns.map((t) => (
            <TurnCard key={t.interaction.id} turn={t} conversationId={conversationId} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

// Render the ordered turns as a thread. Consecutive background turns fold into
// one collapsible run at their position; main turns render inline.
export function ConversationThread({
  turns,
  conversationId,
}: {
  turns: ConversationTurn[];
  conversationId: string;
}) {
  const blocks: { background: boolean; turns: ConversationTurn[] }[] = [];
  for (const turn of turns) {
    const bg = isBackground(turn.interaction.kind);
    const last = blocks[blocks.length - 1];
    if (last && last.background && bg) {
      last.turns.push(turn);
    } else if (bg) {
      blocks.push({ background: true, turns: [turn] });
    } else {
      blocks.push({ background: false, turns: [turn] });
    }
  }

  return (
    <div className={cn("space-y-4")}>
      {blocks.map((block, i) =>
        block.background ? (
          <BackgroundRun key={`bg-${i}`} turns={block.turns} conversationId={conversationId} />
        ) : (
          <TurnCard
            key={block.turns[0].interaction.id}
            turn={block.turns[0]}
            conversationId={conversationId}
          />
        ),
      )}
    </div>
  );
}
