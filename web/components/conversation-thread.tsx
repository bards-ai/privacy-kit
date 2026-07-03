"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import Link from "next/link";
import { useState, type ReactNode } from "react";

import { CategoryBadge } from "@/components/category-badge";
import { EntityChips } from "@/components/entity-chips";
import { TextSegmentsBeforeAfter, TextSegmentsPlain } from "@/components/text-highlight";
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

// One labeled section of a turn (prompt or response) with its text. Prompt
// text (user/tool) gets the before/after anonymization diff; response text
// is rendered plain since the model's output is only ever de-anonymized for
// display, never scrubbed (see TextSegmentsPlain).
function SegmentGroup({
  heading,
  segments,
  plain,
  variant,
}: {
  heading: ReactNode;
  segments: TextSegment[];
  plain?: boolean;
  variant?: "full" | "diff";
}) {
  if (segments.length === 0) return null;
  return (
    <div className="space-y-2">
      {heading}
      {plain ? (
        <TextSegmentsPlain texts={segments} />
      ) : (
        <TextSegmentsBeforeAfter texts={segments} variant={variant} />
      )}
    </div>
  );
}

// Full raw view of one API call: metadata header, the saved prompt (user/tool)
// text, then the agent's response, and a link to the raw per-turn view. Used
// inside the expanded steps of an exchange.
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
              variant="diff"
            />
            <SegmentGroup
              heading={<CategoryBadge category="assistant" />}
              segments={response}
              plain
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// The message the human typed to open an exchange.
function UserMessageCard({ turn }: { turn: ConversationTurn }) {
  const it = turn.interaction;
  const segments = turn.texts.filter((t) => t.category === "user");
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
          <CategoryBadge category="user" />
          <span className="font-medium">{it.source}</span>
          <span className="text-muted-foreground">· {formatDateTime(it.created_at)}</span>
        </div>
        <TextSegmentsBeforeAfter texts={segments} />
      </CardContent>
    </Card>
  );
}

// The model's final answer for an exchange: the assistant text of the last
// main-kind call before the next user message.
function FinalResponseCard({
  turn,
  conversationId,
}: {
  turn: ConversationTurn;
  conversationId: string;
}) {
  const it = turn.interaction;
  const segments = turn.texts.filter((t) => t.category === "assistant");
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
          <CategoryBadge category="assistant" />
          <span className="text-muted-foreground">{it.model}</span>
          <span className="text-muted-foreground">· {formatDateTime(it.created_at)}</span>
          <span className="text-muted-foreground">
            · {formatTokens(it.input_tokens, it.output_tokens)}
          </span>
          <div className="ml-auto">
            <Link
              href={`/conversations/${encodeURIComponent(conversationId)}/interactions/${it.id}`}
              className="text-primary hover:underline"
            >
              View
            </Link>
          </div>
        </div>
        {segments.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No response text saved for this turn (the model may have stopped on a tool call).
          </p>
        ) : (
          <TextSegmentsPlain texts={segments} />
        )}
      </CardContent>
    </Card>
  );
}

// Everything between the user message and the final response — the agentic
// loop (tool results, intermediate model text) plus background side-channel
// calls — folded into one collapsible strip. Expanding shows every raw call.
function ExchangeSteps({
  turns,
  conversationId,
}: {
  turns: ConversationTurn[];
  conversationId: string;
}) {
  const [open, setOpen] = useState(false);
  const background = turns.filter((t) => isBackground(t.interaction.kind)).length;
  const steps = turns.length - background;
  const entities = turns.reduce((n, t) => n + t.interaction.entity_total, 0);
  const parts = [
    steps > 0 ? `${steps} agent step${steps === 1 ? "" : "s"}` : null,
    background > 0 ? `${background} background call${background === 1 ? "" : "s"}` : null,
  ].filter(Boolean);
  return (
    <div className="ml-4 rounded-md border bg-muted/30">
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
        <span className="font-medium">{parts.join(" · ")}</span>
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

// One user → final-response exchange. `turns` is the contiguous run of calls
// starting at a call that carries new user text (if any was saved) up to the
// call before the next one that does.
interface Exchange {
  turns: ConversationTurn[];
}

// Split the ordered calls of a conversation into exchanges. A new exchange
// starts at a main-kind call whose saved texts include a `user` segment — the
// proxy only saves user text that is *novel* on that request (transform.py),
// so this marks exactly the requests where the human said something new
// (including messages queued while the agent was still working). Calls without
// new user text — the agentic tool loop and background side-channels — belong
// to the exchange in progress. Calls before any user-marked one (possible when
// save_texts="anonymized" drops PII-free user text) form a leading exchange
// with no user card.
function splitExchanges(turns: ConversationTurn[]): Exchange[] {
  const exchanges: Exchange[] = [];
  for (const turn of turns) {
    const opensExchange =
      !isBackground(turn.interaction.kind) && turn.texts.some((t) => t.category === "user");
    const last = exchanges[exchanges.length - 1];
    if (opensExchange || !last) {
      exchanges.push({ turns: [turn] });
    } else {
      last.turns.push(turn);
    }
  }
  return exchanges;
}

function ExchangeBlock({
  exchange,
  conversationId,
}: {
  exchange: Exchange;
  conversationId: string;
}) {
  const { turns } = exchange;
  const first = turns[0];
  const hasUserText = first.texts.some((t) => t.category === "user");
  const mains = turns.filter((t) => !isBackground(t.interaction.kind));
  const lastMain = mains[mains.length - 1];
  // The exchange's final answer is the last main-kind call. When every call
  // was classified background (mis-classification, or a genuinely background
  // conversation), fall back to the last call that saved assistant text so
  // the model's answer is never swallowed by the collapsed strip.
  const finalTurn =
    lastMain ??
    [...turns].reverse().find((t) => t.texts.some((x) => x.category === "assistant"));
  // Show the collapsed middle when there is anything beyond the two exposed
  // cards: extra calls, tool data riding along on the opening call, or a
  // background-only run where neither exposed card would render at all.
  const hasMiddle =
    turns.length > 1 ||
    first.texts.some((t) => t.category === "tool") ||
    (!hasUserText && !finalTurn);

  return (
    <div className="space-y-3">
      {hasUserText ? <UserMessageCard turn={first} /> : null}
      {hasMiddle ? <ExchangeSteps turns={turns} conversationId={conversationId} /> : null}
      {finalTurn ? <FinalResponseCard turn={finalTurn} conversationId={conversationId} /> : null}
    </div>
  );
}

// Render the conversation as user → response exchanges: the human's message
// and the model's final answer are exposed; the agentic loop in between is
// collapsed at its position in the thread.
export function ConversationThread({
  turns,
  conversationId,
}: {
  turns: ConversationTurn[];
  conversationId: string;
}) {
  const exchanges = splitExchanges(turns);
  return (
    <div className={cn("space-y-6")}>
      {exchanges.map((exchange) => (
        <ExchangeBlock
          key={exchange.turns[0].interaction.id}
          exchange={exchange}
          conversationId={conversationId}
        />
      ))}
    </div>
  );
}
