import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { ConversationThread } from "@/components/conversation-thread";
import { EntityChips } from "@/components/entity-chips";
import { Card, CardContent, CardHeader, CardTitle, EmptyState, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import { formatDateTime, formatTokens } from "@/lib/format";
import type { ConversationDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

function BackLink() {
  return (
    <Link
      href="/conversations"
      className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
    >
      <ArrowLeft className="h-4 w-4" /> Back to conversations
    </Link>
  );
}

// Aggregate header for the whole conversation.
function SummaryCard({ data }: { data: ConversationDetail }) {
  const s = data.summary;
  const turns = `${s.turn_count}${s.background_count > 0 ? ` (${s.background_count} background)` : ""}`;
  const meta: [string, ReactNode][] = [
    ["Time", `${formatDateTime(s.first_seen)} → ${formatDateTime(s.last_seen)}`],
    ["Turns", turns],
    ["Sources", s.sources.join(", ") || "—"],
    ["Models", s.models.join(", ") || "—"],
    ["Tokens", formatTokens(s.input_tokens, s.output_tokens)],
    ["Entities total", s.entity_total],
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Conversation summary</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-6">
          {meta.map(([k, v]) => (
            <div key={k}>
              <dt className="text-xs uppercase tracking-wide text-muted-foreground">{k}</dt>
              <dd className="mt-1 text-sm font-medium">{v}</dd>
            </div>
          ))}
        </dl>
        <div className="mt-4">
          <span className="text-xs uppercase tracking-wide text-muted-foreground">
            Entity counts
          </span>
          <div className="mt-2">
            <EntityChips counts={s.entity_counts} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default async function ConversationDetailPage({ params }: { params: { id: string } }) {
  const { data } = await apiGetOr<ConversationDetail | null>(
    `/conversations/${encodeURIComponent(params.id)}`,
    null,
  );

  if (!data || data.turns.length === 0) {
    return (
      <>
        <BackLink />
        <EmptyState
          title="Conversation not found"
          description="It may have been deleted, or the gateway is unreachable."
        />
      </>
    );
  }

  const mainCount = data.summary.turn_count - data.summary.background_count;

  return (
    <>
      <BackLink />
      <PageHeader
        title="Conversation"
        description={`${mainCount} turn${mainCount === 1 ? "" : "s"}${
          data.background_count > 0 ? ` + ${data.background_count} background` : ""
        }`}
      />

      <div className="space-y-4">
        <SummaryCard data={data} />

        {data.texts_redacted ? (
          <p className="text-xs text-amber-500">
            Originals are redacted (PII_EXPOSE_PLAINTEXT=false). Showing masked text only.
          </p>
        ) : null}

        <ConversationThread turns={data.turns} conversationId={data.conversation_id} />
      </div>
    </>
  );
}
