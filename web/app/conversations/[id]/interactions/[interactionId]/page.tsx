import { ArrowLeft } from "lucide-react";
import Link from "next/link";

import { InteractionDetailView } from "@/components/interaction-detail-view";
import { EmptyState, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import type { InteractionDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function ConversationInteractionPage({
  params,
}: {
  params: { id: string; interactionId: string };
}) {
  const backHref = `/conversations/${encodeURIComponent(params.id)}`;
  const { data } = await apiGetOr<InteractionDetail | null>(
    `/interactions/${params.interactionId}`,
    null,
  );

  const back = (
    <Link
      href={backHref}
      className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
    >
      <ArrowLeft className="h-4 w-4" /> Back to conversation
    </Link>
  );

  if (!data) {
    return (
      <>
        {back}
        <EmptyState
          title="Interaction not found"
          description="It may have been deleted, or the gateway is unreachable."
        />
      </>
    );
  }

  const it = data.interaction;

  return (
    <>
      {back}
      <PageHeader
        title={`Turn #${it.id}`}
        description={`${it.source} · ${formatDateTime(it.created_at)}`}
      />
      <InteractionDetailView data={data} />
    </>
  );
}
