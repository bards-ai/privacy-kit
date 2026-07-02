import { ArrowLeft } from "lucide-react";
import Link from "next/link";

import { DeleteInteraction } from "@/components/delete-interaction";
import { InteractionDetailView } from "@/components/interaction-detail-view";
import { EmptyState, PageHeader } from "@/components/ui";
import { apiGetOr } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import type { InteractionDetail } from "@/lib/types";

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

export default async function InteractionDetailPage({ params }: { params: { id: string } }) {
  const { data } = await apiGetOr<InteractionDetail | null>(`/interactions/${params.id}`, null);

  if (!data) {
    return (
      <>
        <BackLink />
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
      <BackLink />
      <PageHeader
        title={`Interaction #${it.id}`}
        description={`${it.source} · ${formatDateTime(it.created_at)}`}
        actions={<DeleteInteraction id={it.id} />}
      />
      <InteractionDetailView data={data} />
    </>
  );
}
