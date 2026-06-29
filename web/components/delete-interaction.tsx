"use client";

import { Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/button";

export function DeleteInteraction({ id }: { id: number }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  async function onDelete() {
    if (!window.confirm("Delete this interaction and its saved text? This cannot be undone.")) {
      return;
    }
    setBusy(true);
    try {
      const res = await fetch(`/api/pk/interactions/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`Delete failed (${res.status})`);
      router.push("/interactions");
      router.refresh();
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }

  return (
    <Button variant="danger" size="sm" onClick={onDelete} disabled={busy}>
      <Trash2 className="h-4 w-4" />
      {busy ? "Deleting…" : "Delete"}
    </Button>
  );
}
