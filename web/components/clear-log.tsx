"use client";

import { Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/button";

export function ClearLog() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  async function onClear() {
    const ok = window.confirm(
      "Permanently delete ALL audit data (every interaction, detection, and saved text segment)? This cannot be undone.",
    );
    if (!ok) return;
    setBusy(true);
    try {
      const res = await fetch("/api/pk/audit/clear", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ confirm: true }),
      });
      if (!res.ok) throw new Error(`Clear failed (${res.status})`);
      router.refresh();
    } catch (e) {
      window.alert(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Button variant="danger" size="sm" onClick={onClear} disabled={busy}>
      <Trash2 className="h-4 w-4" />
      {busy ? "Clearing…" : "Clear audit log"}
    </Button>
  );
}
