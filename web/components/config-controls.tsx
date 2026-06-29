"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { cn } from "@/lib/cn";
import type { Policy, SaveTexts } from "@/lib/types";

const fieldClass =
  "h-9 rounded-md border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50";

// Shared client-side writer for the runtime-editable gateway settings. All
// controls go through PATCH /api/pk/config (proxied to the gateway), which is the
// single choke point that a future multi-user build can gate behind admin auth.
function usePatchConfig() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function patch(body: Record<string, unknown>): Promise<boolean> {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/pk/config", {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.error ?? `Update failed (${res.status})`);
      }
      router.refresh();
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }

  return { patch, busy, error };
}

function Field({ error, children }: { error: string | null; children: React.ReactNode }) {
  return (
    <span className="inline-flex flex-col items-end gap-1">
      {children}
      {error && <span className="text-xs text-red-500">{error}</span>}
    </span>
  );
}

const POLICY_OPTIONS: { value: Policy; label: string }[] = [
  { value: "monitor", label: "Monitor (forward unchanged, log only)" },
  { value: "pseudonymize", label: "Pseudonymize (replace PII before forwarding)" },
];

// Runtime policy switch. The proxy reads the policy per request, so the change
// applies to subsequent traffic; it is in-memory and resets to PII_POLICY on
// gateway restart.
export function PolicySelect({ policy }: { policy: Policy }) {
  const { patch, busy, error } = usePatchConfig();
  return (
    <Field error={error}>
      <select
        value={policy}
        disabled={busy}
        onChange={(e) => e.target.value !== policy && patch({ policy: e.target.value })}
        className={cn(fieldClass)}
        aria-label="Policy"
      >
        {POLICY_OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

const SAVE_TEXTS_OPTIONS: { value: SaveTexts; label: string }[] = [
  { value: "all", label: "All eligible segments" },
  { value: "anonymized", label: "Only segments with detected PII" },
];

// Which eligible request text segments are persisted (in-memory; resets to
// PII_SAVE_TEXTS on restart).
export function SaveTextsSelect({ value }: { value: SaveTexts }) {
  const { patch, busy, error } = usePatchConfig();
  return (
    <Field error={error}>
      <select
        value={value}
        disabled={busy}
        onChange={(e) => e.target.value !== value && patch({ save_texts: e.target.value })}
        className={cn(fieldClass)}
        aria-label="Save texts"
      >
        {SAVE_TEXTS_OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}

// Detection confidence cut-off (0.0–1.0). Updates the live detector, so it
// affects subsequent detections; in-memory and resets to PII_THRESHOLD on
// restart. Commits on blur / Enter to avoid a request per keystroke.
export function ThresholdInput({ value }: { value: number }) {
  const { patch, busy, error } = usePatchConfig();
  const [draft, setDraft] = useState(String(value));

  async function commit() {
    const next = Number(draft);
    if (draft.trim() === "" || Number.isNaN(next) || next < 0 || next > 1) {
      setDraft(String(value));
      return;
    }
    if (next === value) return;
    const ok = await patch({ threshold: next });
    if (!ok) setDraft(String(value));
  }

  return (
    <Field error={error}>
      <input
        type="number"
        inputMode="decimal"
        min={0}
        max={1}
        step={0.05}
        value={draft}
        disabled={busy}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
        className={cn(fieldClass, "w-24 text-right")}
        aria-label="Detection threshold"
      />
    </Field>
  );
}
