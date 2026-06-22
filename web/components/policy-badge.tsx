import { ShieldAlert, ShieldCheck } from "lucide-react";

// Monitor = real PII reached the upstream (logged only) → warn. Pseudonymize =
// PII was replaced before forwarding → safe.
export function PolicyBadge({ policy }: { policy: string }) {
  const monitor = policy === "monitor";
  return (
    <span
      className={
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium " +
        (monitor
          ? "border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400"
          : "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400")
      }
      title={
        monitor
          ? "Monitor: the prompt was forwarded unchanged — real PII reached the upstream (detection logged only)."
          : "Pseudonymize: PII was replaced with placeholders before forwarding."
      }
    >
      {monitor ? <ShieldAlert className="h-3 w-3" /> : <ShieldCheck className="h-3 w-3" />}
      {policy}
    </span>
  );
}
