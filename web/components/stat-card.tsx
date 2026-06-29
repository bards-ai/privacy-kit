import type { ReactNode } from "react";

import { cn } from "@/lib/cn";
import { Card } from "@/components/ui";

export function StatCard({
  label,
  value,
  sub,
  icon,
  accent = "default",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  icon?: ReactNode;
  accent?: "default" | "warning" | "success";
}) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        {icon ? <span className="text-muted-foreground">{icon}</span> : null}
      </div>
      <div
        className={cn(
          "mt-2 text-2xl font-semibold tabular-nums",
          accent === "warning" && "text-amber-500",
          accent === "success" && "text-emerald-500",
        )}
      >
        {value}
      </div>
      {sub ? <div className="mt-1 text-xs text-muted-foreground">{sub}</div> : null}
    </Card>
  );
}
