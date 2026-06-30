import { User, Wrench } from "lucide-react";

import type { TextCategory } from "@/lib/types";

// Per-segment origin styling, shared by the texts browser and interaction detail.
// user = text the human typed; tool = data the user's local tools/files produced.
export const CATEGORY_META: Record<
  TextCategory,
  { label: string; badge: string; accent: string; title: string }
> = {
  user: {
    label: "User text",
    badge: "border-blue-500/30 bg-blue-500/10 text-blue-600 dark:text-blue-400",
    accent: "border-l-blue-500/60",
    title: "User text: a message the human typed.",
  },
  tool: {
    label: "Tool data",
    badge: "border-violet-500/30 bg-violet-500/10 text-violet-600 dark:text-violet-400",
    accent: "border-l-violet-500/60",
    title: "Tool data: output from the user's local tools or files (e.g. file reads, command results).",
  },
};

export function CategoryBadge({ category }: { category: TextCategory }) {
  const m = CATEGORY_META[category] ?? CATEGORY_META.user;
  return (
    <span
      className={
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium " +
        m.badge
      }
      title={m.title}
    >
      {category === "tool" ? <Wrench className="h-3 w-3" /> : <User className="h-3 w-3" />}
      {m.label}
    </span>
  );
}
