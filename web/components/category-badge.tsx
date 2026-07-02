import { Bot, User, Wrench } from "lucide-react";

import type { TextCategory } from "@/lib/types";

// Per-segment origin styling, shared by the texts browser and interaction detail.
// user = text the human typed; tool = data the user's local tools/files produced;
// assistant = the agent's response (stored only when the turn had PII).
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
  assistant: {
    label: "Agent response",
    badge: "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
    accent: "border-l-emerald-500/60",
    title: "Agent response: the model's reply, stored because this turn's prompt contained PII.",
  },
};

const CATEGORY_ICON = { user: User, tool: Wrench, assistant: Bot } as const;

export function CategoryBadge({ category }: { category: TextCategory }) {
  const m = CATEGORY_META[category] ?? CATEGORY_META.user;
  const Icon = CATEGORY_ICON[category] ?? User;
  return (
    <span
      className={
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium " +
        m.badge
      }
      title={m.title}
    >
      <Icon className="h-3 w-3" />
      {m.label}
    </span>
  );
}
