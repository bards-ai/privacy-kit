// Presentation metadata for a call's purpose bucket (see proxy/classify.py).
// `main` is the real agent conversation; the rest are background side-channels
// that the interactions list folds away so they don't crowd the conversation.
export interface KindMeta {
  label: string; // full description for group headers
  short: string; // compact badge text
  color: string;
  background: boolean; // collapse into a group rather than list inline
}

const KIND_META: Record<string, KindMeta> = {
  main: { label: "Main conversation", short: "Main", color: "hsl(217 91% 60%)", background: false },
  safety: {
    label: "Safety classifier",
    short: "Safety",
    color: "hsl(38 92% 50%)",
    background: true,
  },
  helper: {
    label: "Title / topic helper",
    short: "Helper",
    color: "hsl(280 65% 60%)",
    background: true,
  },
};

export function kindMeta(kind: string): KindMeta {
  return (
    KIND_META[kind] ?? { label: kind, short: kind, color: "hsl(215 16% 47%)", background: true }
  );
}

export function isBackground(kind: string | undefined): boolean {
  return kindMeta(kind || "main").background;
}
