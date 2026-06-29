import { entityColor } from "@/lib/colors";

export function EntityChips({
  counts,
  max,
}: {
  counts: Record<string, number>;
  max?: number;
}) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) return <span className="text-muted-foreground">—</span>;
  const shown = max ? entries.slice(0, max) : entries;
  const rest = max ? entries.length - shown.length : 0;
  return (
    <div className="flex flex-wrap gap-1">
      {shown.map(([label, n]) => (
        <span
          key={label}
          className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium"
          style={{ backgroundColor: `${entityColor(label)}22`, color: entityColor(label) }}
        >
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: entityColor(label) }}
          />
          {label}
          <span className="opacity-70">×{n}</span>
        </span>
      ))}
      {rest > 0 ? <span className="text-xs text-muted-foreground">+{rest}</span> : null}
    </div>
  );
}
