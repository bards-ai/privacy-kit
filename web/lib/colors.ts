// Deterministic color per entity type, so a given label looks the same
// everywhere (chips, charts, highlights).
export function entityColor(label: string): string {
  let h = 0;
  for (const c of label) h = (h * 31 + c.charCodeAt(0)) % 360;
  return `hsl(${h} 68% 55%)`;
}

// A small categorical palette for charts that aren't keyed by entity type.
export const CHART_PALETTE = [
  "hsl(217 91% 60%)",
  "hsl(160 84% 39%)",
  "hsl(38 92% 50%)",
  "hsl(280 65% 60%)",
  "hsl(346 84% 61%)",
  "hsl(199 89% 48%)",
  "hsl(125 50% 50%)",
  "hsl(20 90% 55%)",
];
