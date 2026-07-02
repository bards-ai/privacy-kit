function hslToHex(h: number, s: number, l: number): string {
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const v = l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
    return Math.round(255 * v)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

// Deterministic color per entity type, so a given label looks the same
// everywhere (chips, charts, highlights).
export function entityColor(label: string): string {
  let h = 0;
  
  for (const c of label) h = (h * 31 + c.charCodeAt(0)) % 360;
  return hslToHex(h, 0.68, 0.55);
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
