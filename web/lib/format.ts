// Locale-independent formatting so server-rendered and client-rendered output
// always match (no hydration drift).

export function formatDateTime(iso: string): string {
  if (!iso) return "—";
  return iso
    .replace("T", " ")
    .replace(/\.\d+/, "")
    .replace(/(\+00:00|Z)$/, " UTC");
}

export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString("en-US");
}

export function formatTokens(input: number | null, output: number | null): string {
  const i = input === null || input === undefined ? "—" : String(input);
  const o = output === null || output === undefined ? "—" : String(output);
  return `${i} / ${o}`;
}
