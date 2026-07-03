// Recover which substrings were masked by diffing an interaction's saved
// `original` against its `anonymized` counterpart.
//
// The gateway replaces each detected PII value with a `[TYPE_N]` placeholder
// (see core/vault.py). Because the surrounding text is otherwise identical, the
// literal chunks between placeholders anchor a left-to-right scan that recovers
// the original value each placeholder stood in for — no spans are persisted, so
// this is derived on the client from the two strings the API already returns.

export interface PiiSpan {
  type: string; // entity type, e.g. "PERSON_NAME"
  placeholder: string; // full token, e.g. "[PERSON_NAME_1]"
  value: string | null; // recovered original, or null when not derivable/redacted
}

export type PiiPart = { kind: "lit"; text: string } | { kind: "pii"; span: PiiSpan };

export interface Aligned {
  parts: PiiPart[]; // ordered literal/placeholder tokens of the anonymized text
  ok: boolean; // every placeholder mapped back to a value
  hasOriginal: boolean; // an original was available to align against
}

// Type names are uppercase words joined by underscores; the trailing `_<n>` is
// the per-request occurrence index. Greedy `[A-Z_]+` backtracks to leave `_\d+]`.
const PLACEHOLDER = /\[([A-Z_]+)_(\d+)\]/g;

export function alignPii(original: string | null, anonymized: string): Aligned {
  // Tokenize the anonymized text into literal chunks and placeholder spans.
  const parts: PiiPart[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  PLACEHOLDER.lastIndex = 0;
  while ((m = PLACEHOLDER.exec(anonymized)) !== null) {
    if (m.index > last) parts.push({ kind: "lit", text: anonymized.slice(last, m.index) });
    parts.push({ kind: "pii", span: { type: m[1], placeholder: m[0], value: null } });
    last = m.index + m[0].length;
  }
  if (last < anonymized.length) parts.push({ kind: "lit", text: anonymized.slice(last) });

  const spans = parts.filter((p) => p.kind === "pii").length;
  if (original === null || spans === 0) {
    return { parts, ok: spans === 0, hasOriginal: original !== null };
  }

  // Anchored scan: each literal chunk also exists verbatim in the original, in
  // order; the gap before a literal match is the value of the placeholder that
  // preceded it. A trailing placeholder takes whatever original text remains.
  let oi = 0;
  let pending: PiiSpan | null = null;
  let ok = true;
  for (const part of parts) {
    if (part.kind === "lit") {
      if (part.text.length === 0) {
        // Adjacent placeholders leave no anchor between them — unrecoverable.
        if (pending) {
          ok = false;
          pending = null;
        }
        continue;
      }
      const idx = original.indexOf(part.text, oi);
      if (idx === -1) {
        ok = false; // alignment drifted; stop assigning values
        break;
      }
      if (pending) {
        pending.value = original.slice(oi, idx);
        pending = null;
      }
      oi = idx + part.text.length;
    } else {
      if (pending) ok = false; // two placeholders in a row; previous stays null
      pending = part.span;
    }
  }
  if (pending) pending.value = original.slice(oi);

  if (parts.some((p) => p.kind === "pii" && p.span.value === null)) ok = false;
  return { parts, ok, hasOriginal: true };
}

// A single logical line of an aligned segment: the ordered parts that fall on
// it, and whether any of them is a PII span.
export interface AlignedLine {
  parts: PiiPart[];
  hasPii: boolean;
}

// A run of consecutive lines. `lines` blocks are shown; `gap` blocks are the
// PII-free stretches collapsed behind a "N lines hidden" expander (their lines
// are retained so expanding can reveal them in place).
export type HunkBlock =
  | { kind: "lines"; lines: AlignedLine[] }
  | { kind: "gap"; count: number; lines: AlignedLine[] };

// Split an aligned segment into per-line blocks so a caller can show only the
// neighborhoods around detected PII (git-diff style) and collapse the rest.
// `context` is how many PII-free lines to keep on each side of a PII line.
export function hunkify(
  aligned: Aligned,
  context = 2,
): { blocks: HunkBlock[]; piiCount: number; lineCount: number } {
  // Walk the parts, splitting literal chunks on newlines into lines. A PII span
  // is attached to the line it starts on (spans never contain a newline).
  const lines: AlignedLine[] = [];
  let current: PiiPart[] = [];
  let currentPii = false;
  const flush = () => {
    lines.push({ parts: current, hasPii: currentPii });
    current = [];
    currentPii = false;
  };
  for (const part of aligned.parts) {
    if (part.kind === "lit") {
      const pieces = part.text.split("\n");
      pieces.forEach((piece, i) => {
        if (i > 0) flush();
        if (piece.length > 0) current.push({ kind: "lit", text: piece });
      });
    } else {
      current.push(part);
      currentPii = true;
    }
  }
  flush();

  const piiCount = aligned.parts.filter((p) => p.kind === "pii").length;

  // No PII at all: one gap covering everything, so the caller can render a
  // single collapsed "no PII" strip.
  if (piiCount === 0) {
    return {
      blocks: [{ kind: "gap", count: lines.length, lines }],
      piiCount,
      lineCount: lines.length,
    };
  }

  // Mark lines within `context` of any PII line as visible.
  const visible = new Array<boolean>(lines.length).fill(false);
  lines.forEach((line, i) => {
    if (!line.hasPii) return;
    for (let j = Math.max(0, i - context); j <= Math.min(lines.length - 1, i + context); j++) {
      visible[j] = true;
    }
  });

  // Coalesce consecutive lines of the same visibility into blocks.
  const blocks: HunkBlock[] = [];
  let i = 0;
  while (i < lines.length) {
    const vis = visible[i];
    let j = i;
    while (j < lines.length && visible[j] === vis) j++;
    const run = lines.slice(i, j);
    blocks.push(vis ? { kind: "lines", lines: run } : { kind: "gap", count: run.length, lines: run });
    i = j;
  }
  return { blocks, piiCount, lineCount: lines.length };
}

// Distinct masked values per entity type across several aligned segments, keyed
// by placeholder (the same value reuses one placeholder within a request).
export function valuesByType(aligneds: Aligned[]): Map<string, PiiSpan[]> {
  const byType = new Map<string, Map<string, PiiSpan>>();
  for (const a of aligneds) {
    for (const p of a.parts) {
      if (p.kind !== "pii") continue;
      let seen = byType.get(p.span.type);
      if (!seen) {
        seen = new Map();
        byType.set(p.span.type, seen);
      }
      if (!seen.has(p.span.placeholder)) seen.set(p.span.placeholder, p.span);
    }
  }
  const out = new Map<string, PiiSpan[]>();
  for (const [type, seen] of byType) {
    out.set(
      type,
      [...seen.values()].sort((a, b) =>
        a.placeholder.localeCompare(b.placeholder, undefined, { numeric: true }),
      ),
    );
  }
  return out;
}
