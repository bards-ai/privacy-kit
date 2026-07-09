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

// One piece of a hunked segment. `text` is shown inline — its parts may include
// PII spans and literals, and literals may span newlines. `gap` is a collapsed
// PII-free stretch behind an expander; `lineShaped` records whether it starts
// and ends on line boundaries (so the caller can label it "N lines hidden") vs.
// a mid-line cut through a very long line/paragraph ("N characters hidden").
export interface HunkText {
  kind: "text";
  parts: PiiPart[];
}
export interface HunkGap {
  kind: "gap";
  parts: PiiPart[]; // the hidden content, revealed on expand
  chars: number;
  lines: number;
  lineShaped: boolean;
}
export type HunkPiece = HunkText | HunkGap;

// Characters of context kept on each side of a PII span; how far past a boundary
// we look for a newline to snap to; and the smallest stretch worth collapsing.
const HUNK_CONTEXT = 120;
const HUNK_SNAP = 200;
const HUNK_MIN_GAP = 80;

function countNewlines(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) if (s.charCodeAt(i) === 10) n++;
  return n;
}

// Collapse the PII-free stretches of an aligned segment, keeping ~HUNK_CONTEXT
// chars of context around each detected PII span. Collapse boundaries snap to a
// nearby newline when there is one, so content with many short lines collapses
// on line boundaries (git-diff style, "N lines hidden") while a single very long
// line/paragraph collapses mid-line ("N characters hidden") — the case pure
// line-based hunking cannot handle. Each literal in `aligned.parts` is already
// the maximal run between two PII anchors (or the segment start/end), so we can
// decide the cut per literal.
export function hunkify(aligned: Aligned): {
  pieces: HunkPiece[];
  piiCount: number;
  lineCount: number;
} {
  const parts = aligned.parts;
  const piiCount = parts.reduce((n, p) => n + (p.kind === "pii" ? 1 : 0), 0);
  const lineCount =
    parts.reduce((n, p) => n + (p.kind === "lit" ? countNewlines(p.text) : 0), 0) + 1;

  const pieces: HunkPiece[] = [];
  let buffer: PiiPart[] = [];
  const flush = () => {
    if (buffer.length) {
      pieces.push({ kind: "text", parts: buffer });
      buffer = [];
    }
  };

  let piiSeen = 0;
  for (const part of parts) {
    if (part.kind === "pii") {
      buffer.push(part);
      piiSeen += 1;
      continue;
    }
    const L = part.text;
    // No PII on the left (segment start / no PII yet) means no left context to
    // keep — the gap can start at this literal's start; likewise on the right.
    const keepLeft = piiSeen > 0 ? HUNK_CONTEXT : 0;
    const keepRight = piiCount - piiSeen > 0 ? HUNK_CONTEXT : 0;
    if (L.length <= keepLeft + keepRight + HUNK_MIN_GAP) {
      buffer.push(part);
      continue;
    }

    // Head: keep `keepLeft` chars after the previous PII, extended to the next
    // newline when one is within HUNK_SNAP so short lines stay whole.
    let head = keepLeft;
    let headSnapped = keepLeft === 0;
    const nlAfter = L.indexOf("\n", keepLeft);
    if (nlAfter !== -1 && nlAfter < keepLeft + HUNK_SNAP) {
      head = nlAfter + 1;
      headSnapped = true;
    }
    // Tail: keep `keepRight` chars before the next PII, retracted to a preceding
    // newline when one is within HUNK_SNAP.
    let tail = L.length - keepRight;
    let tailSnapped = keepRight === 0;
    const nlBefore = L.lastIndexOf("\n", tail - 1);
    if (nlBefore !== -1 && nlBefore >= tail - HUNK_SNAP) {
      tail = nlBefore + 1;
      tailSnapped = true;
    }
    if (tail - head < HUNK_MIN_GAP) {
      buffer.push(part);
      continue;
    }

    if (head > 0) buffer.push({ kind: "lit", text: L.slice(0, head) });
    flush();
    const gapText = L.slice(head, tail);
    pieces.push({
      kind: "gap",
      parts: [{ kind: "lit", text: gapText }],
      chars: gapText.length,
      lines: countNewlines(gapText),
      lineShaped: headSnapped && tailSnapped,
    });
    const rest = L.slice(tail);
    if (rest.length) buffer.push({ kind: "lit", text: rest });
  }
  flush();

  return { pieces, piiCount, lineCount };
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
