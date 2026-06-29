"use client";

import { Search } from "lucide-react";
import { type ReactNode, useState } from "react";

import { Button } from "@/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui";
import { entityColor } from "@/lib/colors";
import type { PreviewResult } from "@/lib/types";

function renderHighlighted(text: string, spans: PreviewResult["spans"]): ReactNode[] {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const nodes: ReactNode[] = [];
  let cursor = 0;
  sorted.forEach((s, i) => {
    if (s.start > cursor) nodes.push(<span key={`t${i}`}>{text.slice(cursor, s.start)}</span>);
    nodes.push(
      <mark
        key={`m${i}`}
        className="rounded px-1 font-medium"
        style={{ backgroundColor: `${entityColor(s.label)}33`, color: entityColor(s.label) }}
      >
        {text.slice(s.start, s.end)}
        <span className="ml-1 text-[10px] opacity-70">
          {s.label} {(s.score * 100).toFixed(0)}%
        </span>
      </mark>,
    );
    cursor = s.end;
  });
  if (cursor < text.length) nodes.push(<span key="tend">{text.slice(cursor)}</span>);
  return nodes;
}

export function PreviewTool() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<PreviewResult | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function detect() {
    if (!text.trim()) return;
    setBusy(true);
    setError(null);
    setStatus("Detecting…");
    try {
      const res = await fetch("/api/pk/preview", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
      setResult(data as PreviewResult);
      setStatus(data.spans.length ? `${data.spans.length} span(s) found` : "No PII detected");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("");
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const counts = result ? Object.entries(result.counts) : [];

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-5">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Hi, I'm Jan Kowalski, jan.kowalski@example.com, +48 501 222 333…"
            className="min-h-[140px] w-full resize-y rounded-md border bg-background p-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          <div className="mt-3 flex items-center gap-3">
            <Button onClick={detect} disabled={busy || !text.trim()}>
              <Search className="h-4 w-4" />
              {busy ? "Detecting…" : "Detect PII"}
            </Button>
            <span className={error ? "text-sm text-red-500" : "text-sm text-muted-foreground"}>
              {error ?? status}
            </span>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Runs on-device. Nothing typed here is stored, logged, or audited.
          </p>
        </CardContent>
      </Card>

      {counts.length > 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>Detected</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {counts.map(([label, n]) => (
                <span
                  key={label}
                  className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium"
                  style={{ backgroundColor: `${entityColor(label)}22`, color: entityColor(label) }}
                >
                  {label} ×{n}
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      ) : null}

      {result ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Your text, highlighted</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                {renderHighlighted(text, result.spans)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Pseudonymized (what would leave the machine)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="whitespace-pre-wrap break-words font-mono text-sm leading-relaxed">
                {result.anonymized}
              </div>
            </CardContent>
          </Card>
        </div>
      ) : null}
    </div>
  );
}
