"""Local web UI for previewing PII.

Served by the gateway itself at ``/ui`` — no separate process, no build step,
and **no external assets** (a privacy tool must not pull JS from a CDN).

Two panels:

* **Live preview** — paste text, see every detected span highlighted by entity
  type plus the pseudonymized version. The text is processed in memory and
  returned to the caller only: nothing is persisted, logged, or audited.
* **Audit** — totals by entity type and recent interactions from the audit
  store. The store also keeps the request text segments selected by
  ``PII_SAVE_TEXTS`` (original + anonymized); showing them here is a follow-up.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from privacy_kit.core.detectors import Detector
from privacy_kit.core.vault import Vault, anonymize_into
from privacy_kit.gateway.store import AuditStore

_MAX_PREVIEW_CHARS = 50_000


def register_ui_routes(app: FastAPI, *, detector: Detector, store: AuditStore) -> None:
    """Mount the preview UI and its JSON endpoints onto ``app``."""

    @app.get("/ui", include_in_schema=False)
    async def ui_page() -> HTMLResponse:
        return HTMLResponse(_PAGE)

    @app.post("/ui/api/preview")
    async def ui_preview(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "expected a JSON body"}, status_code=400)
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            return JSONResponse({"error": "expected {'text': <string>}"}, status_code=400)
        if len(text) > _MAX_PREVIEW_CHARS:
            return JSONResponse(
                {"error": f"text too long (max {_MAX_PREVIEW_CHARS} characters)"},
                status_code=413,
            )

        def run() -> dict[str, Any]:
            vault = Vault()
            spans = detector.detect(text)
            anonymized = anonymize_into(text, detector, vault)
            return {
                "spans": [
                    {"start": s.start, "end": s.end, "label": s.label, "score": s.score}
                    for s in spans
                ],
                "anonymized": anonymized,
                "counts": vault.type_counts,
            }

        # CPU-bound inference off the event loop; the result goes back to the
        # local caller only — deliberately no audit row and no logging.
        return JSONResponse(await run_in_threadpool(run))

    @app.get("/ui/api/summary")
    async def ui_summary() -> JSONResponse:
        return JSONResponse(store.summary())

    @app.get("/ui/api/recent")
    async def ui_recent(limit: int = 50) -> JSONResponse:
        rows = [
            {
                "created_at": row.created_at.isoformat(timespec="seconds"),
                "source": row.source,
                "wire_format": row.wire_format,
                "model": row.model,
                "entity_total": row.entity_total,
                "entity_counts": row.entity_counts,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
            }
            for row in store.recent(limit=min(limit, 200))
        ]
        return JSONResponse({"interactions": rows})

    @app.get("/ui/api/texts")
    async def ui_texts(limit: int = 50) -> JSONResponse:
        interactions = store.recent(limit=min(limit, 200))
        all_texts = []
        for interaction in interactions:
            text_rows = store.texts(interaction.id)
            for text_row in text_rows:
                all_texts.append({
                    "when": interaction.created_at.isoformat(timespec="seconds"),
                    "original": text_row.original,
                    "anonymized": text_row.anonymized,
                })
        return JSONResponse({"texts": all_texts})


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>privacy-kit</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #101418; color: #dde3ea; }
  header { display: flex; align-items: baseline; gap: 1rem; padding: 1rem 1.5rem;
           border-bottom: 1px solid #232a32; }
  header h1 { font-size: 1.1rem; margin: 0; }
  header .sub { color: #7d8a99; font-size: .85rem; }
  nav button { background: none; border: none; color: #7d8a99; font: inherit; padding: .3rem .8rem;
               cursor: pointer; border-radius: 6px; }
  nav button.active { color: #dde3ea; background: #1c232b; }
  main { max-width: 980px; margin: 0 auto; padding: 1.5rem; }
  textarea { width: 100%; min-height: 9rem; background: #161c22; color: #dde3ea;
             border: 1px solid #2a323c; border-radius: 8px; padding: .8rem; font: inherit; resize: vertical; }
  button.primary { margin-top: .6rem; background: #2f6feb; color: #fff; border: none;
                   border-radius: 8px; padding: .55rem 1.2rem; font: inherit; cursor: pointer; }
  button.primary:disabled { opacity: .5; cursor: wait; }
  .card { background: #161c22; border: 1px solid #2a323c; border-radius: 10px;
          padding: 1rem 1.2rem; margin-top: 1.2rem; }
  .card h2 { margin: 0 0 .6rem; font-size: .8rem; text-transform: uppercase;
             letter-spacing: .08em; color: #7d8a99; }
  .rendered { white-space: pre-wrap; word-break: break-word; }
  mark { border-radius: 4px; padding: 0 3px; color: #08090b; font-weight: 600; }
  mark small { font-weight: 400; font-size: .7em; opacity: .8; margin-left: 3px; }
  .chips { display: flex; flex-wrap: wrap; gap: .5rem; }
  .chip { border-radius: 999px; padding: .15rem .7rem; font-size: .8rem; color: #08090b; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: .88rem; }
  th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #232a32; }
  th { color: #7d8a99; font-weight: 500; }
  .muted { color: #7d8a99; }
  .err { color: #ff7b72; }
  .bar { height: 8px; border-radius: 4px; background: #2f6feb; }
  .hidden { display: none; }
  .text-column { white-space: pre-wrap; word-break: break-word; max-width: 350px; }
</style>
</head>
<body>
<header>
  <h1>privacy-kit</h1>
  <span class="sub">on-device PII gateway — nothing on this page leaves your machine</span>
  <nav style="margin-left:auto">
    <button id="tab-preview" class="active">Live preview</button>
    <button id="tab-audit">Audit</button>
  </nav>
</header>
<main>
  <section id="view-preview">
    <p class="muted">Paste anything — a prompt, a log line, a document. Detection runs locally;
    the text is not stored, logged, or audited.</p>
    <textarea id="input" placeholder="Hi, I'm Jan Kowalski, jan.kowalski@example.com, +48 501 222 333…"></textarea>
    <button class="primary" id="scan">Detect PII</button>
    <span id="status" class="muted"></span>
    <div class="card hidden" id="card-counts"><h2>Detected</h2><div class="chips" id="chips"></div></div>
    <div class="card hidden" id="card-highlight"><h2>Your text, highlighted</h2><div class="rendered" id="highlighted"></div></div>
    <div class="card hidden" id="card-anon"><h2>What would leave the machine (pseudonymized)</h2><div class="rendered" id="anonymized"></div></div>
  </section>

  <section id="view-audit" class="hidden">
    <p class="muted">Entity types and counts per interaction. The store also saves the text segments selected by PII_SAVE_TEXTS (original + anonymized).</p>
    <div class="card"><h2>Totals</h2><div id="totals" class="muted">loading…</div>
      <table id="by-type-table" class="hidden"><thead><tr><th>Entity type</th><th>Count</th><th></th></tr></thead><tbody id="by-type"></tbody></table>
    </div>
    <div class="card"><h2>Recent interactions</h2>
      <table><thead><tr><th>When</th><th>Source</th><th>Wire</th><th>Model</th><th>Entities</th><th>Tokens in/out</th></tr></thead>
      <tbody id="recent"></tbody></table>
    </div>
    <div class="card"><h2>Text segments (before & after)</h2>
      <table id="texts-table" class="hidden"><thead><tr><th>When</th><th>Original</th><th>Anonymized</th></tr></thead>
      <tbody id="text-segments"></tbody></table>
      <div id="texts-empty" class="muted">No text segments saved.</div>
    </div>
  </section>
</main>
<script>
"use strict";
const $ = (id) => document.getElementById(id);

function colorFor(label) {
  let h = 0;
  for (const c of label) h = (h * 31 + c.charCodeAt(0)) % 360;
  return `hsl(${h} 70% 70%)`;
}

function show(id, on) { $(id).classList.toggle("hidden", !on); }

// --- tabs -------------------------------------------------------------
function switchTab(name) {
  show("view-preview", name === "preview");
  show("view-audit", name === "audit");
  $("tab-preview").classList.toggle("active", name === "preview");
  $("tab-audit").classList.toggle("active", name === "audit");
  if (name === "audit") loadAudit();
}
$("tab-preview").onclick = () => switchTab("preview");
$("tab-audit").onclick = () => switchTab("audit");

// --- live preview ------------------------------------------------------
function renderHighlighted(target, text, spans) {
  target.replaceChildren();
  let cursor = 0;
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    if (s.start > cursor) target.append(text.slice(cursor, s.start));
    const m = document.createElement("mark");
    m.style.background = colorFor(s.label);
    m.append(text.slice(s.start, s.end));
    const tag = document.createElement("small");
    tag.textContent = `${s.label} ${(s.score * 100).toFixed(0)}%`;
    m.append(tag);
    target.append(m);
    cursor = s.end;
  }
  target.append(text.slice(cursor));
}

$("scan").onclick = async () => {
  const text = $("input").value;
  if (!text.trim()) return;
  $("scan").disabled = true;
  $("status").textContent = "detecting…";
  $("status").className = "muted";
  try {
    const resp = await fetch("/ui/api/preview", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) throw new Error((await resp.json()).error || resp.statusText);
    const data = await resp.json();
    $("status").textContent = data.spans.length
      ? `${data.spans.length} span(s) found`
      : "no PII detected";

    $("chips").replaceChildren();
    for (const [label, n] of Object.entries(data.counts)) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.style.background = colorFor(label);
      chip.textContent = `${label} × ${n}`;
      $("chips").append(chip);
    }
    show("card-counts", Object.keys(data.counts).length > 0);
    renderHighlighted($("highlighted"), text, data.spans);
    show("card-highlight", true);
    $("anonymized").textContent = data.anonymized;
    show("card-anon", true);
  } catch (e) {
    $("status").textContent = String(e.message || e);
    $("status").className = "err";
  } finally {
    $("scan").disabled = false;
  }
};

// --- audit --------------------------------------------------------------
async function loadAudit() {
  try {
    const [summary, recent, texts] = await Promise.all([
      fetch("/ui/api/summary").then((r) => r.json()),
      fetch("/ui/api/recent").then((r) => r.json()),
      fetch("/ui/api/texts").then((r) => r.json()),
    ]);
    $("totals").textContent =
      `${summary.interactions} interaction(s), ${summary.entities_total} PII entit(ies) caught`;
    const byType = Object.entries(summary.entities_by_type).sort((a, b) => b[1] - a[1]);
    const max = byType.length ? byType[0][1] : 1;
    $("by-type").replaceChildren();
    for (const [label, n] of byType) {
      const tr = document.createElement("tr");
      const tdL = document.createElement("td"); tdL.textContent = label;
      const tdN = document.createElement("td"); tdN.textContent = n;
      const tdB = document.createElement("td"); tdB.style.width = "50%";
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.style.width = `${Math.max(4, (n / max) * 100)}%`;
      bar.style.background = colorFor(label);
      tdB.append(bar);
      tr.append(tdL, tdN, tdB);
      $("by-type").append(tr);
    }
    show("by-type-table", byType.length > 0);

    $("recent").replaceChildren();
    for (const row of recent.interactions) {
      const tr = document.createElement("tr");
      const cells = [
        row.created_at.replace("T", " "),
        row.source,
        row.wire_format,
        row.model,
        Object.entries(row.entity_counts).map(([k, v]) => `${k}×${v}`).join(", ") || "—",
        `${row.input_tokens ?? "—"} / ${row.output_tokens ?? "—"}`,
      ];
      for (const c of cells) {
        const td = document.createElement("td");
        td.textContent = c;
        tr.append(td);
      }
      $("recent").append(tr);
    }

    $("text-segments").replaceChildren();
    if (texts.texts.length === 0) {
      show("texts-table", false);
      show("texts-empty", true);
    } else {
      for (const row of texts.texts) {
        const tr = document.createElement("tr");
        const tdW = document.createElement("td"); tdW.textContent = row.when.replace("T", " ");
        const tdO = document.createElement("td"); tdO.className = "text-column"; tdO.textContent = row.original;
        const tdA = document.createElement("td"); tdA.className = "text-column"; tdA.textContent = row.anonymized;
        tr.append(tdW, tdO, tdA);
        $("text-segments").append(tr);
      }
      show("texts-table", true);
      show("texts-empty", false);
    }
  } catch (e) {
    $("totals").textContent = String(e.message || e);
    $("totals").className = "err";
  }
}
</script>
</body>
</html>
"""
