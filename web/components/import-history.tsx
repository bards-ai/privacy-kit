"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, ChevronRight, DownloadCloud } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/button";
import { Badge } from "@/components/ui";
import { clientGet } from "@/lib/client-api";
import { cn } from "@/lib/cn";
import { formatDateTime, formatNumber } from "@/lib/format";
import type {
  ImportPreview,
  ImportRequest,
  ImportSessionsPreview,
  ImportSource,
  ImportStatus,
} from "@/lib/types";

const SOURCE_LABELS: Record<ImportSource, string> = {
  "claude-code": "Claude Code (~/.claude/projects)",
  codex: "Codex (~/.codex/sessions)",
};

const inputClass =
  "h-9 rounded-md border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring";

function todayStr() {
  const d = new Date();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${mm}-${dd}`;
}

export function ImportHistory() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<ImportSource[]>(["claude-code", "codex"]);
  const [since, setSince] = useState("");
  // A date-only "until" covers its whole day, so today ≡ now.
  const [until, setUntil] = useState(todayStr);
  const [project, setProject] = useState("");
  const [projectDraft, setProjectDraft] = useState("");
  const [dryRun, setDryRun] = useState(false);
  const [showList, setShowList] = useState(false);
  // Sessions the user unchecked in the list; sent as an exclude list so
  // sessions beyond the list's display cap stay included.
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const previewQs = new URLSearchParams();
  if (since) previewQs.set("since", since);
  if (until) previewQs.set("until", until);
  if (project) previewQs.set("project", project);
  const qs = previewQs.toString();
  const { data: preview } = useQuery({
    queryKey: ["import", "preview", since, until, project],
    queryFn: () => clientGet<ImportPreview>(`/import/preview${qs ? `?${qs}` : ""}`),
  });
  const { data: status } = useQuery({
    queryKey: ["import", "status"],
    queryFn: () => clientGet<ImportStatus>("/import/status"),
    refetchInterval: (query) => (query.state.data?.state === "running" ? 1000 : false),
  });

  // The session list opens every discovered file server-side, so it is only
  // fetched while expanded. The ["import", "preview"] key prefix keeps it in
  // the post-run invalidation below.
  const sourcesKey = [...selected].sort().join(",");
  const sessionsQs = new URLSearchParams(previewQs);
  sessionsQs.set("sources", sourcesKey);
  const { data: sessionList, error: sessionsError } = useQuery({
    queryKey: ["import", "preview", "sessions", since, until, project, sourcesKey],
    queryFn: () =>
      clientGet<ImportSessionsPreview>(`/import/preview/sessions?${sessionsQs.toString()}`),
    enabled: showList && selected.length > 0,
  });

  const running = status?.state === "running";

  // When a run finishes, refresh the preview split and the server-rendered
  // counts elsewhere on the page.
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) {
      queryClient.invalidateQueries({ queryKey: ["import", "preview"] });
      router.refresh();
    }
    wasRunning.current = Boolean(running);
  }, [running, queryClient, router]);

  function toggle(source: ImportSource) {
    setSelected((prev) =>
      prev.includes(source) ? prev.filter((s) => s !== source) : [...prev, source],
    );
  }

  function toggleExcluded(id: string) {
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function onStart() {
    setStarting(true);
    setStartError(null);
    try {
      const body: ImportRequest = { sources: selected, dry_run: dryRun };
      if (since) body.since = since;
      if (until) body.until = until;
      if (project) body.project = project;
      if (excluded.size > 0) body.exclude_session_ids = [...excluded];
      const res = await fetch("/api/pk/import", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.status === 409) throw new Error("An import is already running.");
      if (!res.ok) {
        const detail = await res
          .json()
          .then((data: { error?: string }) => data.error)
          .catch(() => undefined);
        throw new Error(detail || `Import failed to start (${res.status})`);
      }
      await queryClient.invalidateQueries({ queryKey: ["import", "status"] });
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }

  const runFilters = [
    status?.since ? `since ${status.since.slice(0, 10)}` : null,
    status?.until ? `to ${status.until.slice(0, 10)}` : null,
    status?.project ? `project “${status.project}”` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="space-y-3">
      <div className="space-y-2">
        {(Object.keys(SOURCE_LABELS) as ImportSource[]).map((source) => {
          const stats = preview?.sources?.[source];
          return (
            <label key={source} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={selected.includes(source)}
                onChange={() => toggle(source)}
                disabled={running}
              />
              <span>{SOURCE_LABELS[source]}</span>
              {stats ? (
                <span className="text-xs text-muted-foreground">
                  {formatNumber(stats.new)} new · {formatNumber(stats.imported)} imported
                </span>
              ) : null}
            </label>
          );
        })}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          Since
          <input
            type="date"
            value={since}
            onChange={(e) => setSince(e.target.value)}
            disabled={running}
            className={cn(inputClass, "w-40")}
            aria-label="Only sessions modified on/after"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          To (defaults to now)
          <input
            type="date"
            value={until}
            onChange={(e) => setUntil(e.target.value)}
            disabled={running}
            className={cn(inputClass, "w-40")}
            aria-label="Only sessions modified on/before"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          Project (Claude Code only)
          <input
            type="text"
            value={projectDraft}
            placeholder="directory substring"
            onChange={(e) => setProjectDraft(e.target.value)}
            onBlur={() => setProject(projectDraft.trim())}
            onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
            disabled={running}
            className={cn(inputClass, "w-48")}
            aria-label="Project filter"
          />
        </label>
      </div>
      <label className="flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={dryRun}
          onChange={(e) => setDryRun(e.target.checked)}
          disabled={running}
        />
        <span>Dry run</span>
        <span className="text-xs text-muted-foreground">discover and count, write nothing</span>
      </label>

      <div className="space-y-2">
        <button
          type="button"
          onClick={() => setShowList((v) => !v)}
          disabled={selected.length === 0}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground disabled:opacity-50"
        >
          {showList ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
          {showList ? "Hide sessions" : "Show sessions"}
        </button>
        {showList && selected.length > 0 ? (
          sessionsError ? (
            <p className="text-xs text-red-500">
              Failed to load sessions:{" "}
              {sessionsError instanceof Error ? sessionsError.message : String(sessionsError)}
            </p>
          ) : !sessionList ? (
            <p className="text-xs text-muted-foreground">Loading sessions…</p>
          ) : sessionList.sessions.length === 0 ? (
            <p className="text-xs text-muted-foreground">No sessions match the filters above.</p>
          ) : (
            <>
              {sessionList.titles_redacted ? (
                <p className="text-xs text-muted-foreground">
                  Plaintext exposure is disabled, so session titles are hidden.
                </p>
              ) : null}
              <div className="max-h-64 overflow-y-auto rounded-md border">
                <table className="w-full text-left text-xs">
                  <tbody>
                    {sessionList.sessions.map((s, i) => (
                      <tr
                        key={`${s.source}:${s.id ?? i}`}
                        className={cn(
                          "border-b last:border-0 hover:bg-accent/50",
                          s.imported && "opacity-60",
                        )}
                      >
                        <td
                          className="px-3 py-2"
                          title={
                            s.imported
                              ? "Already in the audit log — skipped on import"
                              : undefined
                          }
                        >
                          {s.imported ? (
                            <Check
                              className="h-3.5 w-3.5 text-muted-foreground"
                              aria-label="Already imported"
                            />
                          ) : s.id ? (
                            <input
                              type="checkbox"
                              checked={!excluded.has(s.id)}
                              onChange={() => s.id && toggleExcluded(s.id)}
                              disabled={running}
                              aria-label="Include this session in the import"
                            />
                          ) : null}
                        </td>
                        <td className="w-full max-w-0 truncate px-3 py-2" title={s.title ?? undefined}>
                          {s.title ?? (
                            <span className="italic text-muted-foreground">Untitled session</span>
                          )}
                        </td>
                        <td
                          className="max-w-[10rem] truncate whitespace-nowrap px-3 py-2 text-muted-foreground"
                          title={s.project ?? undefined}
                        >
                          {s.project ?? "—"}
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-muted-foreground">
                          {formatDateTime(s.modified_at)}
                        </td>
                        <td className="px-3 py-2">
                          <Badge
                            className={
                              s.imported
                                ? "text-muted-foreground"
                                : "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                            }
                          >
                            {s.imported ? "imported" : "new"}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {sessionList.total > sessionList.sessions.length ? (
                <p className="text-xs text-muted-foreground">
                  Showing the {formatNumber(sessionList.sessions.length)} most recent of{" "}
                  {formatNumber(sessionList.total)} sessions.
                </p>
              ) : null}
            </>
          )
        ) : null}
        {excluded.size > 0 ? (
          <p className="text-xs text-muted-foreground">
            {formatNumber(excluded.size)} unchecked — skipped on import.{" "}
            <button
              type="button"
              onClick={() => setExcluded(new Set())}
              disabled={running}
              className="underline hover:text-foreground"
            >
              Recheck all
            </button>
          </p>
        ) : null}
      </div>

      <Button size="sm" onClick={onStart} disabled={running || starting || selected.length === 0}>
        <DownloadCloud className="h-4 w-4" />
        {running ? "Importing…" : dryRun ? "Dry run" : "Import history"}
      </Button>

      {running ? (
        <p className="text-xs text-muted-foreground">
          {status?.dry_run ? "Dry run · " : ""}
          {formatNumber(status?.imported ?? 0)} of {formatNumber(status?.found ?? 0)} sessions ·{" "}
          {formatNumber(status?.turns ?? 0)} turns · {formatNumber(status?.entities ?? 0)} entities
          {status?.current ? ` — ${status.current.split("/").pop()}` : ""}
        </p>
      ) : null}
      {!running && status?.state === "done" ? (
        <p className="text-xs text-muted-foreground">
          {status.dry_run ? (
            <>
              Last dry run: {formatNumber(status.imported ?? 0)} would be imported,{" "}
              {formatNumber(status.skipped ?? 0)} skipped, {formatNumber(status.failed ?? 0)}{" "}
              failed · {formatNumber(status.turns ?? 0)} turns.
            </>
          ) : (
            <>
              Last run: {formatNumber(status.imported ?? 0)} imported,{" "}
              {formatNumber(status.skipped ?? 0)} skipped, {formatNumber(status.failed ?? 0)}{" "}
              failed · {formatNumber(status.turns ?? 0)} turns,{" "}
              {formatNumber(status.entities ?? 0)} PII entities.
            </>
          )}
          {runFilters ? ` · ${runFilters}` : ""}
        </p>
      ) : null}
      {status?.state === "error" ? (
        <p className="text-xs text-red-500">Import failed: {status.error}</p>
      ) : null}
      {startError ? <p className="text-xs text-red-500">{startError}</p> : null}
    </div>
  );
}
