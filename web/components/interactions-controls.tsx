"use client";

import { Download, RotateCcw, Search } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useState } from "react";

import { Button } from "@/components/button";
import { cn } from "@/lib/cn";
import type { FilterValues } from "@/lib/types";

const inputClass =
  "h-9 rounded-md border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring";

function useSetParams() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  return useCallback(
    (updates: Record<string, string | null>, resetPage = true) => {
      const params = new URLSearchParams(searchParams.toString());
      if (resetPage) params.delete("page");
      for (const [k, v] of Object.entries(updates)) {
        if (v === null || v === "") params.delete(k);
        else params.set(k, v);
      }
      const qs = params.toString();
      router.push(qs ? `${pathname}?${qs}` : pathname);
    },
    [router, pathname, searchParams],
  );
}

const FILTER_KEYS = [
  "q", "source", "wire_format", "model", "entity_type", "policy",
  "date_from", "date_to", "min_entities",
];

export function FilterBar({ filters }: { filters: FilterValues }) {
  const searchParams = useSearchParams();
  const setParams = useSetParams();
  const get = (k: string) => searchParams.get(k) ?? "";
  const [q, setQ] = useState(get("q"));

  const selects = [
    { key: "source", label: "Source", options: filters.sources },
    { key: "wire_format", label: "Wire", options: filters.wire_formats },
    { key: "model", label: "Model", options: filters.models },
    { key: "entity_type", label: "Entity", options: filters.entity_types },
    { key: "policy", label: "Policy", options: filters.policies },
  ];
  const hasFilters = FILTER_KEYS.some((k) => get(k));

  return (
    <div className="mb-4 grid grid-cols-2 gap-2 lg:grid-cols-4">
      <form
        className="relative col-span-2"
        onSubmit={(e) => {
          e.preventDefault();
          setParams({ q: q || null });
        }}
      >
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search saved text (original or anonymized)…"
          className={cn(inputClass, "w-full pl-9")}
        />
      </form>
      {selects.map((s) => (
        <select
          key={s.key}
          value={get(s.key)}
          onChange={(e) => setParams({ [s.key]: e.target.value || null })}
          className={cn(inputClass, "w-full")}
          aria-label={s.label}
        >
          <option value="">{s.label}: all</option>
          {s.options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      ))}
      <input
        type="date"
        value={get("date_from")}
        onChange={(e) => setParams({ date_from: e.target.value || null })}
        className={cn(inputClass, "w-full")}
        aria-label="From date"
      />
      <input
        type="date"
        value={get("date_to")}
        onChange={(e) => setParams({ date_to: e.target.value || null })}
        className={cn(inputClass, "w-full")}
        aria-label="To date"
      />
      <input
        type="number"
        min={0}
        value={get("min_entities")}
        onChange={(e) => setParams({ min_entities: e.target.value || null })}
        placeholder="Min entities"
        className={cn(inputClass, "w-full")}
        aria-label="Minimum entities"
      />
      {hasFilters ? (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setParams(Object.fromEntries(FILTER_KEYS.map((k) => [k, null])))}
        >
          <RotateCcw className="h-4 w-4" /> Reset
        </Button>
      ) : null}
    </div>
  );
}

export function SortHeader({ column, label }: { column: string; label: string }) {
  const searchParams = useSearchParams();
  const setParams = useSetParams();
  const activeSort = searchParams.get("sort") ?? "created_at";
  const order = searchParams.get("order") ?? "desc";
  const isActive = activeSort === column;
  const nextOrder = isActive && order === "desc" ? "asc" : "desc";
  return (
    <button
      type="button"
      onClick={() => setParams({ sort: column, order: nextOrder })}
      className={cn(
        "inline-flex items-center gap-1 hover:text-foreground",
        isActive && "text-foreground",
      )}
    >
      {label}
      <span className="text-[10px] leading-none">
        {isActive ? (order === "desc" ? "▼" : "▲") : ""}
      </span>
    </button>
  );
}

export function Pagination({
  page,
  totalPages,
  total,
  pageSize,
}: {
  page: number;
  totalPages: number;
  total: number;
  pageSize: number;
}) {
  const setParams = useSetParams();
  return (
    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm">
      <div className="text-muted-foreground">
        {total === 0 ? "No results" : `Page ${page} of ${Math.max(1, totalPages)} · ${total} total`}
      </div>
      <div className="flex items-center gap-2">
        <select
          value={String(pageSize)}
          onChange={(e) => setParams({ page_size: e.target.value })}
          className={cn(inputClass, "h-8")}
          aria-label="Page size"
        >
          {[25, 50, 100, 200].map((n) => (
            <option key={n} value={n}>
              {n}/page
            </option>
          ))}
        </select>
        <Button
          variant="outline"
          size="sm"
          disabled={page <= 1}
          onClick={() => setParams({ page: String(page - 1) }, false)}
        >
          Prev
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={page >= totalPages}
          onClick={() => setParams({ page: String(page + 1) }, false)}
        >
          Next
        </Button>
      </div>
    </div>
  );
}

export function ExportMenu() {
  const searchParams = useSearchParams();
  const [open, setOpen] = useState(false);
  const base = searchParams.toString();
  const suffix = base ? `&${base}` : "";
  const item = "flex w-full items-center rounded-md px-3 py-1.5 text-sm hover:bg-accent";
  return (
    <div className="relative">
      <Button variant="outline" size="sm" onClick={() => setOpen((o) => !o)}>
        <Download className="h-4 w-4" /> Export
      </Button>
      {open ? (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-20 mt-1 w-56 rounded-md border bg-card p-1 shadow-lg">
            <a href={`/api/pk/export?format=csv${suffix}`} className={item} onClick={() => setOpen(false)}>
              CSV (metadata)
            </a>
            <a
              href={`/api/pk/export?format=json&include_texts=true${suffix}`}
              className={item}
              onClick={() => setOpen(false)}
            >
              JSON (with text)
            </a>
          </div>
        </>
      ) : null}
    </div>
  );
}
