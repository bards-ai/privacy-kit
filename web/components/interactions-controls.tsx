"use client";

import "react-day-picker/style.css";

import { CalendarDays, Download, RotateCcw, Search } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { type CSSProperties, useCallback, useEffect, useRef, useState } from "react";
import { DayPicker } from "react-day-picker";

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

const MIN_YEAR = new Date().getFullYear() - 6;

const pad2 = (n: number) => String(n).padStart(2, "0");

function formatEu(value: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  return match ? `${match[3]}.${match[2]}.${match[1]}` : "";
}

function parseParam(value: string): Date | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  return match ? new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3])) : undefined;
}

function toParam(d: Date): string {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

const CALENDAR_VARS = {
  "--rdp-accent-color": "hsl(var(--primary))",
  "--rdp-accent-background-color": "hsl(var(--accent))",
  "--rdp-today-color": "hsl(var(--primary))",
  "--rdp-day-width": "2.1rem",
  "--rdp-day-height": "2.1rem",
  "--rdp-day_button-width": "2rem",
  "--rdp-day_button-height": "2rem",
  "--rdp-nav-height": "2rem",
} as CSSProperties;

function CalendarPopup({ value, onPick }: { value: string; onPick: (value: string) => void }) {
  const selected = parseParam(value);
  return (
    <div
      className="absolute left-0 top-full z-20 mt-1 rounded-md border bg-background p-3 text-sm shadow-md"
      style={CALENDAR_VARS}
    >
      <DayPicker
        mode="single"
        selected={selected}
        onSelect={(d) => onPick(d ? toParam(d) : "")}
        defaultMonth={selected}
        weekStartsOn={1}
        captionLayout="dropdown"
        reverseYears
        startMonth={new Date(MIN_YEAR, 0)}
        endMonth={new Date()}
      />
    </div>
  );
}

function DateField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  return (
    <div className="flex items-center gap-1.5">
      <span className="w-9 text-xs text-muted-foreground">{label}</span>
      <div ref={ref} className="relative">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className={cn(inputClass, "flex w-36 items-center justify-between gap-2 text-left")}
          aria-label={`${label} date`}
        >
          <span className={value ? undefined : "text-muted-foreground"}>
            {value ? formatEu(value) : "dd.mm.yyyy"}
          </span>
          <CalendarDays className="h-4 w-4 shrink-0 text-muted-foreground" />
        </button>
        {open ? (
          <CalendarPopup
            value={value}
            onPick={(v) => {
              onChange(v);
              setOpen(false);
            }}
          />
        ) : null}
      </div>
    </div>
  );
}

export function DateRangeFilter() {
  const searchParams = useSearchParams();
  const setParams = useSetParams();
  const fromParam = searchParams.get("date_from");
  const toParam = searchParams.get("date_to");
  const [from, setFrom] = useState(fromParam ?? "");
  const [to, setTo] = useState(toParam ?? "");
  useEffect(() => {
    setFrom(fromParam ?? "");
    setTo(toParam ?? "");
  }, [fromParam, toParam]);

  const dirty = (from || null) !== fromParam || (to || null) !== toParam;
  const hasAnything = Boolean(fromParam || toParam || from || to);

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <DateField label="From" value={from} onChange={setFrom} />
      <DateField label="To" value={to} onChange={setTo} />
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          disabled={!dirty}
          onClick={() => setParams({ date_from: from || null, date_to: to || null })}
        >
          Apply
        </Button>
        {hasAnything ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setFrom("");
              setTo("");
              setParams({ date_from: null, date_to: null });
            }}
          >
            <RotateCcw className="h-4 w-4" /> Reset
          </Button>
        ) : null}
      </div>
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
