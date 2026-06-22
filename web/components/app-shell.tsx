"use client";

import { Eye, FileText, LayoutDashboard, List, Settings, ShieldHalf } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";
import { ThemeToggle } from "@/components/theme-toggle";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/interactions", label: "Interactions", icon: List },
  { href: "/texts", label: "Text segments", icon: FileText },
  { href: "/preview", label: "Live preview", icon: Eye },
  { href: "/settings", label: "Settings", icon: Settings },
];

function isActive(pathname: string, href: string): boolean {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r bg-card md:flex">
        <div className="flex h-14 items-center gap-2 border-b px-5">
          <ShieldHalf className="h-5 w-5 text-primary" />
          <span className="font-semibold tracking-tight">privacy-kit</span>
        </div>
        <nav className="flex-1 space-y-1 p-3">
          {NAV.map((item) => {
            const Icon = item.icon;
            const active = isActive(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                  active
                    ? "bg-primary/10 font-medium text-primary"
                    : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="border-t p-4 text-xs text-muted-foreground">
          On-device PII gateway dashboard
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b bg-background/80 px-4 backdrop-blur">
          <div className="flex items-center gap-2 md:hidden">
            <ShieldHalf className="h-5 w-5 text-primary" />
            <span className="font-semibold">privacy-kit</span>
          </div>
          <nav className="flex items-center gap-1 overflow-x-auto md:hidden">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "whitespace-nowrap rounded-md px-2 py-1 text-xs",
                  isActive(pathname, item.href)
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground",
                )}
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 p-4 md:p-8">{children}</main>
      </div>
    </div>
  );
}
