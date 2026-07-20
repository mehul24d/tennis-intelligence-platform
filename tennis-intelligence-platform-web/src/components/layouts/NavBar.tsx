"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import { Activity, Search } from "lucide-react";
import { cn } from "@/utils/cn";

const NAV_LINKS = [
  { href: "/explorer", label: "Matches" },
  { href: "/players", label: "Players" },
  { href: "/compare", label: "Compare" },
  { href: "/rankings", label: "Rankings" },
  { href: "/models", label: "Models" },
  { href: "/model-comparison", label: "Compare Models" },
  { href: "/research", label: "Research" },
];

export function NavBar() {
  const pathname = usePathname();
  const router = useRouter();
  const [query, setQuery] = useState("");

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    router.push(`/players?q=${encodeURIComponent(trimmed)}`);
  };

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto max-w-6xl px-8 h-14 flex items-center gap-6">
        <Link href="/" className="flex items-center gap-2 shrink-0">
          <Activity className="h-5 w-5 text-accent-blue" />
          <span className="font-semibold text-sm tracking-tight">Tennis Intelligence</span>
        </Link>

        <nav className="flex items-center gap-1 whitespace-nowrap">
          {NAV_LINKS.map((link) => {
            const active = pathname === link.href || pathname.startsWith(link.href + "/");
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                  active
                    ? "text-accent-blue bg-accent-blue/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <form onSubmit={handleSearch} className="ml-auto w-full max-w-xs">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search players…"
              className="h-8 w-full rounded-md border border-border bg-background-elevated pl-8 pr-3 text-xs outline-none placeholder:text-muted-foreground transition-shadow focus:border-accent-blue focus:ring-2 focus:ring-accent-blue/20"
            />
          </div>
        </form>
      </div>
    </header>
  );
}