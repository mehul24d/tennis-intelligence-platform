"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { usePlayerSearch } from "@/hooks/usePlayer";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";

// useSearchParams() must be called inside a component wrapped in <Suspense> —
// without this, `npm run build` fails outright with "missing Suspense
// boundary with useSearchParams" (confirmed against Next.js's own docs). This
// only surfaces at build time, not in `next dev`, so it's an easy gap to miss
// during development.
function PlayerSearchContent() {
  const searchParams = useSearchParams();
  const [query, setQuery] = useState(searchParams.get("q") ?? "");
  const { data, isLoading } = usePlayerSearch(query);

  return (
    <>
      <Input
        placeholder="Search by name…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        autoFocus
      />

      {isLoading && <p className="text-caption">Searching…</p>}

      {data && data.length > 0 && (
        <Card className="divide-y divide-border overflow-hidden">
          {data.map((p) => (
            <Link
              key={p.player_id}
              href={`/players/${encodeURIComponent(p.player_id)}`}
              className="block px-4 py-3 hover:bg-accent-blue/5 hover:text-accent-blue transition-colors font-medium"
            >
              {p.player_name}
            </Link>
          ))}
        </Card>
      )}

      {data && data.length === 0 && query.length > 1 && (
        <p className="text-caption">No players found.</p>
      )}
    </>
  );
}

export default function PlayerSearchPage() {
  return (
    <div className="mx-auto max-w-2xl p-8 space-y-4">
      <h1 className="text-page-title">Find a Player</h1>
      <Suspense fallback={<p className="text-caption">Loading…</p>}>
        <PlayerSearchContent />
      </Suspense>
    </div>
  );
}