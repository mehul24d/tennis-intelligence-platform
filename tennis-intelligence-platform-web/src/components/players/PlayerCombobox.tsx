"use client";

import { useState } from "react";
import { usePlayerSearch } from "@/hooks/usePlayer";
import { Input } from "@/components/ui/input";
import { PlayerSearchResult } from "@/types/playerProfile";

export function PlayerCombobox({
  label,
  selected,
  onSelect,
}: {
  label: string;
  selected: PlayerSearchResult | null;
  onSelect: (player: PlayerSearchResult | null) => void;
}) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const { data } = usePlayerSearch(query);

  return (
    <div className="relative">
      <p className="text-label mb-1.5">{label}</p>
      <Input
        placeholder="Search by name…"
        value={selected ? selected.player_name : query}
        onChange={(e) => {
          setQuery(e.target.value);
          if (selected) onSelect(null);
        }}
        onFocus={() => setFocused(true)}
        onBlur={() => setTimeout(() => setFocused(false), 150)}
      />
      {focused && !selected && data && data.length > 0 && (
        <div className="absolute z-10 mt-1 w-full glass rounded-lg overflow-hidden shadow-xl">
          {data.map((p) => (
            <button
              key={p.player_id}
              onMouseDown={() => onSelect(p)}
              className="block w-full text-left px-4 py-2.5 text-sm hover:bg-accent-blue/10 hover:text-accent-blue transition-colors"
            >
              {p.player_name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}