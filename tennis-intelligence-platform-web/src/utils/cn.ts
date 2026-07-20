// utils/cn.ts — the standard shadcn/ui className-merging helper: combines clsx
// (conditional classes) with tailwind-merge (dedupes conflicting Tailwind classes,
// e.g. "px-2 px-4" -> "px-4"), so components can accept a className prop and merge
// it safely with their own defaults.
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
