"""
diagnose_frozen_join_schema.py — prints every column in frozen_join and day6, plus a
sample row, so the match-list API endpoint can be built against CONFIRMED column names
rather than guessed ones. This project has already had real bugs from assuming a
column name existed without checking (the surface second-serve merge-list gap, the
build_point_dataset.py KeyError) — this is the same discipline applied before writing
new code instead of after finding a bug.
"""

import pandas as pd

PROCESSED = "data/processed"


def main() -> None:
    frozen_join = pd.read_parquet(f"{PROCESSED}/joined_matches_m.parquet")
    day6 = pd.read_parquet(f"{PROCESSED}/matches_with_day6_features.parquet")

    print(f"=== frozen_join: {len(frozen_join):,} rows ===")
    print(f"Columns ({len(frozen_join.columns)}):")
    for c in sorted(frozen_join.columns):
        print(f"  {c}")
    print("\nSample row (first row, all columns):")
    print(frozen_join.iloc[0].to_dict())

    print(f"\n\n=== day6: {len(day6):,} rows ===")
    print(f"Columns ({len(day6.columns)}):")
    for c in sorted(day6.columns):
        print(f"  {c}")
    print("\nSample row (first row, all columns):")
    print(day6.iloc[0].to_dict())


if __name__ == "__main__":
    main()