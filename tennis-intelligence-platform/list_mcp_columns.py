"""
inspect_mcp_sample_rows.py — prints actual sample rows (not just headers) for the three
files flagged as highest-priority in the feature-recommendation report: KeyPointsServe,
KeyPointsReturn, and Overview. Specifically needed to confirm what the categorical `row`
column actually contains in the KeyPoints files (e.g. "BreakPoints"? "GamePoints"?) and
what `set` values look like in Overview (integers 1-5? something else?) — headers alone
don't reveal this, and guessing at it would repeat the same mistake as inferring column
names without checking.

Usage:
    python inspect_mcp_sample_rows.py /path/to/tennis_MatchChartingProject
"""

import sys
from pathlib import Path

import pandas as pd

FILES_TO_INSPECT = [
    "charting-m-stats-KeyPointsServe.csv",
    "charting-m-stats-KeyPointsReturn.csv",
    "charting-m-stats-Overview.csv",
]


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not root.exists():
        print(f"Path does not exist: {root}")
        sys.exit(1)

    for filename in FILES_TO_INSPECT:
        path = root / filename
        if not path.exists():
            print(f"=== {filename}: NOT FOUND at {path} ===\n")
            continue

        print(f"=== {filename} ===")
        try:
            df = pd.read_csv(path, nrows=20)
        except Exception as e:
            print(f"  ERROR reading file: {type(e).__name__}: {e}")
            continue

        # Print full rows for the first single match_id found, so the relationship
        # between match_id/player/row (or set) is visible together, not just isolated
        # column values across unrelated matches.
        first_match_id = df["match_id"].iloc[0]
        subset = df[df["match_id"] == first_match_id]
        print(f"All rows for match_id={first_match_id!r} (first {len(subset)} rows in file):")
        print(subset.to_string(index=False))

        if "row" in df.columns:
            print(f"\nUnique 'row' values seen in this 20-row sample: "
                  f"{sorted(df['row'].dropna().unique().tolist())}")
        if "set" in df.columns:
            print(f"\nUnique 'set' values seen in this 20-row sample: "
                  f"{sorted(df['set'].dropna().unique().tolist())}")
        print()


if __name__ == "__main__":
    main()