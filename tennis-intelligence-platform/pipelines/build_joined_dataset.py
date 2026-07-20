"""
build_joined_dataset.py — pipeline entrypoint.

Runs the full TML <-> MCP join (Stages 1-6) for both genders, prints the Stage 5 validation
report to the console, and writes the joined dataset(s) to data/processed/.

Usage (from project root, with .venv activated):
    python pipelines/build_joined_dataset.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from tennis_intel.data.join_tml_mcp import run_full_join, write_joined_dataset
from tennis_intel.data.join_validation import build_validation_report, sample_unmatched

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TML_DIR = PROJECT_ROOT / "data" / "raw" / "TML-Database"
MCP_DIR = PROJECT_ROOT / "data" / "raw" / "tennis_MatchChartingProject"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    for gender, label in [("m", "men's / ATP"), ("w", "women's / WTA")]:
        logger.info("=== Running join for %s ===", label)

        if gender == "w":
            print(
                f"\n=== {label} ===\n"
                "SKIPPED: TML-Database contains ATP match-level data only (documented gap, "
                "see data/README.md). MCP has women's point-level data, but there is nothing "
                "in the TML pool to join it against — this will correctly return 0% coverage, "
                "not a join bug. Revisit once a live WTA match-level source is found.\n"
                f"{'=' * 60}\n"
            )
            continue

        result = run_full_join(TML_DIR, MCP_DIR, gender=gender)

        # tml_total / mcp_total for the report — reload lightly just for counts
        from tennis_intel.data.join_tml_mcp import load_tml_matches, load_mcp_matches
        tml_total = len(load_tml_matches(TML_DIR))
        mcp_total = len(load_mcp_matches(MCP_DIR, gender=gender))

        report = build_validation_report(tml_total, mcp_total, result)
        print(report.render())

        if not result.unmatched_mcp.empty:
            print(f"\nSample of unmatched {label} rows (for alias review):")
            print(sample_unmatched(result, n=20).to_string(index=False))

        output_path = OUTPUT_DIR / f"joined_matches_{gender}.parquet"
        write_joined_dataset(result, output_path)
        print(f"\nWrote {output_path}\n{'=' * 60}\n")


if __name__ == "__main__":
    main()