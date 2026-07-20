"""
test_feature_list_consistency.py — permanent regression test guarding against exactly
the class of bug found via check_deciding_set_importance.py's guard: a column can be
correctly listed in feature_schema.py's PREMATCH_FEATURE_NAMES (what the classifier is
TOLD to look for) while never actually being merged into the point-level dataframe by
build_point_dataset.py's separate, manually-maintained PREMATCH_FEATURE_COLS list (what
ACTUALLY gets merged from the Day 6 parquet file). feature_schema.py's request has
silently no effect if the corresponding merge was never added — exactly what happened
with winner_/loser_second_serve_win_pct_surface_career, caught only because a classifier
retrain's own defensive guard happened to flag it, not because any test existed for
this specific two-list consistency requirement.

This test exists so the NEXT time a pre-match feature is added to one list but not the
other, it fails immediately and explicitly, rather than silently training a model with
one fewer feature than intended and only surfacing the gap much later, incidentally.
"""

import pandas as pd
import pytest

from tennis_intel.live.build_point_dataset import PREMATCH_FEATURE_COLS
from tennis_intel.live.feature_schema import PREMATCH_FEATURE_NAMES


class TestFeatureListConsistency:
    def test_every_prematch_feature_is_actually_merged(self):
        """Every column feature_schema.py expects to be available pre-match must
        actually be present in build_point_dataset.py's merge list — otherwise the
        classifier is silently trained without a feature it was told to look for."""
        missing = [c for c in PREMATCH_FEATURE_NAMES if c not in PREMATCH_FEATURE_COLS]
        assert not missing, (
            f"These columns are listed in feature_schema.py's PREMATCH_FEATURE_NAMES "
            f"but are NOT in build_point_dataset.py's PREMATCH_FEATURE_COLS, meaning "
            f"they will never actually reach the point-level dataframe: {missing}"
        )

    def test_no_orphaned_merge_columns(self):
        """The reverse direction, for completeness: a column merged into the point-
        level dataframe but never requested by feature_schema.py isn't a correctness
        bug (harmless, unused data), but is worth surfacing as a prompt to check
        whether it should actually be added to the schema, or was a genuine leftover."""
        orphaned = [c for c in PREMATCH_FEATURE_COLS if c not in PREMATCH_FEATURE_NAMES]
        if orphaned:
            print(f"\nNOTE (not a failure): these columns are merged by "
                  f"build_point_dataset.py but not requested by feature_schema.py's "
                  f"PREMATCH_FEATURE_NAMES -- harmless, but worth checking whether "
                  f"they should be added to the schema: {orphaned}")

    def test_combined_serve_win_pct_career_computation_does_not_raise(self):
        """Regression test for a real, production-breaking bug: PREMATCH_FEATURE_COLS
        was edited to remove loser_second_serve_win_pct_career (a change that was
        correct FOR THE CLASSIFIER's own feature list), without checking that
        build_point_dataset.py's own internal combined_serve_win_pct_career loop
        unconditionally references {prefix}_second_serve_win_pct_career for BOTH
        winner and loser prefixes — a completely different, load-bearing consumer
        (the return-seed/Beta-Binomial seeding) than the point-level classifier.
        Removing a column from the classifier's feature list is NOT the same
        question as whether that column is still needed elsewhere in the pipeline —
        this test directly reproduces the exact computation against a synthetic
        dataframe built purely from PREMATCH_FEATURE_COLS, confirming it never
        raises a KeyError regardless of what feature_schema.py separately excludes."""
        synthetic = pd.DataFrame({col: [0.5] for col in PREMATCH_FEATURE_COLS})
        for _prefix in ("winner", "loser"):
            # Must not raise KeyError -- this IS the test.
            _ = (
                synthetic[f"{_prefix}_first_serve_in_pct_career"]
                * synthetic[f"{_prefix}_first_serve_win_pct_career"]
                + (1.0 - synthetic[f"{_prefix}_first_serve_in_pct_career"])
                * synthetic[f"{_prefix}_second_serve_win_pct_career"]
            )