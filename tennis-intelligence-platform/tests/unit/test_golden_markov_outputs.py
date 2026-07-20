from pathlib import Path

import pandas as pd
import pytest


GOLDEN = [
    ("20220120-M-Australian_Open-R64-Steve_Johnson-Jannik_Sinner", 1, 0.363702),
    ("20220120-M-Australian_Open-R64-Steve_Johnson-Jannik_Sinner", 2, 0.376799),
    ("20220120-M-Australian_Open-R64-Steve_Johnson-Jannik_Sinner", 3, 0.404097),
]


def test_markov_golden_outputs_from_saved_eval_file():
    """Golden regression guard on real-match rows from the saved Day 11 evaluation
    artifact. Skips if the artifact is not present in this environment."""
    pred_path = Path("data/processed/day11_head_to_head_v2_predictions.parquet")
    if not pred_path.exists():
        pytest.skip("day11 predictions parquet not present")

    df = pd.read_parquet(pred_path)

    for match_id, pt, expected in GOLDEN:
        row = df[(df["match_id"] == match_id) & (df["Pt"] == pt)]
        assert len(row) == 1, f"missing golden row for {match_id} Pt={pt}"
        got = float(row.iloc[0]["markov_pred"])
        assert got == pytest.approx(expected, abs=1e-6)
