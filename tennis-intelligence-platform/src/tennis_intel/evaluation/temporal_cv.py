"""
temporal_cv.py — rolling-origin (walk-forward) temporal cross-validation, consistent with
the methodology used in the project owner's prior World Cup project (train strictly before
a target year, test on that year).

LEAKAGE PROOF: for every fold, the train set is defined as `date < test_year_start` — by
construction, no training row can have a date on or after any test row's date. Verified
explicitly in tests/unit/test_temporal_cv.py::test_no_leakage_across_folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalFold:
    test_year: int
    train_idx: pd.Index
    test_idx: pd.Index


def generate_temporal_folds(
    df: pd.DataFrame, date_col: str, test_years: list[int]
) -> list[TemporalFold]:
    """
    For each year in `test_years`, produces a fold where:
      - train = every row with date_col STRICTLY BEFORE Jan 1 of that test year
      - test  = every row with date_col falling within that test year

    Folds are independent (not nested) — each test_year gets its own expanding training
    window. This mirrors rolling-origin backtesting: as more real matches accumulate, later
    folds simply have more training data, never less, and never any future data.
    """
    dates = pd.to_datetime(df[date_col])
    folds = []
    for year in test_years:
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year + 1}-01-01")
        train_mask = dates < year_start
        test_mask = (dates >= year_start) & (dates < year_end)
        folds.append(TemporalFold(
            test_year=year,
            train_idx=df.index[train_mask],
            test_idx=df.index[test_mask],
        ))
    return folds


def held_out_split(df: pd.DataFrame, date_col: str, holdout_start_year: int) -> tuple[pd.Index, pd.Index]:
    """A single train/final-holdout split: everything before `holdout_start_year` is
    available for model development (including internal temporal CV via
    generate_temporal_folds); everything from that year onward is reserved as a final,
    untouched test set for the results section — must only be evaluated ONCE, at the end."""
    dates = pd.to_datetime(df[date_col])
    cutoff = pd.Timestamp(f"{holdout_start_year}-01-01")
    dev_idx = df.index[dates < cutoff]
    holdout_idx = df.index[dates >= cutoff]
    return dev_idx, holdout_idx