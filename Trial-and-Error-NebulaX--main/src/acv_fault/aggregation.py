"""Per-timestamp anomaly scores -> per-car scores -> a descending ranking."""
from __future__ import annotations

import pandas as pd

AGGREGATIONS = ("median", "p90")


def aggregate_scores(
    df: pd.DataFrame, score_col: str, case_col: str = "case_id", car_col: str = "car_number"
) -> pd.DataFrame:
    """One row per (case_id, car_number): its median and 90th-percentile score."""
    grouped = df.groupby([case_col, car_col])[score_col]
    out = grouped.median().rename("median_score").to_frame()
    out["p90_score"] = grouped.quantile(0.90)
    return out.reset_index()


def rank_cars(
    df_agg: pd.DataFrame, score_col: str, case_col: str = "case_id", car_col: str = "car_number"
) -> dict[str, list[str]]:
    """Per case_id: car_numbers sorted descending by score_col (index 0 = most anomalous).

    NOTE: ties (common on this heavily-duplicated sensor data -- see README) are
    broken by a stable sort falling back to the pre-sort row order, which is
    ascending car_number. A rank produced under a tie is NOT genuine model
    discrimination; use `n_tied_with_car` to detect this before trusting a rank.
    """
    rankings = {}
    for case_id, group in df_agg.groupby(case_col):
        # kind="stable" makes the tie-break for exactly-equal scores a documented,
        # reproducible fallback to ascending car_number, not a pandas sort-algorithm
        # implementation detail. See n_tied_with_car -- a rank decided by a tie is
        # never genuine model discrimination regardless of which car it lands on.
        ordered = group.sort_values(score_col, ascending=False, kind="stable")
        rankings[case_id] = ordered[car_col].tolist()
    return rankings


def n_tied_with_car(
    df_agg: pd.DataFrame,
    score_col: str,
    case_id: str,
    car_number: str,
    case_col: str = "case_id",
    car_col: str = "car_number",
) -> int:
    """How many of the 8 cars (including itself) share `car_number`'s exact
    score_col value in this file. 1 = genuinely distinguished; >1 = its rank
    was decided by tie-break order, not by the model.
    """
    group = df_agg[df_agg[case_col] == case_id]
    car_score = group.loc[group[car_col] == car_number, score_col].iloc[0]
    return int((group[score_col] == car_score).sum())
