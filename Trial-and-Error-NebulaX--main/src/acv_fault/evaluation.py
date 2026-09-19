"""Leave-one-case-out evaluation, scored with PS3's actual ranking formula.

For each of the 5 files held out in turn, for the settled model (Elliptic
Envelope -- see models/__init__.py for why): refit
standardization + the model on the pooled OTHER 4 files only (never on the
held-out file), score the held-out file's rows, aggregate to a per-car ranking
under both `median` and `p90`, and score that ranking with PS3's formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import aggregation, config, preprocessing
from .models import build_models


def primary_metric(rank: int, n: int = config.N_CARS) -> float:
    """PS3's formula: (n - (r - 1)) / n, r = 1-indexed rank of the true faulty car."""
    return (n - (rank - 1)) / n


def mrr_secondary(rank: int) -> float:
    return 1.0 / rank


@dataclass
class RunResult:
    case_id: str
    model: str
    aggregation: str
    rank_of_true_car: int
    primary_metric: float
    mrr_secondary: float
    n_tied_with_true_car: int  # 1 = genuine discrimination; >1 = rank decided by tie-break order


def run_loco_evaluation(
    prepared_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols_to_standardize: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, list[str]]], pd.DataFrame]:
    """Returns (per-run results table, human-readable per-fold notes,
    {model: {case_id: ranked_car_list}} using the median aggregation --
    kept around so the caller can build the submission-format dry-run without
    re-running the folds -- and a per-(file, model, car) anomaly-count table
    from model.flag(), which never affects the ranking above -- that is
    always computed from the continuous score(), unchanged).
    """
    case_ids = prepared_df["case_id"].unique().tolist()
    results: list[RunResult] = []
    fold_notes: list[str] = []
    rankings_by_model: dict[str, dict[str, dict[str, list[str]]]] = {}
    anomaly_count_rows: list[dict] = []

    for held_out_case in case_ids:
        print(f"\n=== LOCO fold: held-out file = {held_out_case} ===")
        train_pool = prepared_df[prepared_df["case_id"] != held_out_case]
        held_out = prepared_df[prepared_df["case_id"] == held_out_case]
        true_faulty_car = held_out["faulty_car"].iloc[0]

        standardizer = preprocessing.fit_standardizer(train_pool, numeric_cols_to_standardize)
        train_pool_z = preprocessing.apply_standardizer(train_pool, standardizer, numeric_cols_to_standardize)
        held_out_z = preprocessing.apply_standardizer(held_out, standardizer, numeric_cols_to_standardize)

        X_train = train_pool_z[feature_cols].to_numpy(dtype=float)
        X_held = held_out_z[feature_cols].to_numpy(dtype=float)
        case_ids_train = train_pool_z["case_id"].to_numpy()

        for model in build_models(fold_label=held_out_case):
            model.fit(X_train, case_ids=case_ids_train)
            scores = model.score(X_held)

            scored_df = held_out_z[["case_id", "car_number"]].copy()
            scored_df["score"] = scores
            agg_df = aggregation.aggregate_scores(scored_df, "score")

            for agg_name, score_col in (("median", "median_score"), ("p90", "p90_score")):
                ranking = aggregation.rank_cars(agg_df, score_col)[held_out_case]
                rank_of_true = ranking.index(true_faulty_car) + 1
                n_tied = aggregation.n_tied_with_car(agg_df, score_col, held_out_case, true_faulty_car)
                results.append(
                    RunResult(
                        case_id=held_out_case,
                        model=model.name,
                        aggregation=agg_name,
                        rank_of_true_car=rank_of_true,
                        primary_metric=primary_metric(rank_of_true),
                        mrr_secondary=mrr_secondary(rank_of_true),
                        n_tied_with_true_car=n_tied,
                    )
                )
                rankings_by_model.setdefault(model.name, {}).setdefault(agg_name, {})[held_out_case] = ranking
                if n_tied > 1:
                    note = (
                        f"[{held_out_case}] {model.name}/{agg_name}: true car {true_faulty_car}'s rank "
                        f"({rank_of_true}) was decided by tie-break order -- {n_tied} of 8 cars share the "
                        f"exact same {agg_name} score."
                    )
                    print(note)
                    fold_notes.append(note)

            note = _model_fold_note(model, held_out_case)
            if note:
                print(note)
                fold_notes.append(note)

            flags = model.flag(X_held)
            flagged_df = held_out_z[["case_id", "car_number"]].copy()
            flagged_df["flagged"] = flags
            per_car = flagged_df.groupby("car_number")["flagged"].agg(n_flagged="sum", n_total="count")
            for car_number, row in per_car.iterrows():
                anomaly_count_rows.append(
                    {
                        "case_id": held_out_case,
                        "model": model.name,
                        "car_number": car_number,
                        "is_true_car": car_number == true_faulty_car,
                        "n_flagged": int(row["n_flagged"]),
                        "n_total": int(row["n_total"]),
                        "frac_flagged": row["n_flagged"] / row["n_total"],
                    }
                )

    results_df = pd.DataFrame([r.__dict__ for r in results])
    anomaly_counts_df = pd.DataFrame(anomaly_count_rows)
    return results_df, fold_notes, rankings_by_model, anomaly_counts_df


def _model_fold_note(model, fold_label: str) -> str | None:
    if model.name == "elliptic_envelope" and model.fallback_used:
        return f"[{fold_label}] EllipticEnvelope: LedoitWolf fallback used ({model.fallback_reason})"
    return None


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """model x aggregation -> mean primary_metric, mean mrr_secondary, and the
    actual rank given to the true faulty car in each of the 5 held-out files.
    Sorted best -> worst by primary_metric_mean.
    """
    rows = []
    for (model, agg), group in results_df.groupby(["model", "aggregation"]):
        group_sorted = group.sort_values("case_id")
        rows.append(
            {
                "model": model,
                "aggregation": agg,
                "primary_metric_mean": group["primary_metric"].mean(),
                "mrr_secondary_mean": group["mrr_secondary"].mean(),
                "ranks_by_file": dict(zip(group_sorted["case_id"], group_sorted["rank_of_true_car"])),
                "n_files_with_tied_rank": int((group_sorted["n_tied_with_true_car"] > 1).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    # Deterministic, documented tie-break for exact primary_metric ties (pandas'
    # default sort is not guaranteed stable): prefer fewer tie-break-decided
    # ranks (genuine discrimination over arbitrary car-number order), then
    # higher mrr_secondary (finer-grained than the integer-rank primary
    # metric), then model/aggregation name so the result is fully reproducible.
    return summary.sort_values(
        ["primary_metric_mean", "n_files_with_tied_rank", "mrr_secondary_mean", "model", "aggregation"],
        ascending=[False, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
