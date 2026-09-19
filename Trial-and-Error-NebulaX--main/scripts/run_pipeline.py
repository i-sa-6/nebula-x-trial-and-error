"""End-to-end entrypoint: load the 5 labeled files, preprocess, run the
leave-one-case-out evaluation (Elliptic Envelope x 2 aggregations), print
the results table, and write a PS3-format submission dry-run CSV for the
best-scoring aggregation.

Usage:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from acv_fault import config, data_loading, evaluation, preprocessing, submission


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("#" * 80)
    print("STEP 1/4: Loading + melting the 5 labeled files")
    print("#" * 80)
    full_long, labels_df, load_reports = data_loading.load_all_cases()
    for r in load_reports:
        print(
            f"  {r.filename}: car_model={r.car_model}, n_timestamps={r.n_timestamps}, "
            f"n_long_rows={r.n_long_rows}, outside_temp_end_car_fills={r.outside_temp_end_car_fills}"
        )

    print("\n" + "#" * 80)
    print("STEP 2/4: Global preprocessing (missing values, rare-category merge, one-hot, feature list)")
    print("#" * 80)
    prepared, report = preprocessing.prepare_global(full_long)

    print("\nRows dropped/imputed per file:")
    missing_report_df = pd.DataFrame([m.__dict__ for m in report.missing_value_reports])
    print(missing_report_df.to_string(index=False))
    missing_report_df.to_csv(config.RESULTS_DIR / "missing_value_report.csv", index=False)

    rarity_df = pd.DataFrame([r.__dict__ for r in report.rarity_merges])
    rarity_df.to_csv(config.RESULTS_DIR / "rare_category_merges.csv", index=False)

    print("\n" + "#" * 80)
    print("STEP 3/4: Leave-one-case-out evaluation (5 files x 2 aggregations, Elliptic Envelope)")
    print("#" * 80)
    results_df, fold_notes, rankings_by_model, anomaly_counts_df = evaluation.run_loco_evaluation(
        prepared, report.feature_cols, report.numeric_cols_to_standardize
    )
    results_df.to_csv(config.RESULTS_DIR / "loco_run_results.csv", index=False)
    anomaly_counts_df.to_csv(config.RESULTS_DIR / "anomaly_counts.csv", index=False)

    with open(config.RESULTS_DIR / "fold_notes.txt", "w") as f:
        f.write("\n".join(fold_notes))

    ee_fallback_notes = [n for n in fold_notes if "LedoitWolf fallback" in n]
    print(f"\nEllipticEnvelope folds needing LedoitWolf fallback: {len(ee_fallback_notes)}")
    for n in ee_fallback_notes:
        print(" ", n)

    print("\n=== Anomaly counts flagged per model (sum of n_flagged across all 5 held-out files) ===")
    count_summary = anomaly_counts_df.groupby("model").agg(
        n_flagged_total=("n_flagged", "sum"), n_rows_total=("n_total", "sum")
    )
    count_summary["frac_flagged"] = (count_summary["n_flagged_total"] / count_summary["n_rows_total"]).round(4)
    print(count_summary.to_string())
    count_summary.to_csv(config.RESULTS_DIR / "anomaly_counts_summary.csv")

    print("\n=== Anomaly counts flagged specifically for the TRUE faulty car, per (file, model) ===")
    true_car_counts = anomaly_counts_df[anomaly_counts_df["is_true_car"]].sort_values(["case_id", "model"])
    print(true_car_counts[["case_id", "model", "car_number", "n_flagged", "n_total", "frac_flagged"]].to_string(index=False))

    summary_df = evaluation.summarize_results(results_df)
    print("\n=== Results table: model x aggregation, best -> worst by primary_metric ===")
    display_df = summary_df.copy()
    display_df["primary_metric_mean"] = display_df["primary_metric_mean"].round(4)
    display_df["mrr_secondary_mean"] = display_df["mrr_secondary_mean"].round(4)
    print(display_df.to_string(index=False))
    summary_df.to_csv(config.RESULTS_DIR / "results_summary.csv", index=False)

    print("\n=== Per-file rank of the true faulty car (1 = ranked most likely faulty) ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']:22s} / {row['aggregation']:6s}: {row['ranks_by_file']}")

    print("\n" + "#" * 80)
    print("STEP 4/4: Submission-format dry-run CSV for the best (model, aggregation)")
    print("#" * 80)
    best = summary_df.iloc[0]
    best_model, best_agg = best["model"], best["aggregation"]
    print(
        f"Best combination by primary_metric: model={best_model!r}, aggregation={best_agg!r} "
        f"(mean primary_metric={best['primary_metric_mean']:.4f})"
    )
    best_rankings = rankings_by_model[best_model][best_agg]
    out_path = config.RESULTS_DIR / "submission_dryrun.csv"
    submission_df = submission.write_submission_csv(best_rankings, out_path)
    print(f"\nWrote {out_path}:")
    print(submission_df.to_string(index=False))

    print("\nDone. All artifacts written to results/.")


if __name__ == "__main__":
    main()
