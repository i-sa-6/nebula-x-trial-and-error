"""PS3 submission-format CSV writer, and the one function that scores a new
(future) file the exact same way the LOCO folds are scored.

Format required by the organisers, one row per file:
    file_id,ranked_cars
    acv_case_01.xlsx,03|01|05|02|04|06|07|08
Car identifiers are exactly as they appear in that file's own column headers
("03", not "Car 3"), pipe-separated, most-to-least likely faulty.
"""
from __future__ import annotations

import pandas as pd

from . import aggregation, data_loading, preprocessing
from .models import build_models


def write_submission_csv(rankings: dict[str, list[str]], out_path) -> pd.DataFrame:
    rows = [{"file_id": file_id, "ranked_cars": "|".join(cars)} for file_id, cars in rankings.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def score_new_file(
    xlsx_path,
    prepared_labeled_df: pd.DataFrame,
    prep_report: preprocessing.PrepReport,
    model_name: str,
    aggregation_name: str,
) -> list[str]:
    """Score one new .xlsx file (e.g. the eventual acv_test_case.xlsx) with a
    model+aggregation combination already chosen from the LOCO results.

    Fits the standardizer and the model on ALL 5 labeled files (there is no
    label left to hold out once this is a real, unlabeled test file), reusing
    the category vocabulary/rare-category map already fit in `prep_report` --
    never refitting those on the new file. This is the single entrypoint the
    task asks for: once acv_test_case.xlsx exists, generating its prediction
    is just this one call.
    """
    case_id = xlsx_path.name if hasattr(xlsx_path, "name") else str(xlsx_path).rsplit("/", 1)[-1]
    df_wide = pd.read_excel(xlsx_path, sheet_name=0)
    long_df = data_loading.melt_case(df_wide, case_id)
    long_df, _n_filled = data_loading.apply_outside_temperature_fix(long_df)

    new_prepared, _missing_report = preprocessing.prepare_new_file(
        long_df, prep_report.merge_map, prep_report.vocab
    )

    standardizer = preprocessing.fit_standardizer(prepared_labeled_df, prep_report.numeric_cols_to_standardize)
    train_z = preprocessing.apply_standardizer(
        prepared_labeled_df, standardizer, prep_report.numeric_cols_to_standardize
    )
    new_z = preprocessing.apply_standardizer(new_prepared, standardizer, prep_report.numeric_cols_to_standardize)

    X_train = train_z[prep_report.feature_cols].to_numpy(dtype=float)
    X_new = new_z[prep_report.feature_cols].to_numpy(dtype=float)
    case_ids_train = train_z["case_id"].to_numpy()

    model_by_name = {m.name: m for m in build_models(fold_label="score_new_file")}
    if model_name not in model_by_name:
        raise ValueError(f"Unknown model_name={model_name!r}, choices={list(model_by_name)}")
    model = model_by_name[model_name]
    model.fit(X_train, case_ids=case_ids_train)
    scores = model.score(X_new)

    scored_df = new_z[["case_id", "car_number"]].copy()
    scored_df["score"] = scores
    agg_df = aggregation.aggregate_scores(scored_df, "score")
    score_col = "median_score" if aggregation_name == "median" else "p90_score"
    return aggregation.rank_cars(agg_df, score_col)[case_id]
