"""Preprocessing pipeline.

Split into two explicit stages so fold-safety is structurally hard to violate:

- ``prepare_global``: steps that are safe to run once across all 5 files (no
  fitted statistic crosses file boundaries -- only a fixed category vocabulary,
  a deterministic missing-value rule evaluated per-file, and a deterministic
  per-timestamp engineered feature).
- ``fit_standardizer`` / ``apply_standardizer``: the one step the task spec
  requires to be refit inside each cross-validation fold. Call ``fit_standardizer``
  on a training pool only, then ``apply_standardizer`` (never re-fit) on both the
  training pool and the held-out file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class MissingValueReport:
    case_id: str
    n_total_rows: int
    n_missing_rows: int
    frac_missing: float
    strategy: str  # "drop" or "forward_fill"
    n_rows_after: int
    n_residual_dropped: int = 0  # rows still NaN after ffill+bfill (edge case), then dropped


@dataclass
class RarityMergeRow:
    param: str
    category: object
    count: int
    frequency: float
    merged_into: str


@dataclass
class PrepReport:
    missing_value_reports: list[MissingValueReport]
    rarity_merges: list[RarityMergeRow]
    merge_map: dict
    vocab: dict
    feature_cols: list
    feature_source: dict
    numeric_cols_to_standardize: list


@dataclass
class Standardizer:
    stats: dict  # (car_model, col) -> (mean, std)


def handle_missing_values(
    long_df: pd.DataFrame, threshold: float = config.DROP_VS_FFILL_THRESHOLD
) -> tuple[pd.DataFrame, list[MissingValueReport]]:
    """Per file: drop rows with any missing base parameter; if that would lose
    more than `threshold` of the file's rows, forward-fill within each car's
    own time series instead (falling back to back-fill for any leading-edge
    gap with no prior value, then dropping the rare residual that still fails).
    """
    reports = []
    cleaned_parts = []

    for case_id, group in long_df.groupby("case_id", sort=False):
        group = group.sort_values(["car_number", "timestamp"]).reset_index(drop=True)
        is_missing = group[config.BASE_PARAMS].isna().any(axis=1)
        n_total = len(group)
        n_missing = int(is_missing.sum())
        frac_missing = n_missing / n_total if n_total else 0.0

        n_residual_dropped = 0
        if frac_missing <= threshold:
            strategy = "drop"
            cleaned = group.loc[~is_missing].copy()
        else:
            strategy = "forward_fill"
            cleaned = group.copy()
            filled = cleaned.groupby("car_number", sort=False)[config.BASE_PARAMS].transform(
                lambda s: s.ffill().bfill()
            )
            cleaned[config.BASE_PARAMS] = filled
            still_missing = cleaned[config.BASE_PARAMS].isna().any(axis=1)
            n_residual_dropped = int(still_missing.sum())
            if n_residual_dropped:
                cleaned = cleaned.loc[~still_missing]

        reports.append(
            MissingValueReport(
                case_id=case_id,
                n_total_rows=n_total,
                n_missing_rows=n_missing,
                frac_missing=frac_missing,
                strategy=strategy,
                n_rows_after=len(cleaned),
                n_residual_dropped=n_residual_dropped,
            )
        )
        cleaned_parts.append(cleaned)

    return pd.concat(cleaned_parts, axis=0, ignore_index=True), reports


def compute_rare_category_merge_map(
    long_df: pd.DataFrame,
    categorical_params: list[str] = config.CATEGORICAL_PARAMS,
    threshold: float = config.RARE_CATEGORY_THRESHOLD,
) -> tuple[dict[str, dict], list[RarityMergeRow]]:
    """Global (union-of-all-5-files) category frequency check. Safe to do once:
    the category vocabulary is a fixed domain fact, not a fitted distribution.
    """
    merge_map: dict[str, dict] = {}
    report_rows: list[RarityMergeRow] = []

    for param in categorical_params:
        non_na = long_df[param].dropna()
        total = len(non_na)
        counts = non_na.value_counts()
        param_map = {}
        for label, cnt in counts.items():
            freq = cnt / total if total else 0.0
            if freq < threshold:
                param_map[label] = config.RARE_CATEGORY_BUCKET
                report_rows.append(
                    RarityMergeRow(
                        param=param, category=label, count=int(cnt), frequency=freq,
                        merged_into=config.RARE_CATEGORY_BUCKET,
                    )
                )
        merge_map[param] = param_map

    return merge_map, report_rows


def apply_rare_category_merge(long_df: pd.DataFrame, merge_map: dict[str, dict]) -> pd.DataFrame:
    long_df = long_df.copy()
    for param, param_map in merge_map.items():
        if param_map:
            long_df[param] = long_df[param].replace(param_map)
    return long_df


def compute_onehot_vocab(
    long_df: pd.DataFrame, categorical_params: list[str] = config.CATEGORICAL_PARAMS
) -> dict[str, list]:
    return {param: sorted(long_df[param].dropna().unique().tolist()) for param in categorical_params}


def one_hot_encode(
    long_df: pd.DataFrame, vocab: dict[str, list]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    long_df = long_df.copy()
    onehot_cols_by_param: dict[str, list[str]] = {}
    for param, categories in vocab.items():
        cols_for_param = []
        for cat in categories:
            col_name = f"{param}={cat}"
            long_df[col_name] = (long_df[param] == cat).astype(float)
            cols_for_param.append(col_name)
        onehot_cols_by_param[param] = cols_for_param
    return long_df, onehot_cols_by_param


def engineer_cross_car_temp_deviation(long_df: pd.DataFrame) -> pd.DataFrame:
    """Each car's |deviation| from that timestamp's cross-car median Indoor
    Average Temperature (computed across all 8 cars at that timestamp, on the
    raw/unstandardized value). Deterministic per (case_id, timestamp) -- no fold
    dependence -- so safe to compute globally; only its z-scoring is fold-fit.
    """
    long_df = long_df.copy()
    median_per_ts = long_df.groupby(["case_id", "timestamp"])["Indoor Average Temperature"].transform(
        "median"
    )
    long_df[config.ENGINEERED_FEATURE] = (long_df["Indoor Average Temperature"] - median_per_ts).abs()
    return long_df


def assemble_feature_columns(
    onehot_cols_by_param: dict[str, list[str]], verbose: bool = True
) -> tuple[list[str], dict[str, str], list[str]]:
    feature_source: dict[str, str] = {}
    feature_cols: list[str] = []

    def add(cols, source_param):
        for c in cols:
            feature_cols.append(c)
            feature_source[c] = source_param

    add(onehot_cols_by_param["ACV Running Mode"], "ACV Running Mode")
    add(onehot_cols_by_param["ACV Setting Mode"], "ACV Setting Mode")
    add(["ACV Control Temperature (Cooling)_z"], "ACV Control Temperature (Cooling)")
    add(["ACV Control Temperature (Heating)_z"], "ACV Control Temperature (Heating)")
    add(["Indoor Average Temperature_z"], "Indoor Average Temperature")
    add(["Outside Temperature_z"], "Outside Temperature")
    add(onehot_cols_by_param["Load Halved"], "Load Halved")
    add(onehot_cols_by_param["ACV Information Valid"], "ACV Information Valid")
    add([f"{config.ENGINEERED_FEATURE}_z"], config.ENGINEERED_FEATURE)

    represented = set(feature_source.values())
    missing = [p for p in config.BASE_PARAMS if p not in represented]
    if missing:
        raise AssertionError(
            f"Feature assembly is missing base parameter(s): {missing}. "
            f"Parameters represented so far: {sorted(represented)}"
        )

    if verbose:
        print(f"Final feature list ({len(feature_cols)} features):")
        for c in feature_cols:
            print(f"  - {c}   [from base parameter: {feature_source[c]}]")

    numeric_cols_to_standardize = config.NUMERIC_PARAMS + [config.ENGINEERED_FEATURE]
    return feature_cols, feature_source, numeric_cols_to_standardize


def prepare_global(long_df: pd.DataFrame) -> tuple[pd.DataFrame, PrepReport]:
    """Run every preprocessing step that is safe to fit once globally. Returns
    the prepared long dataframe (base parameters + one-hot columns + the raw
    engineered feature; numeric "_z" columns are NOT yet present -- those are
    added per-fold by fit_standardizer/apply_standardizer) plus a report.
    """
    long_df, missing_reports = handle_missing_values(long_df)
    merge_map, rarity_rows = compute_rare_category_merge_map(long_df)

    print("Rare-category merges (global frequency < {:.1%}):".format(config.RARE_CATEGORY_THRESHOLD))
    if rarity_rows:
        for r in rarity_rows:
            print(f"  - {r.param!r}: {r.category!r} ({r.count} rows, {r.frequency:.4%}) -> {r.merged_into!r}")
    else:
        print("  (none)")

    long_df = apply_rare_category_merge(long_df, merge_map)
    vocab = compute_onehot_vocab(long_df)
    long_df, onehot_cols_by_param = one_hot_encode(long_df, vocab)
    long_df = engineer_cross_car_temp_deviation(long_df)
    feature_cols, feature_source, numeric_cols_to_standardize = assemble_feature_columns(onehot_cols_by_param)

    report = PrepReport(
        missing_value_reports=missing_reports,
        rarity_merges=rarity_rows,
        merge_map=merge_map,
        vocab=vocab,
        feature_cols=feature_cols,
        feature_source=feature_source,
        numeric_cols_to_standardize=numeric_cols_to_standardize,
    )
    return long_df, report


def prepare_new_file(
    long_df_raw: pd.DataFrame, merge_map: dict[str, dict], vocab: dict[str, list]
) -> tuple[pd.DataFrame, list[MissingValueReport]]:
    """Apply preprocessing to a brand-new (e.g. future test) file using the
    category vocabulary and rare-category merge map already fit on the 5
    labeled files -- never refit on the new file itself, so its one-hot
    feature columns line up exactly with the trained models' feature space.
    Missing-value handling is still evaluated per-file (it is a deterministic
    per-file rule, not a fitted statistic), same as any labeled file.
    """
    long_df, missing_reports = handle_missing_values(long_df_raw)
    long_df = apply_rare_category_merge(long_df, merge_map)
    long_df, _ = one_hot_encode(long_df, vocab)
    long_df = engineer_cross_car_temp_deviation(long_df)
    return long_df, missing_reports


def fit_standardizer(df_train_fold: pd.DataFrame, numeric_cols: list[str]) -> Standardizer:
    """Fit per-car_model mean/std on TRAINING-FOLD rows only."""
    stats = {}
    for car_model, group in df_train_fold.groupby("car_model"):
        for col in numeric_cols:
            mean = float(group[col].mean())
            std = float(group[col].std(ddof=0))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0  # constant-within-fold column: becomes all-zero after centering, not a div/0
            stats[(car_model, col)] = (mean, std)
    return Standardizer(stats=stats)


def apply_standardizer(df: pd.DataFrame, standardizer: Standardizer, numeric_cols: list[str]) -> pd.DataFrame:
    """Apply already-fitted per-car_model mean/std. Never refits -- safe to call
    on the held-out fold using stats fit only on the training pool.
    """
    df = df.copy()
    for col in numeric_cols:
        z_col = f"{col}_z"
        df[z_col] = np.nan
        for car_model in df["car_model"].unique():
            key = (car_model, col)
            if key not in standardizer.stats:
                raise KeyError(
                    f"No fitted standardizer stats for car_model={car_model!r}, col={col!r}. "
                    "The training pool for this fold must contain at least one row of every "
                    "car_model present in the data being scored."
                )
            mean, std = standardizer.stats[key]
            mask = (df["car_model"] == car_model).to_numpy()
            df.loc[mask, z_col] = (df.loc[mask, col].to_numpy() - mean) / std
    return df
