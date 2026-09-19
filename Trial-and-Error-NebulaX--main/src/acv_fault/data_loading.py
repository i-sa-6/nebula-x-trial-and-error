"""Load raw ACV .xlsx files and melt them into long (case, timestamp, car) format.

Column-naming reality (verified against the actual files, not assumed from the
spec): each file has one sheet ("Sheet1"), one row per timestamp, and columns
named "Car {01..08} - {Parameter}" plus "Car model" / "Train number" / "Time".
Model A files (cases 01/02/03) use "Outdoor Average Temperature"; Model C files
(cases 05/06) use "Outside Temperature Sensor Reading". Both normalize here to
one column: "Outside Temperature".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class LoadReport:
    """Per-file facts collected while loading, echoed back for the write-up."""

    filename: str
    car_model: str
    n_timestamps: int
    n_long_rows: int
    outside_temp_end_car_fills: int  # rows where a non-end car borrowed the end-car reading


def load_labels(labels_csv=config.LABELS_FILE) -> pd.DataFrame:
    """Read filename,faulty_car and validate faulty_car is a known 2-digit car id."""
    df = pd.read_csv(labels_csv, dtype=str)
    expected_cols = {"filename", "faulty_car"}
    if set(df.columns) != expected_cols:
        raise ValueError(f"labels.csv has unexpected columns: {df.columns.tolist()}")
    bad = df.loc[~df["faulty_car"].isin(config.CAR_NUMBERS)]
    if len(bad):
        raise ValueError(f"labels.csv has faulty_car values outside {config.CAR_NUMBERS}: {bad}")
    return df.reset_index(drop=True)


def _find_car_columns(columns: list[str]) -> dict[str, list[str]]:
    """Map each base parameter name (canonical, alias-normalized) -> its 8 raw column names."""
    by_param: dict[str, list[str]] = {}
    for c in columns:
        m = re.match(r"^Car (\d{2}) - (.+)$", c)
        if not m:
            continue
        param = m.group(2)
        if param in config.TEMPERATURE_ALIASES:
            param = "Outside Temperature"
        by_param.setdefault(param, []).append(c)
    return by_param


def melt_case(df_wide: pd.DataFrame, case_id: str) -> pd.DataFrame:
    """Wide (1 row/timestamp, 8-cars-wide) -> long (1 row per (timestamp, car))."""
    missing_meta = [c for c in config.METADATA_COLS if c not in df_wide.columns]
    if missing_meta:
        raise ValueError(f"{case_id}: missing expected metadata columns {missing_meta}")

    by_param = _find_car_columns(list(df_wide.columns))
    missing_params = [p for p in config.BASE_PARAMS if p not in by_param]
    if missing_params:
        raise ValueError(
            f"{case_id}: could not find any source column for parameter(s) {missing_params}. "
            f"Columns seen: {sorted(df_wide.columns)}"
        )

    car_frames = []
    for car_num in config.CAR_NUMBERS:
        car_frame = pd.DataFrame(
            {
                "case_id": case_id,
                "timestamp": pd.to_datetime(df_wide["Time"]),
                "car_number": car_num,
                "car_model": df_wide["Car model"].astype(str),
                "train_number": df_wide["Train number"],
            }
        )
        for param in config.BASE_PARAMS:
            col_name = f"Car {car_num} - {param}"
            # Model C's "Outside Temperature" source column name differs; resolve via alias map.
            if col_name not in df_wide.columns:
                candidates = [c for c in by_param[param] if c.startswith(f"Car {car_num} - ")]
                if not candidates:
                    raise ValueError(f"{case_id}: no '{param}' column for car {car_num}")
                col_name = candidates[0]
            raw = df_wide[col_name]
            if param in config.NUMERIC_PARAMS:
                # Coerces the "Invalid" sentinel string (Model C, non-end cars) to NaN.
                car_frame[param] = pd.to_numeric(raw, errors="coerce")
            else:
                # Normalize whatever missing-value sentinel the source dtype uses
                # (pd.NA, None, NaN) to plain np.nan, uniformly, as a plain object column.
                raw_obj = raw.astype(object)
                car_frame[param] = raw_obj.where(pd.notna(raw_obj), np.nan)
        car_frames.append(car_frame)

    long_df = pd.concat(car_frames, axis=0, ignore_index=True)
    long_df = long_df.sort_values(["timestamp", "car_number"]).reset_index(drop=True)

    car_models = long_df["car_model"].unique()
    if len(car_models) != 1:
        raise ValueError(f"{case_id}: expected one car_model per file, found {car_models}")

    return long_df


def apply_outside_temperature_fix(long_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Broadcast end-car (01/08) 'Outside Temperature' readings to cars missing their own.

    Ambient outdoor air temperature is physically a train-level quantity, and
    verification against the real files found that Model C's non-end cars (02-07)
    report the literal string "Invalid" for this sensor 100% of the time (never a
    real reading) while cars 01/08 always do. Per user decision: at each
    timestamp, fill any car's missing/invalid reading with the mean of whichever
    end car(s) have a valid reading; a car's own valid reading is always kept
    (this is a true no-op for Model A, where every car already reports its own
    value). This is a deterministic, within-timestamp, within-file sensor-fusion
    step using only redundant information already present in that same row's
    timestamp -- not a fitted statistic -- so, like the one-hot category
    vocabulary, it is safe to apply once globally rather than per CV fold.
    """
    long_df = long_df.copy()
    col = "Outside Temperature"

    end_car_rows = long_df[long_df["car_number"].isin(config.END_CARS)]
    ref = end_car_rows.groupby(["case_id", "timestamp"])[col].mean().rename("_ref_outside_temp")

    long_df = long_df.merge(ref, on=["case_id", "timestamp"], how="left")
    needs_fill = long_df[col].isna() & long_df["_ref_outside_temp"].notna()
    n_filled = int(needs_fill.sum())
    long_df.loc[needs_fill, col] = long_df.loc[needs_fill, "_ref_outside_temp"]
    long_df = long_df.drop(columns=["_ref_outside_temp"])
    return long_df, n_filled


def load_all_cases(
    labels_csv=config.LABELS_FILE, train_dir=config.TRAIN_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, list[LoadReport]]:
    """Load + melt + fix every file listed in labels.csv.

    Returns (long_df covering all files, labels_df, per-file LoadReport list).
    long_df carries a `faulty_car` column purely as evaluation metadata -- it is
    never a model input feature.
    """
    labels_df = load_labels(labels_csv)
    all_long = []
    reports = []

    for _, row in labels_df.iterrows():
        filename, faulty_car = row["filename"], row["faulty_car"]
        path = train_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found: {path}")
        case_id = filename
        df_wide = pd.read_excel(path, sheet_name="Sheet1")
        long_df = melt_case(df_wide, case_id)
        long_df, n_filled = apply_outside_temperature_fix(long_df)
        long_df["faulty_car"] = faulty_car

        reports.append(
            LoadReport(
                filename=filename,
                car_model=str(long_df["car_model"].iloc[0]),
                n_timestamps=long_df["timestamp"].nunique(),
                n_long_rows=len(long_df),
                outside_temp_end_car_fills=n_filled,
            )
        )
        all_long.append(long_df)

    full_long = pd.concat(all_long, axis=0, ignore_index=True)
    return full_long, labels_df, reports
