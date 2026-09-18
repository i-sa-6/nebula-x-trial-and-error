"""Inference script for Door Resistance Classification.

Takes in input telemetry data (.csv) and segments (.csv), extracts features,
runs the trained model, and produces predictions in the competition schema:
start_time,end_time,prediction
"""

import argparse
import csv
from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

# Add paths to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from door_segmentation import segment_door_data
from model.feature_extractor import extract_cycle_features


def parse_datetime(ts_str: str) -> datetime:
    """Parse custom hyphen-separated datetime string."""
    parts = list(map(int, ts_str.strip().split("-")))
    return datetime(
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6] * 1000
    )


def load_telemetry_rows(data_csv_path: str) -> List[Dict[str, str]]:
    """Load raw telemetry rows from CSV."""
    with open(data_csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_segments_from_file_or_data(
    telemetry_rows: List[Dict[str, str]],
    segments_csv_path: Optional[str] = None,
) -> List[List[Dict[str, str]]]:
    """Retrieve segment slices from a segments CSV or by auto-segmenting telemetry."""
    if segments_csv_path and os.path.exists(segments_csv_path):
        with open(segments_csv_path, "r", newline="", encoding="utf-8") as f:
            seg_reader = list(csv.DictReader(f))

        # Check if segments CSV contains n_rows
        if seg_reader and "n_rows" in seg_reader[0]:
            seg_slices = []
            curr_idx = 0
            for r in seg_reader:
                n = int(r["n_rows"])
                seg_slices.append(telemetry_rows[curr_idx : curr_idx + n])
                curr_idx += n
            return seg_slices
        elif seg_reader and "start_time" in seg_reader[0] and "end_time" in seg_reader[0]:
            # Match by start_time and end_time
            # Pre-index telemetry rows by Datetime
            time_to_idx = {row["Datetime"]: idx for idx, row in enumerate(telemetry_rows)}
            seg_slices = []
            for r in seg_reader:
                s_idx = time_to_idx.get(r["start_time"])
                e_idx = time_to_idx.get(r["end_time"])
                if s_idx is not None and e_idx is not None:
                    seg_slices.append(telemetry_rows[s_idx : e_idx + 1])
                else:
                    raise ValueError(f"Could not locate timestamp range: {r['start_time']} to {r['end_time']}")
            return seg_slices

    # Fallback to automated segmentation if no segments CSV supplied
    print("No segment CSV provided (or found); running automated segmentation on telemetry...")
    return segment_door_data(telemetry_rows)


def predict_door_resistance(
    data_csv_path: str,
    segments_csv_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_csv_path: Optional[str] = "door_predictions.csv",
) -> List[Dict[str, str]]:
    """Predict Normal vs Abnormal resistance for each segment in the data stream.

    Args:
        data_csv_path: Path to raw continuous telemetry CSV (e.g. Train.csv or Test.csv).
        segments_csv_path: Optional path to segments CSV.
        model_path: Path to serialized model artifact (.joblib).
        output_csv_path: Path to output predictions CSV.

    Returns:
        List of prediction dictionaries: [{'start_time', 'end_time', 'prediction'}]
    """
    if model_path is None:
        model_path = os.path.join(BASE_DIR, "model", "door_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_eval.py first.")

    model_artifact = joblib.load(model_path)
    pipeline = model_artifact["pipeline"]
    target_mapping = model_artifact.get("target_mapping", {0: "Normal", 1: "Abnormal resistance"})

    telemetry_rows = load_telemetry_rows(data_csv_path)
    cycle_slices = get_segments_from_file_or_data(telemetry_rows, segments_csv_path)

    predictions = []
    features_list = []
    metadata_list = []

    for seg_rows in cycle_slices:
        if not seg_rows:
            continue
        start_time = seg_rows[0]["Datetime"]
        end_time = seg_rows[-1]["Datetime"]

        feats = extract_cycle_features(seg_rows)
        features_list.append(feats)
        metadata_list.append((start_time, end_time))

    df_X = pd.DataFrame(features_list)
    y_pred = pipeline.predict(df_X)

    for (start_time, end_time), pred_int in zip(metadata_list, y_pred):
        label_str = target_mapping.get(pred_int, "Normal")
        predictions.append({
            "start_time": start_time,
            "end_time": end_time,
            "prediction": label_str,
        })

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["start_time", "end_time", "prediction"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in predictions:
                writer.writerow(row)
        print(f"Predictions saved to {output_csv_path} ({len(predictions)} segments).")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run trained Door Resistance Classification model on telemetry data and segments."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to input raw telemetry CSV (e.g. Test.csv or Train.csv).",
    )
    parser.add_argument(
        "--segments",
        "-s",
        type=str,
        default=None,
        help="Path to segments CSV (e.g. Train_Segments_Answer.csv). If omitted, automatically segments input.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Path to trained model .joblib file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="door_predictions.csv",
        help="Output CSV path (default: door_predictions.csv).",
    )

    args = parser.parse_args()

    results = predict_door_resistance(
        data_csv_path=args.input,
        segments_csv_path=args.segments,
        model_path=args.model,
        output_csv_path=args.output,
    )
    print(f"Generated {len(results)} predictions in '{args.output}'.")


if __name__ == "__main__":
    main()

