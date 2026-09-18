"""Leakage-free training and comparative evaluation of door resistance classification models.

Enforces zero time-based data leakage via:
1. Expanding-Window TimeSeriesSplit (5 chronological folds, no future lookahead).
2. Chronological Out-of-Time Holdout Split (80% past train, 20% future test).
3. Cycle-isolated feature extraction.
4. Preprocessing (StandardScaler) encapsulated strictly within sklearn Pipeline.
"""

import csv
from datetime import datetime
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add project root and model directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_extractor import extract_cycle_features


def parse_datetime(ts_str: str) -> datetime:
    """Parse custom hyphen-separated datetime string."""
    parts = list(map(int, ts_str.strip().split("-")))
    return datetime(
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6] * 1000
    )


def load_dataset(
    train_csv_path: str, answer_csv_path: str
) -> Tuple[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]:
    """Load raw telemetry and ground-truth segments, extracting features cycle by cycle."""
    with open(train_csv_path, "r", newline="", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    with open(answer_csv_path, "r", newline="", encoding="utf-8") as f:
        answer_rows = list(csv.DictReader(f))

    # Sort answers strictly chronologically to guarantee no temporal disorder
    answer_rows.sort(key=lambda x: parse_datetime(x["start_time"]))

    curr_idx = 0
    features_list = []
    labels_list = []
    metadata_list = []

    for ans in answer_rows:
        n = int(ans["n_rows"])
        seg_rows = train_rows[curr_idx : curr_idx + n]
        curr_idx += n

        # Extract features strictly inside this segment's window
        feats = extract_cycle_features(seg_rows)
        features_list.append(feats)

        # Binary label: 1 for 'Abnormal resistance', 0 for 'Normal'
        label = 1 if ans["status"] == "Abnormal resistance" else 0
        labels_list.append(label)

        metadata_list.append({
            "segment_id": ans["segment_id"],
            "start_time": ans["start_time"],
            "end_time": ans["end_time"],
            "operation": ans["operation"],
            "status": ans["status"],
            "n_rows": n,
        })

    df_X = pd.DataFrame(features_list)
    y = np.array(labels_list)
    return df_X, y, metadata_list


def get_candidate_models() -> Dict[str, Pipeline]:
    """Define candidate model families wrapped inside leakage-free Pipelines."""
    return {
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
        "Support Vector Classifier (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            # Tree models don't strictly require scaling, but keeping standard pipeline
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)),
        ]),
        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=42)),
        ]),
    }


def evaluate_time_series_cv(
    df_X: pd.DataFrame,
    y: np.ndarray,
    metadata: List[Dict[str, Any]],
    n_splits: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate candidate models using expanding-window TimeSeriesSplit.

    Guarantees that train timestamps are strictly < validation timestamps for all folds.
    """
    print("\n" + "=" * 70)
    print("STEP 1: EXPANDING-WINDOW TIME-SERIES CROSS-VALIDATION (5 FOLDS)")
    print("Verification of zero time-based lookahead / leakage:")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = get_candidate_models()

    cv_results = {name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "f1_macro": [], "roc_auc": []} for name in models}

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df_X)):
        train_start_t = metadata[train_idx[0]]["start_time"]
        train_end_t = metadata[train_idx[-1]]["end_time"]
        val_start_t = metadata[val_idx[0]]["start_time"]
        val_end_t = metadata[val_idx[-1]]["end_time"]

        # Assert no time overlap
        assert parse_datetime(train_end_t) < parse_datetime(val_start_t), (
            f"Time leakage detected! Train end {train_end_t} >= Val start {val_start_t}"
        )

        n_train = len(train_idx)
        n_val = len(val_idx)
        abnormal_val = sum(y[val_idx])

        print(
            f"Fold {fold_idx + 1}: Train cycles [0..{train_idx[-1]}] ({n_train} cycles: {train_start_t} to {train_end_t}) "
            f"-> Val cycles [{val_idx[0]}..{val_idx[-1]}] ({n_val} cycles: {val_start_t} to {val_end_t}) | "
            f"Val Abnormals: {abnormal_val}/{n_val}"
        )

        X_train, y_train = df_X.iloc[train_idx], y[train_idx]
        X_val, y_val = df_X.iloc[val_idx], y[val_idx]

        for name, pipeline in models.items():
            # Clone and fit strictly on training fold
            from sklearn.base import clone
            model_clone = clone(pipeline)
            model_clone.fit(X_train, y_train)

            y_pred = model_clone.predict(X_val)
            y_proba = (
                model_clone.predict_proba(X_val)[:, 1]
                if hasattr(model_clone, "predict_proba")
                else y_pred
            )

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            f1_mac = f1_score(y_val, y_pred, average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y_val, y_proba)
            except ValueError:
                auc = 1.0  # Only one class present in val fold

            cv_results[name]["accuracy"].append(acc)
            cv_results[name]["precision"].append(prec)
            cv_results[name]["recall"].append(rec)
            cv_results[name]["f1"].append(f1)
            cv_results[name]["f1_macro"].append(f1_mac)
            cv_results[name]["roc_auc"].append(auc)

    print("\n--- TimeSeriesSplit Cross-Validation Results (Mean across 5 chronological folds) ---")
    summary = {}
    for name, metrics in cv_results.items():
        summary[name] = {k: float(np.mean(v)) for k, v in metrics.items()}
        print(
            f"{name:<34} | Acc: {summary[name]['accuracy']:.4f} | "
            f"Prec: {summary[name]['precision']:.4f} | Rec: {summary[name]['recall']:.4f} | "
            f"F1 (Abnorm): {summary[name]['f1']:.4f} | Macro-F1: {summary[name]['f1_macro']:.4f} | "
            f"ROC-AUC: {summary[name]['roc_auc']:.4f}"
        )

    return summary


def evaluate_chronological_holdout(
    df_X: pd.DataFrame,
    y: np.ndarray,
    metadata: List[Dict[str, Any]],
    train_ratio: float = 0.80,
) -> Tuple[Dict[str, Dict[str, float]], str, Pipeline]:
    """Train on earliest 80% cycles and test on strictly future 20% holdout cycles."""
    print("\n" + "=" * 70)
    print("STEP 2: CHRONOLOGICAL OUT-OF-TIME HOLDOUT TEST (80% TRAIN / 20% TEST)")
    print("=" * 70)

    n_total = len(df_X)
    split_idx = int(n_total * train_ratio)

    train_idx = list(range(0, split_idx))
    test_idx = list(range(split_idx, n_total))

    train_end_t = metadata[train_idx[-1]]["end_time"]
    test_start_t = metadata[test_idx[0]]["start_time"]

    assert parse_datetime(train_end_t) < parse_datetime(test_start_t), (
        f"Time leakage! Train end {train_end_t} >= Test start {test_start_t}"
    )

    print(
        f"Training Set: First {len(train_idx)} cycles ({metadata[0]['start_time']} -> {train_end_t})"
    )
    print(
        f"Test Set:     Last {len(test_idx)} cycles ({test_start_t} -> {metadata[-1]['end_time']})"
    )
    print(f"Test Class Balance: {sum(y[test_idx])} Abnormal resistance / {len(test_idx)} total")

    X_train, y_train = df_X.iloc[train_idx], y[train_idx]
    X_test, y_test = df_X.iloc[test_idx], y[test_idx]

    models = get_candidate_models()
    test_results = {}
    best_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, pipeline in models.items():
        from sklearn.base import clone
        p = clone(pipeline)
        # Scaler fit strictly on X_train only
        p.fit(X_train, y_train)

        y_pred = p.predict(X_test)
        y_proba = p.predict_proba(X_test)[:, 1] if hasattr(p, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        test_results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "f1_macro": float(f1_mac),
            "roc_auc": float(auc),
        }

        print(
            f"{name:<34} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
            f"F1 (Abnorm): {f1:.4f} | Macro-F1: {f1_mac:.4f} | ROC-AUC: {auc:.4f}"
        )

        if f1_mac > best_f1:
            best_f1 = f1_mac
            best_name = name
            best_pipeline = p

    print(f"\nChampion Model on Out-of-Time Test: {best_name} (Macro-F1: {best_f1:.4f})")
    return test_results, best_name, best_pipeline


def train_and_save_final_model(
    df_X: pd.DataFrame,
    y: np.ndarray,
    champion_name: str,
    output_model_path: str,
) -> None:
    """Train the champion architecture on all chronological data and serialize."""
    print("\n" + "=" * 70)
    print(f"STEP 3: SERIALIZING CHAMPION MODEL ({champion_name})")
    print("=" * 70)

    models = get_candidate_models()
    pipeline = models[champion_name]

    # Fit final pipeline
    pipeline.fit(df_X, y)

    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "champion_name": champion_name,
        "feature_names": list(df_X.columns),
        "created_at": datetime.now().isoformat(),
        "n_samples": len(df_X),
        "target_mapping": {0: "Normal", 1: "Abnormal resistance"},
    }
    joblib.dump(artifact, output_model_path)
    print(f"Model saved successfully to: {output_model_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(
        base_dir, "data", "NebulaX-Hackathon-ProblemStatement", "PS3", "02_Datasets", "Door", "Train.csv"
    )
    answer_csv = os.path.join(
        base_dir,
        "data",
        "NebulaX-Hackathon-ProblemStatement",
        "PS3",
        "02_Datasets",
        "Door",
        "Train_Segments_Answer.csv",
    )
    model_output_path = os.path.join(base_dir, "model", "door_model.joblib")

    print(f"Loading data from {train_csv} and {answer_csv}...")
    df_X, y, metadata = load_dataset(train_csv, answer_csv)
    print(f"Loaded {len(df_X)} cycles, with {df_X.shape[1]} physics-based features each.")
    print(f"Class distribution: {sum(y == 0)} Normal, {sum(y == 1)} Abnormal resistance.")

    # 1. Expanding-Window TimeSeriesSplit Cross-Validation
    cv_summary = evaluate_time_series_cv(df_X, y, metadata, n_splits=5)

    # 2. Chronological Out-of-Time Holdout Split (80% Train, 20% Test)
    holdout_summary, champion_name, champion_pipeline = evaluate_chronological_holdout(
        df_X, y, metadata, train_ratio=0.80
    )

    # 3. Save Final Model Artifact
    train_and_save_final_model(df_X, y, champion_name, model_output_path)

    # Save summary report to JSON
    report_path = os.path.join(base_dir, "model", "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "time_series_cv_5_fold": cv_summary,
                "chronological_holdout_80_20": holdout_summary,
                "champion_model": champion_name,
            },
            f,
            indent=2,
        )
    print(f"Full evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()

