"""
Official Inference CLI Script for Problem Statement 3 - Rail Corrugation Subsystem.
Complies with official hackathon specifications:
Usage:
    python predict.py --input <path_to_test_dir> --output <path_to_output_csv>
"""
import os
import sys
import argparse
import pickle
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from src.feature_extraction import extract_features_from_df
from src.config import MODELS_DIR, CLASSES
from src.models import EnsembleClassifier


DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "rail_corrugation_champion.pkl")

def process_single_file(fpath, feature_cols, scaler, model, class_weights):
    """
    Extracts features from a single CSV and predicts health state.
    """
    fname = os.path.basename(fpath)
    try:
        df = pd.read_csv(fpath)
        feats = extract_features_from_df(df, filename=fname)
        
        # Build feature vector
        x_vec = np.array([feats.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
        x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=1e6, neginf=-1e6)
        x_scaled = scaler.transform(x_vec)
        
        probs = model.predict_proba(x_scaled)[0]
        weighted_probs = probs * class_weights
        pred_idx = int(np.argmax(weighted_probs))
        pred_label = CLASSES[pred_idx]
        confidence = float(probs[pred_idx] / (np.sum(probs) + 1e-9))
        
        return {
            "file_id": fname,
            "prediction": pred_label,
            "confidence": confidence,
            "speed_kmh": feats.get("speed_kmh", 0.0),
            "sdi_rms": feats.get("sdi_rms", 0.0),
            "prob_normal": float(probs[0]),
            "prob_side1": float(probs[1]),
            "prob_side2": float(probs[2])
        }
    except Exception as e:
        print(f"[-] Error predicting on {fname}: {e}")
        return {
            "file_id": fname,
            "prediction": "Normal",
            "confidence": 0.0,
            "speed_kmh": 0.0,
            "sdi_rms": 0.0,
            "prob_normal": 1.0,
            "prob_side1": 0.0,
            "prob_side2": 0.0
        }

def run_inference(input_path, output_path, model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first.")
        
    print(f"[*] Loading model bundle from {model_path}...")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
        
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    class_weights = bundle.get("class_weights", np.array([1.0, 1.2, 1.2]))
    
    # Collect input files
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.csv")), key=lambda p: [int(s) if s.isdigit() else s for s in os.path.basename(p).replace('.', '_').split('_')])
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise ValueError(f"Input path {input_path} does not exist.")
        
    print(f"[*] Processing {len(files)} test files from {input_path}...")
    
    results = []
    for i, fpath in enumerate(files, 1):
        res = process_single_file(fpath, feature_cols, scaler, model, class_weights)
        results.append(res)
        if i % 10 == 0 or i == len(files):
            print(f"[+] Predicted {i}/{len(files)}: {res['file_id']} -> {res['prediction']} ({res['confidence']*100:.1f}%)")
            
    res_df = pd.DataFrame(results)
    
    # Official submission format: exactly file_id and prediction
    submission_df = res_df[["file_id", "prediction"]]
    
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    submission_df.to_csv(output_path, index=False)
    print(f"\n[✓] Official predictions saved to {output_path} ({len(submission_df)} rows).")
    
    # Also save detailed diagnostics alongside for review
    diag_path = output_path.replace(".csv", "_detailed.csv")
    res_df.to_csv(diag_path, index=False)
    print(f"[✓] Detailed diagnostic report saved to {diag_path}.")
    
    print("\nPrediction Summary:")
    print(submission_df["prediction"].value_counts())
    return submission_df, res_df

def main():
    parser = argparse.ArgumentParser(description="Rail Corrugation Subsystem Inference CLI")
    parser.add_argument("--input", required=True, help="Path to input test directory or CSV file")
    parser.add_argument("--output", default="submission/rail_predictions.csv", help="Path to output predictions CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model bundle")
    args = parser.parse_args()
    
    run_inference(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
