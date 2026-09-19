"""
Dataset builder: Extracts physics-informed features from all available train files
and caches them into a clean feature table (train_features.csv).
Uses multi-processing for high-speed extraction.
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import TRAIN_DIR, TRAIN_LABELS_PATH, DATA_DIR
from src.feature_extraction import extract_features_from_df

FEATURES_OUTPUT_PATH = os.path.join(DATA_DIR, "train_features.csv")

def process_file_worker(args):
    """
    Worker function to process a single CSV file.
    """
    fname, label = args
    fpath = os.path.join(TRAIN_DIR, fname)
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
        feats = extract_features_from_df(df, filename=fname)
        feats["label"] = label
        return feats
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None

def build_features_dataset(max_workers=8):
    """
    Builds and saves the train features dataframe using multi-processing.
    """
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    file_label_pairs = [(row['filename'], row['label']) for _, row in labels_df.iterrows()]
    
    total = len(file_label_pairs)
    print(f"[*] Starting parallel feature extraction for {total} files using {max_workers} worker processes...")
    start_time = time.time()
    
    feature_rows = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file_worker, pair): pair[0] for pair in file_label_pairs}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res is not None:
                feature_rows.append(res)
            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 0.1)
                print(f"[+] [{completed}/{total}] files extracted ({elapsed:.1f}s, {rate:.1f} files/s)")
                
    out_df = pd.DataFrame(feature_rows)
    # Sort by filename to keep consistent order
    out_df = out_df.sort_values("filename").reset_index(drop=True)
    out_df.to_csv(FEATURES_OUTPUT_PATH, index=False)
    
    elapsed_total = time.time() - start_time
    print(f"\n[✓] Feature extraction finished in {elapsed_total:.1f}s.")
    print(f"[✓] Feature dataset shape: {out_df.shape} saved to {FEATURES_OUTPUT_PATH}")
    print("\nClass breakdown in extracted features:\n", out_df['label'].value_counts())
    return out_df

if __name__ == "__main__":
    build_features_dataset()
