import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rainflow


# -----------------------------
# Feature extraction
# -----------------------------
from features import extract_features


# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True,
    help="Folder containing SHM CSV files"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output CSV file"
)

args = parser.parse_args()


# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "final_model.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------------
# Predict
# -----------------------------
input_folder = Path(args.input)

results = []

for file in sorted(input_folder.glob("*.csv")):

    df = pd.read_csv(
        file,
        header=None,
        names=["stress"]
    )

    features = extract_features(df)

    X_new = pd.DataFrame([features])

    prediction = model.predict(X_new)[0]

    results.append({
        "file_id": file.name,
        "prediction": prediction
    })


# -----------------------------
# Save output
# -----------------------------
output_df = pd.DataFrame(results)

output_df.to_csv(
    args.output,
    index=False
)

print(f"Predictions saved to {args.output}")