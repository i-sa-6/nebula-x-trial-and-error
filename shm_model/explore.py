import pandas as pd
import numpy as np
from pathlib import Path


TRAIN_DIR = Path("PS3/02_Datasets/SHM/Train")
LABEL_FILE = Path("PS3/02_Datasets/SHM/Train_Labels.csv")

def extract_features(df):
    stress = df["stress"]

    return {
        "mean": stress.mean(),
        "std": stress.std(),
        "min": stress.min(),
        "max": stress.max(),
        "range": stress.max() - stress.min(),
        "rms": np.sqrt(np.mean(stress**2)),
        "median": stress.median(),
        "p95": stress.quantile(0.95),
        "p99": stress.quantile(0.99),
    }

all_features = []

for file in sorted(TRAIN_DIR.glob("train*.csv")):
    df = pd.read_csv(
        file,
        header=None,
        names=["stress"]
    )

    features = extract_features(df)
    features["filename"] = file.name

    all_features.append(features)

feature_table = pd.DataFrame(all_features)

labels = pd.read_csv(LABEL_FILE)

training_data = feature_table.merge(
    labels,
    on="filename"
)

X = training_data.drop(
    columns=["filename", "damage"]
)

y = training_data["damage"]

print("Training data:")
print(training_data.head())

print("\nTraining data shape:")
print(training_data.shape)

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

import pandas as pd
import numpy as np
from pathlib import Path


# -----------------------------
# Paths
# -----------------------------
TRAIN_DIR = Path("PS3/02_Datasets/SHM/Train")
LABEL_FILE = Path("PS3/02_Datasets/SHM/Train_Labels.csv")


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(df):
    stress = df["stress"]

    mean = stress.mean()
    std = stress.std()

    features = {
        "mean": mean,
        "std": std,
        "min": stress.min(),
        "max": stress.max(),
        "range": stress.max() - stress.min(),
        "rms": np.sqrt(np.mean(stress**2)),
        "median": stress.median(),
        "p95": stress.quantile(0.95),
        "p99": stress.quantile(0.99),

        # New features
        "mean_abs": np.mean(np.abs(stress)),
        "skew": stress.skew(),
        "kurtosis": stress.kurt(),
        "zero_crossings": np.sum(np.diff(np.sign(stress)) != 0),
        "above_mean_plus_2std": np.sum(stress > mean + 2 * std),
        "below_mean_minus_2std": np.sum(stress < mean - 2 * std),
    }

    return features


# -----------------------------
# Extract features from all files
# -----------------------------
all_features = []

for file in sorted(TRAIN_DIR.glob("train*.csv")):
    df = pd.read_csv(
        file,
        header=None,
        names=["stress"]
    )

    features = extract_features(df)
    features["filename"] = file.name

    all_features.append(features)

feature_table = pd.DataFrame(all_features)


# -----------------------------
# Add damage labels
# -----------------------------
labels = pd.read_csv(LABEL_FILE)

training_data = feature_table.merge(
    labels,
    on="filename"
)


# -----------------------------
# Prepare machine-learning data
# -----------------------------
X = training_data.drop(
    columns=["filename", "damage"]
)

y = training_data["damage"]


# -----------------------------
# Check results
# -----------------------------
print("Training data:")
print(training_data.head())

print("\nTraining data shape:")
print(training_data.shape)

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())
