import joblib
import rainflow
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Paths
# -----------------------------
TRAIN_DIR = Path("PS3/02_Datasets/SHM/Train")
LABEL_FILE = Path("PS3/02_Datasets/SHM/Train_Labels.csv")


# -----------------------------
# Feature extraction
# -----------------------------
from features import extract_features

# -----------------------------
# Build feature table
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

labels = pd.read_csv(LABEL_FILE)

training_data = feature_table.merge(
    labels,
    on="filename"
)

X = training_data.drop(
    columns=["filename", "damage"]
)

y = training_data["damage"]


# -----------------------------
# Models
# -----------------------------
models = {
    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# -----------------------------
# Cross-validation
# -----------------------------
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

mape_scorer = make_scorer(
    mean_absolute_percentage_error,
    greater_is_better=False
)


# -----------------------------
# Compare models
# -----------------------------
print("Model comparison:\n")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=mape_scorer
    )

    mean_mape = -scores.mean()
    std_mape = scores.std()

    competition_score = max(
        0,
        1 - mean_mape
    )

    print(name)
    print(f"Mean CV MAPE: {mean_mape:.4f}")
    print(f"Std CV MAPE:  {std_mape:.4f}")
    print(f"Approx. competition score: {competition_score:.4f}")
    print("-" * 40)

final_model = ExtraTreesRegressor(
    n_estimators=300,
    random_state=42
)

final_model.fit(X, y)

joblib.dump(
    final_model,
    "shm_model/final_model.pkl"
)

print("\nFinal Extra Trees model saved.")