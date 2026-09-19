import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    cross_val_predict
)

from sklearn.metrics import (
    make_scorer,
    mean_absolute_percentage_error
)

from sklearn.linear_model import Ridge

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features


# -----------------------------
# Paths
# -----------------------------
TRAIN_DIR = Path("PS3/02_Datasets/SHM/Train")
LABEL_FILE = Path("PS3/02_Datasets/SHM/Train_Labels.csv")


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
# Standard models
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

    "Log Extra Trees": TransformedTargetRegressor(
        regressor=ExtraTreesRegressor(
            n_estimators=300,
            random_state=42
        ),
        func=np.log,
        inverse_func=np.exp
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        random_state=42
    ),

    "SVR RBF": make_pipeline(
    StandardScaler(),
    SVR(
        kernel="rbf",
        C=10,
        epsilon=0.01,
        gamma="scale"
        )
    ),

    "Log Gradient Boosting": TransformedTargetRegressor(
    regressor=GradientBoostingRegressor(
        random_state=42
    ),
    func=np.log,
    inverse_func=np.exp
    ),
    
}


# -----------------------------
# Compare standard models
# -----------------------------
print("\nSTANDARD MODEL COMPARISON")
print("=" * 50)

model_scores = {}

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

    model_scores[name] = mean_mape

    print(f"\n{name}")
    print(f"Mean CV MAPE: {mean_mape:.4f}")
    print(f"Std CV MAPE:  {std_mape:.4f}")
    print(
        f"Approx. competition score: "
        f"{competition_score:.4f}"
    )


# -----------------------------
# Best standard model
# -----------------------------
best_model_name = min(
    model_scores,
    key=model_scores.get
)

best_model = models[best_model_name]

print("\n" + "=" * 50)
print("BEST STANDARD MODEL")
print("=" * 50)

print(f"Model: {best_model_name}")
print(
    f"CV MAPE: "
    f"{model_scores[best_model_name]:.4f}"
)

print(
    f"Approx. score: "
    f"{max(0, 1 - model_scores[best_model_name]):.4f}"
)


# -----------------------------
# Out-of-fold validation
# using best standard model
# -----------------------------
oof_predictions = cross_val_predict(
    best_model,
    X,
    y,
    cv=cv
)

validation_results = pd.DataFrame({
    "filename": training_data["filename"],
    "actual": y,
    "predicted": oof_predictions
})

validation_results["percentage_error"] = (
    np.abs(
        validation_results["actual"]
        - validation_results["predicted"]
    )
    / validation_results["actual"]
) * 100

# -----------------------------
# Error by damage range
# -----------------------------
validation_results["damage_group"] = pd.cut(
    validation_results["actual"],
    bins=[0, 0.10, 0.50, np.inf],
    labels=[
        "Low (<0.10)",
        "Medium (0.10-0.50)",
        "High (>0.50)"
    ],
    include_lowest=True
)

damage_group_summary = (
    validation_results
    .groupby("damage_group", observed=False)
    .agg(
        sample_count=("actual", "count"),
        mean_actual_damage=("actual", "mean"),
        mean_percentage_error=("percentage_error", "mean"),
        median_percentage_error=("percentage_error", "median"),
        max_percentage_error=("percentage_error", "max")
    )
    .reset_index()
)

print("\n" + "=" * 60)
print("VALIDATION ERROR BY DAMAGE RANGE")
print("=" * 60)
print(damage_group_summary.to_string(index=False))


validation_results = validation_results.sort_values(
    "percentage_error",
    ascending=False
)

validation_results.to_csv(
    "shm_validation_results.csv",
    index=False
)

print(
    "\nSaved validation results to "
    "shm_validation_results.csv"
)



# -----------------------------
# Tune Log Extra Trees
# -----------------------------
print("\n" + "=" * 60)
print("TUNING LOG EXTRA TREES")
print("=" * 60)

tuning_model = TransformedTargetRegressor(
    regressor=ExtraTreesRegressor(
        random_state=42
    ),
    func=np.log,
    inverse_func=np.exp
)

param_distributions = {
    "regressor__n_estimators": [300, 500, 800, 1000],
    "regressor__max_depth": [None, 4, 6, 8, 10, 15],
    "regressor__min_samples_split": [2, 3, 4, 5, 8],
    "regressor__min_samples_leaf": [1, 2, 3, 4],
    "regressor__max_features": [1.0, 0.8, 0.6, "sqrt"]
}

random_search = RandomizedSearchCV(
    estimator=tuning_model,
    param_distributions=param_distributions,
    n_iter=30,
    scoring=mape_scorer,
    cv=cv,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)

tuned_model = random_search.best_estimator_
tuned_mape = -random_search.best_score_

print("\nBest parameters:")
print(random_search.best_params_)

print(f"\nTuned CV MAPE: {tuned_mape:.4f}")
print(
    f"Approx. tuned score: "
    f"{max(0, 1 - tuned_mape):.4f}"
)

# -----------------------------
# Repeated cross-validation
# -----------------------------
print("\n" + "=" * 60)
print("REPEATED CROSS-VALIDATION")
print("=" * 60)

repeated_cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

repeated_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=repeated_cv,
    scoring=mape_scorer
)

repeated_mape = -repeated_scores

print(f"Mean repeated CV MAPE: {repeated_mape.mean():.4f}")
print(f"Std repeated CV MAPE:  {repeated_mape.std():.4f}")
print(f"Best fold MAPE:         {repeated_mape.min():.4f}")
print(f"Worst fold MAPE:        {repeated_mape.max():.4f}")

print(
    f"Approx. repeated CV score: "
    f"{max(0, 1 - repeated_mape.mean()):.4f}"
)

# -----------------------------
# Train + save best standard model
# -----------------------------
print("\n" + "=" * 50)
print("FINAL MODEL")
print("=" * 50)

best_model.fit(
    X,
    y
)

joblib.dump(
    best_model,
    "shm_model/final_model.pkl"
)

print(
    f"Saved final model: "
    f"{best_model_name}"
)