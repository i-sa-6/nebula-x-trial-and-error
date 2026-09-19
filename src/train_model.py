"""
Model Training & Stratified 5-Fold Cross-Validation Engine.
Evaluates single models and champion ensemble (HistGradientBoosting + Balanced Logistic Regression).
Optimizes class decision thresholds to maximize competition Macro F1.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

from src.config import DATA_DIR, MODELS_DIR, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from src.models import EnsembleClassifier

FEATURES_PATH = os.path.join(DATA_DIR, "train_features.csv")



def optimize_thresholds(y_true, oof_probs):
    def objective(weights):
        weighted_probs = oof_probs * weights
        preds = np.argmax(weighted_probs, axis=1)
        return -f1_score(y_true, preds, average='macro', zero_division=0)
        
    res = minimize(objective, [1.0, 1.2, 1.2], method='Nelder-Mead')
    best_weights = res.x / res.x[0]
    return best_weights

def evaluate_predictions(y_true, y_pred, title=""):
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    print(f"\n{'='*60}")
    print(f" {title} - Performance Report")
    print(f"{'='*60}")
    print(f"★ MACRO F1 SCORE: {macro_f1:.4f} (Competition Metric)\n")
    
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 60)
    for idx, cname in enumerate(CLASSES):
        support = np.sum(y_true == idx)
        print(f"{cname:<12} | {prec[idx]:<10.4f} | {rec[idx]:<10.4f} | {per_class_f1[idx]:<10.4f} | {support:<8}")
    print("-" * 60)
    
    print("\nConfusion Matrix (Rows: True, Cols: Predicted):")
    print(f"             Pred_Normal  Pred_SideI  Pred_SideII")
    for i, cname in enumerate(CLASSES):
        print(f"True_{cname:<7}:    {cm[i, 0]:<10} {cm[i, 1]:<10} {cm[i, 2]:<10}")
    print(f"{'='*60}\n")
    
    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm
    }

def train_and_cross_validate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH)
    print(f"[*] Loaded dataset with {len(df)} samples and {df.shape[1]} columns.")
    
    feature_cols = [c for c in df.columns if c not in ['filename', 'label']]
    X = df[feature_cols].values
    y = df['label'].map(CLASS_TO_IDX).values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate individual models and champion ensemble
    candidates = {
        "Logistic Regression (Balanced)": LogisticRegression(class_weight='balanced', max_iter=2000, C=0.5, random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=250, class_weight='balanced_subsample', max_depth=12, random_state=42, n_jobs=-1),
        "HistGradientBoosting (Balanced)": HistGradientBoostingClassifier(class_weight='balanced', max_iter=200, learning_rate=0.08, max_depth=6, random_state=42),
        "Champion Ensemble (50% HGB + 50% LR)": EnsembleClassifier(w_hgb=0.5, w_lr=0.5)
    }
    
    results = {}
    
    for model_name, model in candidates.items():
        print(f"\n>>> Running 5-Fold Stratified CV: {model_name}...")
        oof_probs = np.zeros((len(X), 3))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            scaler = RobustScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            model.fit(X_tr_scaled, y_tr)
            oof_probs[val_idx] = model.predict_proba(X_val_scaled)
            
        opt_weights = optimize_thresholds(y, oof_probs)
        opt_preds = np.argmax(oof_probs * opt_weights, axis=1)
        rep = evaluate_predictions(y, opt_preds, title=model_name)
        
        results[model_name] = {
            "macro_f1": rep["macro_f1"],
            "per_class_f1": rep["per_class_f1"],
            "precision": rep["precision"],
            "recall": rep["recall"],
            "confusion_matrix": rep["confusion_matrix"],
            "opt_weights": opt_weights,
            "oof_probs": oof_probs
        }
        
    champion_name = "Champion Ensemble (50% HGB + 50% LR)"
    champion_info = results[champion_name]
    
    print("\n" + "="*65)
    print(f"★ SELECTED CHAMPION: {champion_name}")
    print(f"★ 5-FOLD STRATIFIED CV MACRO F1: {champion_info['macro_f1']:.4f}")
    print(f"★ OPTIMAL WEIGHT VECTOR: {np.round(champion_info['opt_weights'], 3)}")
    print("="*65)
    
    # Train champion on all 272 training samples
    full_scaler = RobustScaler()
    X_full_scaled = full_scaler.fit_transform(X)
    
    champion_model = EnsembleClassifier(
        w_hgb=0.5, w_lr=0.5, class_weights=champion_info['opt_weights']
    )
    champion_model.fit(X_full_scaled, y)
    
    bundle = {
        "model": champion_model,
        "scaler": full_scaler,
        "feature_cols": feature_cols,
        "class_weights": champion_info['opt_weights'],
        "classes": CLASSES,
        "model_name": champion_name,
        "cv_macro_f1": champion_info["macro_f1"],
        "cv_results": results
    }
    
    save_path = os.path.join(MODELS_DIR, "rail_corrugation_champion.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[✓] Saved complete champion model artifact to {save_path}")
    
    return results, bundle

if __name__ == "__main__":
    train_and_cross_validate()
