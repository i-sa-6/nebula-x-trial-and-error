"""
Model definitions for Rail Corrugation Subsystem.
Defines EnsembleClassifier for consistent serialization and inference.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class EnsembleClassifier:
    """
    Weighted soft-voting ensemble of HistGradientBoosting and Balanced Logistic Regression.
    """
    def __init__(self, w_hgb=0.5, w_lr=0.5, class_weights=None):
        self.w_hgb = w_hgb
        self.w_lr = w_lr
        self.class_weights = np.array(class_weights if class_weights is not None else [1.0, 1.0, 1.0])
        self.hgb = HistGradientBoostingClassifier(
            class_weight='balanced', max_iter=200, learning_rate=0.08, max_depth=6, random_state=42
        )
        self.lr = LogisticRegression(
            class_weight='balanced', max_iter=2000, C=0.5, random_state=42
        )
        
    def fit(self, X, y):
        self.hgb.fit(X, y)
        self.lr.fit(X, y)
        return self
        
    def predict_proba(self, X):
        p_hgb = self.hgb.predict_proba(X)
        p_lr = self.lr.predict_proba(X)
        return self.w_hgb * p_hgb + self.w_lr * p_lr
        
    def predict(self, X):
        probs = self.predict_proba(X)
        weighted_probs = probs * self.class_weights
        return np.argmax(weighted_probs, axis=1)
