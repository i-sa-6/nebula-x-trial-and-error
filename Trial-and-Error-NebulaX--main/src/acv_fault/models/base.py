"""Shared interface for the anomaly-detection model(s) -- Elliptic Envelope,
the sole settled model (see models/__init__.py for why).

`score(X)` returns one float per row on a consistent scale where HIGHER =
MORE ANOMALOUS: EllipticEnvelope's Mahalanobis distance is naturally higher =
more anomalous. `flag(X)` returns one bool per row -- a count-only anomaly
decision, independent of score()/the ranking, using sklearn's own
contamination-based `predict()`.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class AnomalyModel(Protocol):
    name: str

    def fit(self, X_train: np.ndarray, case_ids: np.ndarray | None = None) -> "AnomalyModel": ...

    def score(self, X: np.ndarray) -> np.ndarray: ...

    def flag(self, X: np.ndarray) -> np.ndarray: ...
