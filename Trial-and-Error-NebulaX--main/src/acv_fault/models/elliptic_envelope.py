from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope, LedoitWolf

from .. import config


class EllipticEnvelopeModel:
    """Mahalanobis-distance outlier model. Falls back to a LedoitWolf shrinkage
    covariance (manual Mahalanobis distance) when the native FastMCD fit hits a
    singular/near-singular covariance matrix -- which real folds here do hit,
    since the one-hot "Load Halved=Normal" column is constant (zero variance)
    across all 5 labeled files.

    support_fraction=0.35 (sklearn's own default is None, ~0.5 of the
    training rows for the large n here) fits the covariance to a smaller,
    purer "core" of the training data: the same absolute deviation counts for
    more standard deviations against a tighter estimate of normal, which is
    the standard lever for making Mahalanobis distance more sensitive to
    small deviations. contamination=0.15 (sklearn default 0.1) only affects
    flag()'s count, never score()'s ranking. Both were chosen after a LOCO
    evaluation comparing this "high sensitivity" setting against sklearn's
    defaults confirmed the ranking on every labeled file was identical or
    better (case 05's previously tie-decided result resolved into a genuine
    one) -- see this project's git history / README for that comparison.
    """

    name = "elliptic_envelope"

    def __init__(
        self,
        random_state: int = config.RANDOM_SEED,
        fold_label: str = "?",
        support_fraction: float | None = 0.35,
        contamination: float = 0.15,
    ):
        self.random_state = random_state
        self.fold_label = fold_label
        self.support_fraction = support_fraction
        self.contamination = contamination
        self.fallback_used = False
        self.fallback_reason: str | None = None
        self.model: EllipticEnvelope | None = None
        self._mean = None
        self._cov_inv = None
        self._fallback_flag_threshold = None

    def fit(self, X_train: np.ndarray, case_ids: np.ndarray | None = None) -> "EllipticEnvelopeModel":
        try:
            model = EllipticEnvelope(
                random_state=self.random_state,
                support_fraction=self.support_fraction,
                contamination=self.contamination,
            )
            model.fit(X_train)
            self.model = model
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see class docstring
            self.fallback_used = True
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            print(
                f"[elliptic_envelope] fold={self.fold_label}: native fit failed "
                f"({self.fallback_reason}); falling back to LedoitWolf shrinkage covariance."
            )
            lw = LedoitWolf()
            lw.fit(X_train)
            self._mean = lw.location_
            self._cov_inv = np.linalg.pinv(lw.covariance_)
            train_scores = self._mahalanobis(X_train)
            self._fallback_flag_threshold = np.quantile(train_scores, 1 - self.contamination)
        return self

    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        diff = X - self._mean
        return np.einsum("ij,jk,ik->i", diff, self._cov_inv, diff)

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.fallback_used:
            return self.model.mahalanobis(X)  # squared Mahalanobis distance
        return self._mahalanobis(X)

    def flag(self, X: np.ndarray) -> np.ndarray:
        if not self.fallback_used:
            return self.model.predict(X) == -1
        return self._mahalanobis(X) > self._fallback_flag_threshold
