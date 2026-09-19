from .elliptic_envelope import EllipticEnvelopeModel

__all__ = [
    "EllipticEnvelopeModel",
]


def build_models(fold_label: str = "?"):
    """Fresh, unfit instance(s) of the settled model(s) (call once per CV fold).

    This project evaluated 5 unsupervised models (also Isolation Forest, PCA
    Reconstruction, Local Outlier Factor, and a small autoencoder) at both a
    baseline and a "high sensitivity" hyperparameter preset -- 10 (model,
    aggregation) combinations in a leave-one-case-out evaluation, then briefly
    kept Isolation Forest alongside Elliptic Envelope for paradigm diversity
    (they tied on accuracy: 4 of 5 labeled files correctly ranked). Elliptic
    Envelope alone is the final, settled choice: it matches Isolation
    Forest's accuracy while being the more informative of the two on the one
    file both miss (a narrow 2-way tie vs. Isolation Forest's uninformative
    complete 8-way tie), has zero tie-decided ranks anywhere in the labeled
    evaluation, and is simpler to maintain and explain as a single model. See
    this project's git history for the full 5-model, then 2-model,
    comparisons that led here.
    """
    return [
        EllipticEnvelopeModel(fold_label=fold_label),
    ]
