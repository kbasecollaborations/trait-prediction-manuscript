"""Re-export the baseline classifiers implemented in ``trait_prediction.classifiers``."""

from trait_prediction.classifiers import (
    BernoulliClassifier,
    Classifier,
    IdentityClassifier,
    NearestNeighborClassifier,
)

__all__ = [
    "BernoulliClassifier",
    "Classifier",
    "IdentityClassifier",
    "NearestNeighborClassifier",
]
