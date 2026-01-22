"""
Baseline classifier utilities.

This module provides backward-compatible access to baseline classifiers from
trait_prediction. The actual implementations are in trait_prediction.classifiers.
"""

# Re-export classifiers from trait_prediction for backward compatibility
from trait_prediction.classifiers import (
    BernoulliClassifier,
    Classifier,
    IdentityClassifier,
    NearestNeighborClassifier,
)

__all__ = [
    "Classifier",
    "BernoulliClassifier",
    "IdentityClassifier",
    "NearestNeighborClassifier",
]
