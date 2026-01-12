"""
Data splitting utilities.

This module provides backward-compatible access to data splitters from
trait_prediction. The actual implementations are in trait_prediction.pipeline.splitters.
"""

# Re-export splitters from trait_prediction for backward compatibility
from trait_prediction.pipeline.splitters import (
    DataSplitter,
    InCladeSplitter,
    LargeTreeTraverseOOCSplitter,
    OutOfCladeSplitter,
    RandomSplitter,
)

__all__ = [
    "DataSplitter",
    "RandomSplitter",
    "LargeTreeTraverseOOCSplitter",
    "OutOfCladeSplitter",
    "InCladeSplitter",
]
