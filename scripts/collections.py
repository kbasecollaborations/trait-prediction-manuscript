"""
Utilities for managing data collections during machine learning
"""

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from trait_prediction.main import FeatureIndex, PhenotypeIndex


@dataclass
class DataIndex:
    """
    A unique identifier for data in a collection.

    Parameters
    ----------
    key : str
        A string identifier for the data
    pindex : PhenotypeIndex
        Index for phenotype data
    findex : FeatureIndex
        Index for feature data
    """

    key: str
    pindex: PhenotypeIndex
    findex: FeatureIndex

    def __hash__(self) -> int:
        _pindex_hash = f"{self.pindex.name}-{self.pindex.category}"
        _findex_hash = f"{self.findex.name}-{self.findex.ftype}-{self.findex.dtype}"
        _hash = f"{self.key}-{_pindex_hash}-{_findex_hash}"
        return hash(_hash)


@dataclass
class XyItem:
    """
    Container for paired feature (X) and target (y) data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data matrix
    y : pd.Series
        Target variable vector
    """

    X: pd.DataFrame
    y: pd.Series


class XyCollection(MutableMapping[DataIndex, XyItem]):
    """Mutable mapping of DataIndex to XyItem."""

    def __init__(self) -> None:
        self._items: dict[DataIndex, XyItem] = {}

    def __getitem__(self, key: DataIndex) -> XyItem:
        return self._items[key]

    def __setitem__(self, key: DataIndex, value: XyItem) -> None:
        self._items[key] = value

    def __delitem__(self, key: DataIndex) -> None:
        del self._items[key]

    def __iter__(self):
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def items(self):
        return list(self._items.items())

    def keys(self):
        return list(self._items.keys())

    def values(self):
        return list(self._items.values())

    def add(
        self,
        key: str,
        pindex: PhenotypeIndex,
        findex: FeatureIndex,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        index = DataIndex(key, pindex, findex)
        item = XyItem(X, y)
        self[index] = item

    def filter_keys(self, key: str | None = None) -> Iterable[DataIndex]:
        if key is None:
            return iter(self.keys())
        return (dindex for dindex in self.keys() if dindex.key == key)

    def filter_values(self, key: str | None = None) -> Iterable[XyItem]:
        if key is None:
            return iter(self.values())
        return (item[1] for item in self.items() if item[0].key == key)
