"""Module that defines the nearest neighbor null model"""

from warnings import warn

import numpy as np
import pandas as pd
from ete3 import Tree

from .classifier import Classifier


class NearestNeighborClassifier(Classifier):
    """Null model that predicts class using the nearest neighbor in the tree"""

    def _calculate_distance_matrix(self, tree: Tree) -> pd.DataFrame:
        """Calculate the distance matrix from the tree"""
        leaves: list[str] = list(tree.get_leaf_names())
        distance_matrix = np.zeros((len(leaves), len(leaves)))
        for i, leaf1 in enumerate(leaves):
            for j, leaf2 in enumerate(leaves):
                if leaf1 == leaf2:
                    distance_matrix[i, j] = np.inf
                    continue
                distance = tree.get_distance(leaf1, leaf2)
                distance_matrix[i, j] = distance
        ind = pd.Index(leaves)
        return pd.DataFrame(distance_matrix, index=ind, columns=ind)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the classifier to the data"""
        tree = self.kwargs.get("tree", None)
        k = self.kwargs.get("k", 1)
        if tree is None:
            raise ValueError("Tree must be provided")
        self.tree = tree.copy()
        self.k = k
        # Map all dist of all genomes to training genomes
        distances = self.kwargs.get("distances", None)
        if distances is None:
            distances = self._calculate_distance_matrix(self.tree)
        cols = distances.index.intersection(y.index)
        distances_train = distances.loc[:, cols]
        self.distances = distances_train
        self.y = y.loc[cols]

    def predict(self, X: pd.DataFrame, round_to_int: bool = True) -> pd.Series:
        """Predict the target variable"""
        distances = self.distances
        values = []
        for test_ind in X.index:
            # FIXME: What should we return instead of np.nan
            if test_ind in distances.index:
                node_distances = self.distances.loc[test_ind, :]
                nearest = node_distances.nsmallest(self.k).index
                if round_to_int:
                    value = int(self.y.loc[nearest].mean().round())
                else:
                    value = float(self.y.loc[nearest].mean())
            else:
                value = np.nan
                warn("Test index not in training data")
            values.append(value)
        return pd.Series(values, index=X.index)
