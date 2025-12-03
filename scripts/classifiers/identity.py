"""Module that defines the identity classifier null model"""

import pandas as pd

from .classifier import Classifier


class IdentityClassifier(Classifier):
    """Null model that predicts the most frequent class in the training data"""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the classifier to the data"""
        value_counts = y.value_counts()
        max_class = value_counts.idxmax()
        self.most_frequent_class = max_class

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the target variable"""
        s = pd.Series([self.most_frequent_class] * X.shape[0], index=X.index)
        return s
