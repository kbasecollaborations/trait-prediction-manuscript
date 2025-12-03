"""Module that defines the base classifier class"""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.metrics import get_scorer


class Classifier(ABC):
    """Base class for classifiers"""

    def __init__(
        self, random_state: int, categorical_feature_names: list[str], **kwargs
    ) -> None:
        self.random_state = random_state
        self.categorical_feature_names = categorical_feature_names
        self.kwargs = kwargs

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the classifier to the data"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the target variable"""
        pass

    def score(
        self, X: pd.DataFrame, y: pd.Series, scoring: list[str]
    ) -> dict[str, float]:
        """Return the mean accuracy on the given test data and labels"""
        score_dict = dict()
        for scorer_name in scoring:
            scorer = get_scorer(scorer_name)
            score = scorer(self, X, y)
            score_dict[scorer_name] = score
        return score_dict

    def get_params(self) -> dict:
        """Get the parameters of the classifier"""
        return self.kwargs

    def set_params(self, **params) -> None:
        """Set the parameters of the classifier"""
        self.kwargs.update(params)
