"""Module that defines the bernoulli classifier null model"""

import pandas as pd
from scipy.stats import bernoulli

from .classifier import Classifier


class BernoulliClassifier(Classifier):
    """Null model that predicts class using bernoulli distribution"""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the classifier to the data"""
        self.p = y.mean()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict the target variable"""
        values = bernoulli.rvs(self.p, size=X.shape[0], random_state=self.random_state)
        s = pd.Series(values, index=X.index)
        return s
