import numpy as np
import pandas as pd
from sklearn import feature_extraction, preprocessing


class FeatureHasher(feature_extraction.FeatureHasher):
    def __init__(self, n_features=(2 ** 8)):
        super().__init__(
            n_features=n_features,
            input_type="string"
        )

    @staticmethod
    def _prepare_data(x: pd.Series):
        return x.astype('str').values.reshape((-1, 1))

    def fit(self, X=None, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params: dict):
        return super().fit_transform(self._prepare_data(X), y, **fit_params
                                     ).toarray()

    def transform(self, X):
        return super().transform(self._prepare_data(X)).toarray()


class StandardScaler(preprocessing.StandardScaler):
    @staticmethod
    def _prepare_data(x: pd.Series):
        return x.values.reshape((-1, 1))

    def fit(self, X, y=None):
        return super().fit(self._prepare_data(X), y)

    def fit_transform(self, X, y=None, **fit_params: dict):
        return super().fit_transform(self._prepare_data(X), y, **fit_params)

    def transform(self, X, copy=None):
        return super().transform(self._prepare_data(X), copy=copy)


class LabelEncoder(preprocessing.LabelEncoder):
    @staticmethod
    def _prepare_data(x: pd.Series):
        return x.values.flatten()

    def fit(self, X):
        return super().fit(self._prepare_data(X))

    def fit_transform(self, X):
        return super().fit_transform(self._prepare_data(X)).reshape((-1, 1))

    def transform(self, X):
        return super().transform(self._prepare_data(X)).reshape((-1, 1))


class OneHotEncoder(preprocessing.OneHotEncoder):
    def __init__(self,
                 categories='auto',
                 drop=None
                 ):
        super().__init__(
            categories=categories,
            drop=drop,
            sparse=False,
        )

    @staticmethod
    def _prepare_data(x: pd.Series):
        res = x.values.reshape((-1, 1))
        return res

    def fit(self, X, y=None):
        return super().fit(self._prepare_data(X), y)

    def fit_transform(self, X, y=None):
        return super().fit_transform(self._prepare_data(X), y)

    def transform(self, X):
        return super().transform(self._prepare_data(X))


class NoTransform:
    def __init__(self):
        pass

    @staticmethod
    def _prepare_data(x: pd.Series):
        res = x.values.reshape((-1, 1))
        return res

    def fit(self, X):
        return self

    def fit_transform(self, X):
        return self._prepare_data(X)

    def transform(self, X):
        return self._prepare_data(X)


class TimeEncoder:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def _prepare_data(self, x: pd.Series):
        return 2 * np.pi * (x.values.flatten() - self.min) / self.max

    def fit(self, X):
        return self

    def fit_transform(self, X):
        prepared = self._prepare_data(X)
        res = np.zeros((prepared.shape[0], 2))
        res[:, 0] = np.cos(prepared)
        res[:, 1] = np.sin(prepared)
        return res

    def transform(self, X):
        return self.fit_transform(X)
