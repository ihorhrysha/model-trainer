import numpy as np
from sklearn import preprocessing


class LogModelWrapper:
    """
    Performs transformation of `y` before `fit` and inverse transformation after `predict`.
    By default uses Sklearn's StandardScaler
    """

    def __init__(
            self,
            model,
            encoder=preprocessing.StandardScaler,
            encoder_params=None
    ):
        self.model = model
        if encoder_params is None:
            encoder_params = {}
        self.y_tr = encoder(**encoder_params)

    def fit(self, X, y, **kwargs):
        y = np.array(y).reshape((-1, 1))
        y = np.log1p(y)
        return self.model.fit(X, self.y_tr.fit_transform(y), **kwargs)

    def predict(self, X, **kwargs):
        return np.expm1(self.y_tr.inverse_transform(
            self.model.predict(X, **kwargs)
        ))
