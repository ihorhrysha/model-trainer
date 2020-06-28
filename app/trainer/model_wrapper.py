from sklearn import preprocessing
import numpy as np


class ModelWrapper:
    """
    Performs transformation of `y` before `fit` and inverse transformation after `predict`.
    By default uses Sklearn's StandardScaler
    """

    def __init__(self, model, encoder=preprocessing.StandardScaler, encoder_params={}):
        self.model = model
        self.y_tr = encoder(**encoder_params)

    def fit(self, X, y, **kwargs):
        y = np.array(y).reshape((-1, 1))
        return self.model.fit(X, self.y_tr.fit_transform(y), **kwargs)

    def predict(self, X, **kwargs):
        return self.y_tr.inverse_transform(self.model.predict(X, **kwargs))
