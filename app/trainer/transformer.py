import os

import numpy as np
from joblib import delayed, Parallel
from sklearn import preprocessing, feature_extraction
from tqdm import tqdm
from typing import List, Dict
import pandas as pd


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


class Transformer:
    def __init__(self,
                 encoding_params: dict
                 ):
        self.features = {}
        for name, el in encoding_params.items():
            params = {'kwargs': el.get('kwargs', {})}
            if el['encoder'] is None:
                params['encoder'] = NoTransform
            elif el['encoder'] == 'time':
                params['encoder'] = TimeEncoder
            elif el['encoder'] == 'standard':
                params['encoder'] = StandardScaler
            elif el['encoder'] == 'ohe':
                params['encoder'] = OneHotEncoder
                if 'drop' in el.keys():
                    params['kwargs']['drop'] = el['drop']
            elif el['encoder'] == 'label':
                params['encoder'] = LabelEncoder
            elif el['encoder'] == 'hash':
                params['encoder'] = FeatureHasher
                if 'bins' in el.keys():
                    params['kwargs']['n_features'] = el['bins']
            else:
                params['encoder'] = el['encoder']

            params['encoder'] = params['encoder'](**params['kwargs'])
            params['encoder_type'] = el['encoder']
            params['column'] = el['column']
            if name in self.features.keys():
                raise ValueError(f'Duplicate feature name `{name}`')
            self.features[name] = params

    @staticmethod
    def _transform_column(df, params, mode: str):
        encoder = params['encoder']
        if params['column'] not in df.columns:
            raise ValueError(f"No `{params['column']}` column")
        x = df[params['column']].copy()
        if x.isna().any():
            print(f"{x.isna().sum()} of {x.shape[0]} are NaN in "
                  f"`{params['column']}` column")
            if params['encoder_type'] in ('hash', 'ohe'):
                print(f"Encoder type is {params['encoder_type']}, "
                      f"which is ABLE to transform")
                x = x.fillna('nan')
            else:
                raise ValueError(f"Encoder type is {params['encoder_type']}, "
                                 f"which is UNABLE to transform")
        if mode == 'fit':
            return encoder.fit(x)
        elif mode == 'transform':
            return encoder.transform(x)
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def fit(self, df):
        encoders = Parallel(n_jobs=1)(
            delayed(self._transform_column)(df, params, mode='fit')
            for params in tqdm(self.features.values(), desc='fit')
        )
        for name, new_encoder in zip(self.features.keys(), encoders):
            self.features[name]['encoder'] = new_encoder

    def transform(self, df, verbose=False, return_type='dict'):
        """
        return_type: str, one of {'dict', 'df'}
        """
        res = Parallel(n_jobs=os.cpu_count() - 1)(
            delayed(self._transform_column)(df, params, mode='transform')
            for params in tqdm(self.features.values(), desc='transform')
        )
        res = {name: value for name, value in zip(self.features.keys(), res)}

        if return_type == 'dict':
            return res
        elif return_type == 'df':
            return self._dict_to_df(res, df.index.values)
        else:
            raise ValueError(f"Unknown return type `{return_type}`. "
                             "Valid options are {'dict', 'df'}")

    @staticmethod
    def _dict_to_df(d, index):
        df = pd.DataFrame(index=index)
        for k, v in d.items():
            if v.shape[1] > 1:
                for j in range(v.shape[1]):
                    df[f'{k}_{j}'] = v[:, j]
            else:
                df[k] = v
        return df
