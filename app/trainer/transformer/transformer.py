import os

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from app.trainer.transformer.encoders import (
    FeatureHasher,
    LabelEncoder,
    NoTransform,
    OneHotEncoder,
    StandardScaler,
    TimeEncoder,
)


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

    def transform(self, df, return_type='dict'):
        """
        return_type: str, one of {'dict', 'df'}
        """
        # res = Parallel(n_jobs=os.cpu_count() - 1)(
        #     delayed(self._transform_column)(df, params, mode='transform')
        #     for params in tqdm(self.features.values(), desc='transform')
        # )
        res = tuple(
            self._transform_column(df, params, mode='transform')
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
