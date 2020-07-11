import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from trainer_app.trainer.model_wrapper import LogModelWrapper
from trainer_app.trainer.pipelines.abstract_pipeline import AbstractPipeline
from trainer_app.trainer.utils import load_yaml


class TreePipeline(AbstractPipeline):
    ModelInputType = pd.DataFrame

    def train_model(self, ds_train: ModelInputType):
        base_model = HistGradientBoostingRegressor(
            random_state=42,
            **self.model_params
        )
        model = LogModelWrapper(base_model)
        y = ds_train.pop('UserDiscount')
        X = ds_train
        model.fit(X, y)
        self.model = model

    @property
    def encoding_params(self):
        return load_yaml(
            'trainer_app/trainer/transformer/encoding_params/'
            'tree_encoding_params.yml'
        )
