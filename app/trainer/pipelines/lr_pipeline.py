import pandas as pd
from sklearn.linear_model import LinearRegression

from app.trainer.model_wrapper import LogModelWrapper
from app.trainer.pipelines.abstract_pipeline import AbstractPipeline


class LRPipeline(AbstractPipeline):
    ModelInputType = pd.DataFrame

    def train_model(self, ds_train: ModelInputType):
        model = LogModelWrapper(LinearRegression())
        y = ds_train.pop('UserDiscount')
        X = ds_train
        model.fit(X, y)
        self.model = model
