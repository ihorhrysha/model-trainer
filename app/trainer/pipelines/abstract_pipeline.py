import abc
import logging
import pickle as pkl
from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
from rq import get_current_job
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.models import Task
from app.trainer.data_preprocessor import DataPreprocessor
from app.trainer.data_source_service import DataSource, ModelSource
from app.trainer.transformer import Transformer
from app.trainer.utils import load_yaml, unpickle

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def set_task_progress(progress):
    job = get_current_job()
    if job:
        job.meta['progress'] = progress
        job.save_meta()
        task = Task.query.filter(Task.job_id == job.get_id()).first()
        task.update_task_progress()


class AbstractPipeline(abc.ABC):
    ModelInputType = Union[pd.DataFrame, Dict[str, np.ndarray]]

    def __init__(self, model_type, **model_params):
        self._gbq_ds = DataSource('bigquery.cred.json')
        self.model_id = str(int(datetime.now().timestamp() * 1e6))
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.transformer = None
        self.metrics = None

    def run(self):
        logger.info('Loading dataset...')
        # dataset = self.load_dataset()
        # with open('/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/df.pkl', 'rb') as f:
        #     dataset = pkl.load(f)
        set_task_progress(10)
        logger.info('Dataset loaded.')

        logger.info('Preprocessing dataset...')
        # dataset = self.preprocess_dataset(dataset)
        with open('/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/df_preprocessed.pkl', 'rb') as f:
            dataset = pkl.load(f).sample(frac=.1, random_state=42)
        set_task_progress(30)
        logger.info('Dataset preprocessed.')

        logger.info('Splitting dataset...')
        ds_train, ds_test = self.split_dataset(dataset)
        set_task_progress(40)
        logger.info('Dataset splitted.')
        del dataset

        logger.info('Transforming features...')
        ds_train_transformed, ds_test_transformed = \
            self.transform_features(ds_train, ds_test)
        # pickle(
        #     (ds_test, ds_train_transformed, ds_test_transformed),
        #     '/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/ds_train_test_transformed.pkl'
        # )
        set_task_progress(60)
        logger.info('Features transformed.')
        del ds_train

        # self.save_model()
        # ds_test, ds_train_transformed, ds_test_transformed = unpickle('/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/ds_train_test_transformed.pkl')
        # self.load_model(None)

        logger.info('Training model...')
        self.train_model(ds_train_transformed)
        set_task_progress(80)
        logger.info('Model trained.')
        del ds_train_transformed

        logger.info('Validating model...')
        self.test_model(ds_test, ds_test_transformed)
        set_task_progress(90)
        logger.info('Model validated.')
        del ds_test, ds_test_transformed

        logger.info('Saving model...')
        # self.load_model(None)
        model_id = self.save_model()
        set_task_progress(100)
        logger.info(f'Model saved with model_id={model_id}.')
        return self.model_id

    def load_dataset(self) -> pd.DataFrame:
        dataset = self._gbq_ds.main_query()
        # with open('df.pkl', 'wb') as f:
        #     pkl.dump(dataset, f)
        return dataset

    @staticmethod
    def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
        dp = DataPreprocessor()
        dataset = dp.preprocess(dataset)
        # with open('df_preprocessed.pkl', 'wb') as f:
        #     pkl.dump(dataset, f)
        return dataset

    @staticmethod
    def split_dataset(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """ Splits dataset into `train` and `test`, taking for `test` 9th, 10th,
        11th months of 2019 and all the previous months for `train`.
        December 2020 is dropped.
        """
        train_ids = df.index[
            (df['OrderDate'].dt.month < 9) | (df['OrderDate'].dt.year < 2019)]
        test_ids = df.index[df['OrderDate'].dt.month.isin([9, 10, 11]) & df[
            'OrderDate'].dt.year.isin([2019])]
        return df.loc[train_ids], df.loc[test_ids]

    def transform_features(
            self,
            ds_train: pd.DataFrame,
            ds_test: pd.DataFrame,
            return_type: str = 'df'  # 'df' or 'dict'
    ) -> (ModelInputType, Transformer):
        encoding_params = self.encoding_params
        transformer = Transformer(encoding_params)
        transformer.fit(ds_train)
        ds_train_transformed = transformer.transform(ds_train,
                                                     return_type=return_type)
        ds_test_transformed = transformer.transform(ds_test,
                                                    return_type=return_type)
        self.transformer = transformer
        return ds_train_transformed, ds_test_transformed

    @abc.abstractmethod
    def train_model(self, ds_train: ModelInputType):
        """
        creates and fits `self.model`
        """

    def test_model(
            self,
            ds_test: pd.DataFrame,
            ds_test_transformed: ModelInputType
    ):
        y_test = ds_test_transformed.pop('UserDiscount')
        X_test = ds_test_transformed
        pred = self.model.predict(X_test)

        # TODO: discount_metric() function consumes too many memory
        # disc_metric, pred_revenue, true_revenue = \
        #     discount_metric(y_test, pred, ds_test)
        metrics = {
            'mse': mean_squared_error(y_test, pred),
            'mae': mean_absolute_error(y_test, pred),
            'discount_metric': None,  # disc_metric,
            'true_revenue': None,  # true_revenue,
            'pred_revenue': None,  # pred_revenue
        }
        logger.info("MSE:              {:.6f}".format(metrics['mse']))
        logger.info("MAE:              {:.6f}".format(metrics['mae']))
        # logger.info("Discount metric:  {:.6f}".format(
        #     metrics['discount_metric']))
        # logger.info("Revenue:")
        # logger.info("  True:           {:.6f}".format(true_revenue))
        # logger.info("  Predicted:      {:.6f}".format(pred_revenue))
        self.metrics = metrics

    def save_model(self) -> str:
        artifacts = self.serialized_artifacts
        # with open('/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/model.pkl', 'wb') as f:
        #     pkl.dump(artifacts, f)
        model_source = ModelSource()
        model_source.push_model(**artifacts)
        return self.model_id

    def load_model(self, model_id: str):
        # with open('/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/model.pkl', 'rb') as f:
        #     artifacts = pkl.load(f)
        model_source = ModelSource()
        artifacts = model_source.get_model(model_id)
        self.from_serialized_artifacts(**artifacts)

    @property
    def artifacts(self):
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'model': self.model,
            'transformer': self.transformer,
            'metrics': self.metrics
        }

    @property
    def serialized_artifacts(self):
        artifacts = self.artifacts
        artifacts.update({
            'model': pkl.dumps(artifacts['model']),
            'transformer': pkl.dumps(artifacts['transformer']),
        })
        return artifacts

    def from_artifacts(
            self,
            *,
            model_id=None,
            model_type=None,
            model_params=None,
            model=None,
            transformer=None,
            metrics=None
    ):
        if model_id:
            self.model_id = model_id
        if model_type:
            self.model_type = model_type
        self.model_params = model_params
        self.model = model
        self.transformer = transformer
        self.metrics = metrics

    def from_serialized_artifacts(
            self,
            *,
            model_id=None,
            model_type=None,
            model_params=None,
            model=None,
            transformer=None,
            metrics=None
    ):
        self.from_artifacts(
            model_id=model_id,
            model_type=model_type,
            model_params=model_params,
            model=pkl.loads(model) if model else None,
            transformer=pkl.loads(transformer) if transformer else None,
            metrics=metrics
        )

    @property
    def encoding_params(self):
        return load_yaml(
            'app/trainer/transformer/encoding_params/base_encoding_params.yml')
