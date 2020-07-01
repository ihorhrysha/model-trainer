import abc
import logging
import pickle as pkl
from datetime import datetime
from typing import Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

from app.trainer.data_preprocessor import DataPreprocessor
from app.trainer.data_source_service import DataSource, ModelSource
from app.trainer.encoding_params import default_encoding_params
from app.trainer.transformer import Transformer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pkl.dump(obj, f)


def unpickle(filepath):
    with open(filepath, 'rb') as f:
        obj = pkl.load(f)
    return obj


def order_price_feature(df1):
    """ Create TotalOrderPrice feature (total BasePrice of items in order)
    """
    df = df1.copy()
    # df = df[['OrderId', 'BasePrice', 'OrderQty']].copy()
    df['TotalOrderProductPrice'] = df['BasePrice'] * df['OrderQty']

    right_df = df[['OrderId', 'TotalOrderProductPrice']]. \
        groupby('OrderId'). \
        sum(). \
        reset_index(). \
        rename(columns={'TotalOrderProductPrice': 'TotalOrderPrice'})

    if 'TotalOrderPrice' in df.columns:
        df = df.drop(columns='TotalOrderPrice')
    return df.merge(
        right_df,
        how='left',
        on='OrderId'
    )


def order_revenue_feature(df1):
    """ Create TotalOrderRevenue feature (total OrderPrice of items in order)
    """
    df = df1.copy()
    # df = df[['OrderId', 'BasePrice', 'OrderQty']].copy()
    df['TotalOrderProductRevenue'] = df['OrderPrice'] * df['OrderQty']

    right_df = df[['OrderId', 'TotalOrderProductRevenue']]. \
        groupby('OrderId'). \
        sum(). \
        reset_index(). \
        rename(columns={'TotalOrderProductRevenue': 'TotalOrderRevenue'})

    if 'TotalOrderRevenue' in df.columns:
        df = df.drop(columns='TotalOrderRevenue')
    return df.merge(
        right_df,
        how='left',
        on='OrderId'
    )


def order_cost_feature(df1):
    """ Create TotalOrderCost feature (total BasePrice of items in order)
    """
    df = df1.copy()
    # df = df[['OrderId', 'BasePrice', 'OrderQty']].copy()
    df['TotalOrderProductCost'] = df['CostPerItem'] * df['OrderQty']

    right_df = df[['OrderId', 'TotalOrderProductCost']]. \
        groupby('OrderId'). \
        sum(). \
        reset_index(). \
        rename(columns={'TotalOrderProductCost': 'TotalOrderCost'})

    if 'TotalOrderCost' in df.columns:
        df = df.drop(columns='TotalOrderCost')
    return df.merge(
        right_df,
        how='left',
        on='OrderId'
    )


def discount_metric(y, y_pred, ds_test):
    '''
      Function to calculate metric to estimate the goodness of the discount policy

      ! Needs input in original scale !
    '''

    predicted_df = ds_test.copy()
    predicted_df["BaseDiscount"] = y_pred
    predicted_df["OrderPrice"] = predicted_df["BasePrice"] - predicted_df[
        "BaseDiscount"] * predicted_df["BasePrice"]

    cost_df = order_cost_feature(ds_test)
    predicted_revenue_df = order_revenue_feature(predicted_df)
    revenue_df = order_revenue_feature(ds_test)

    predicted_revenue = predicted_revenue_df.TotalOrderProductRevenue.sum()
    real_revenue = revenue_df.TotalOrderProductRevenue.sum()

    # special metric
    predicted_income_product = predicted_revenue_df.TotalOrderProductRevenue - cost_df.TotalOrderProductCost
    mses = (y - y_pred) ** 2
    mses = mses.reset_index(drop=True)
    loss_orders_ind = \
    np.where(predicted_income_product.reset_index(drop=True) <= 0)[0]
    profit_orders_ind = \
    np.where(predicted_income_product.reset_index(drop=True) > 0)[0]

    mses1 = (mses[loss_orders_ind] * 10).to_list() + mses[
        profit_orders_ind].to_list()
    return np.mean(mses1), predicted_revenue, real_revenue


class LogModelWrapper:
    def __init__(
            self,
            model,
            encoder=preprocessing.StandardScaler,
            encoder_params={}
    ):
        self.model = model
        self.y_tr = encoder(**encoder_params)

    def fit(self, X, y, **kwargs):
        y = np.array(y).reshape((-1,1))
        y = np.log1p(y)
        return self.model.fit(X, self.y_tr.fit_transform(y), **kwargs)

    def predict(self, X, **kwargs):
        return np.expm1(self.y_tr.inverse_transform(
            self.model.predict(X, **kwargs)
        ))


class AbstractTrainer(abc.ABC):
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
        dataset = self.load_dataset()
        # with open('/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/df.pkl', 'rb') as f:
        #     dataset = pkl.load(f)
        logger.info('Dataset loaded.')

        logger.info('Preprocessing dataset...')
        dataset = self.preprocess_dataset(dataset)
        # with open('/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/df_preprocessed.pkl', 'rb') as f:
        #     dataset = pkl.load(f)
        logger.info('Dataset preprocessed.')

        logger.info('Splitting dataset...')
        ds_train, ds_test = self.split_dataset(dataset)
        logger.info('Dataset splitted.')
        del dataset

        logger.info('Transforming features...')
        ds_train_transformed, ds_test_transformed = \
            self.transform_features(ds_train, ds_test)
        # pickle(
        #     (ds_test, ds_train_transformed, ds_test_transformed),
        #     '/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/ds_train_test_transformed.pkl'
        # )
        logger.info('Features transformed.')
        del ds_train

        # self.save_model()
        # ds_test, ds_train_transformed, ds_test_transformed = unpickle('/home/andrii/.config/JetBrains/PyCharm2020.1/scratches/python-proj/ds_train_test_transformed.pkl')
        # self.load_model(None)

        logger.info('Training model...')
        self.train_model(ds_train_transformed)
        logger.info('Model trained.')
        del ds_train_transformed

        logger.info('Validating model...')
        self.test_model(ds_test, ds_test_transformed)
        logger.info('Model validated.')
        del ds_test, ds_test_transformed

        logger.info('Saving model...')
        # self.load_model(None)
        model_id = self.save_model()
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
        encoding_params = self.model_params.get('encoding_params',
                                                default_encoding_params)
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

    def test_model(self, ds_test: pd.DataFrame, ds_test_transformed: ModelInputType):
        y_test = ds_test_transformed.pop('UserDiscount')
        X_test = ds_test_transformed
        pred = self.model.predict(X_test)

        # TODO: discount_metric() function consumes too many memory
        # disc_metric, pred_revenue, true_revenue = \
        #     discount_metric(y_test, pred, ds_test)
        metrics = {
            'mse': mean_squared_error(y_test, pred),
            'mae': mean_absolute_error(y_test, pred),
            'discount_metric': None,  #disc_metric,
            'true_revenue': None,  #true_revenue,
            'pred_revenue': None,  #pred_revenue
        }
        logger.info("MSE:              {:.6f}".format(metrics['mse']))
        logger.info("MAE:              {:.6f}".format(metrics['mae']))
        # logger.info("Discount metric:  {:.6f}".format(metrics['discount_metric']))
        # logger.info("Revenue:")
        # logger.info("  True:           {:.6f}".format(true_revenue))
        # logger.info("  Predicted:      {:.6f}".format(pred_revenue))
        self.metrics = metrics

    def save_model(self) -> str:
        artifacts = self.serialized_artifacts
        # with open('model.pkl', 'wb') as f:
        #     pkl.dump(artifacts, f)
        model_source = ModelSource()
        model_source.push_model(**artifacts)
        return self.model_id

    def load_model(self, model_id: str):
        with open('model.pkl', 'rb') as f:
            artifacts = pkl.load(f)
        # model_source = ModelSource()
        # artifacts = model_source.get_model(model_id)
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


class LRTrainer(AbstractTrainer):
    ModelInputType = pd.DataFrame

    def train_model(self, ds_train: ModelInputType):
        model = LogModelWrapper(LinearRegression())
        y = ds_train.pop('UserDiscount')
        X = ds_train
        model.fit(X, y)
        self.model = model


class NNTrainer(AbstractTrainer):
    ModelInputType = Dict[str, np.ndarray]

    def transform_features(
            self,
            ds_train: pd.DataFrame,
            ds_test: pd.DataFrame,
            return_type: str = 'dict'
    ) -> (ModelInputType, Transformer):
        return super(NNTrainer, self).transform_features(
            ds_train,
            ds_test,
            return_type=return_type
        )

    def train_model(self, ds_train: ModelInputType):
        y = ds_train.pop('UserDiscount')
        X = ds_train

        tf.random.set_seed(42)
        inputs_num = [
            keras.Input(
                shape=(v.shape[1], ),
                dtype=tf.float32,
                name=k,
            ) for k, v in X.items()
        ]
        x = keras.layers.concatenate(inputs_num, axis=-1, name='input')
        activation = 'relu'  # worked better than 'elu'

        x = keras.layers.Dense(
            16,
            activation=activation,
            name='dense_1',
            kernel_initializer=keras.initializers.RandomUniform(seed=42)
            # works better than 'RandomNorm' and 'glorot'
        )(x)
        # x = tfkl.BatchNormalization()(x) # works worse
        x = keras.layers.Dense(
            8,
            activation=activation,
            name='dense_2',
            kernel_initializer=keras.initializers.RandomUniform(seed=42)
            # works better than 'RandomNorm' and 'glorot'
        )(x)
        # x = tfkl.BatchNormalization()(x) # works worse
        x_out = keras.layers.Dense(1, name='dense_out', use_bias=False)(x)

        # model.summary()
        model = keras.Model(inputs=inputs_num, outputs=x_out)

        model.compile(
            loss=keras.losses.Huber(delta=2.),
            # Works better than with delta=1. and better than MSE and MSLE
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.losses.MSE, keras.losses.MAE, keras.losses.MSLE]
        )

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                monitor='val_mean_squared_logarithmic_error',
                # works better than mse and huber
                filepath='model.hdf5',
                verbose=1,
                save_best_only=True
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_mean_squared_logarithmic_error',
                # works better than mse and huber
                patience=10,
                verbose=1
            )
        ]
        model = LogModelWrapper(model)
        history = model.fit(
            X, y,
            batch_size=512,
            epochs=100,
            verbose=1,
            validation_split=.3,
            callbacks=callbacks
        )
        model.model.load_weights('model.hdf5')
        self.model = model

    @property
    def serialized_artifacts(self):
        artifacts = self.artifacts
        artifacts.update({
            'model': self._serialize_keras_log_model(artifacts['model']),
            'transformer': pkl.dumps(artifacts['transformer']),
        })
        return artifacts

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
            model=self._deserialize_keras_log_model(model) if model else None,
            transformer=pkl.loads(transformer) if transformer else None,
            metrics=metrics
        )

    @staticmethod
    def _serialize_keras_log_model(log_model: LogModelWrapper) -> bytes:
        tmp_file_name = 'model.tmp.h5'
        log_model.model.save(tmp_file_name)
        with open(tmp_file_name, 'rb') as f:
            bytes = f.read()
        log_model.model = bytes
        return pkl.dumps(log_model)

    @staticmethod
    def _deserialize_keras_log_model(serialized: bytes) -> LogModelWrapper:
        log_model = pkl.loads(serialized)
        if not isinstance(log_model, LogModelWrapper):
            return log_model
        else:
            tmp_file_name = 'model.tmp.h5'
            with open(tmp_file_name, 'wb') as f:
                f.write(log_model.model)
            log_model.model = keras.models.load_model(tmp_file_name)
            return log_model


class TreeTrainer(AbstractTrainer):
    ModelInputType = pd.DataFrame

    def train_model(self, ds_train: ModelInputType):
        model = LogModelWrapper(HistGradientBoostingRegressor(random_state=42))
        y = ds_train.pop('UserDiscount')
        X = ds_train
        model.fit(X, y)
        self.model = model
