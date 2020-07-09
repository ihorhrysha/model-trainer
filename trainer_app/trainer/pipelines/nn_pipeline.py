from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from tensorflow import keras

from trainer_app.trainer.model_wrapper import LogModelWrapper
from trainer_app.trainer.pipelines.abstract_pipeline import AbstractPipeline
from trainer_app.trainer.transformer import Transformer


class NNPipeline(AbstractPipeline):
    ModelInputType = Dict[str, np.ndarray]

    def transform_features(
            self,
            ds_train: pd.DataFrame,
            ds_test: pd.DataFrame,
            return_type: str = 'dict'
    ) -> (ModelInputType, Transformer):
        return super(NNPipeline, self).transform_features(
            ds_train,
            ds_test,
            return_type=return_type
        )

    def train_model(self, ds_train: ModelInputType):
        y = ds_train.pop('UserDiscount')
        X = ds_train

        n_hidden = (
            self.model_params.get('n_dense1', 16),
            self.model_params.get('n_dense2', 8),
        )

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

        for layer in range(2):
            x = keras.layers.Dense(
                n_hidden[layer],
                activation=activation,
                name=f'dense_{layer}',
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
            serialized_keras_model = f.read()
        serialized_y_tr = pkl.dumps(log_model.y_tr)
        return pkl.dumps({
            'model': serialized_keras_model,
            'y_tr': serialized_y_tr
        })

    @staticmethod
    def _deserialize_keras_log_model(serialized: bytes) -> LogModelWrapper:
        serialized_model = pkl.loads(serialized)
        if serialized_model:
            keras_model = serialized_model.get('model')
            tmp_file_name = 'model.tmp.h5'
            with open(tmp_file_name, 'wb') as f:
                f.write(keras_model)
            keras_model = keras.models.load_model(tmp_file_name)
            log_model = LogModelWrapper(keras_model)
            log_model.y_tr = serialized_model.get('y_tr')
            return log_model
        else:
            return serialized_model
