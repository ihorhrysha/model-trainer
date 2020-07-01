from flask import request
from flask_restplus import Resource, Namespace

from app.trainer.data_source_service import DataSource
from .dto import TrainDto
from app.rest.models.dto import ModelDto
from app.trainer.data_preprocessor import DataPreprocessor
from .service import train_model


api = Namespace('tasks', description='Operations related to models mgmt')
api = TrainDto.api
train_dto = TrainDto.train_item
model_dto = ModelDto.model_item


@api.route('/run')
class TrainModel(Resource):
    def get(self, **model_params):
        """
        Start default training(for testing purposes)
        """

        return train_model(**model_params)