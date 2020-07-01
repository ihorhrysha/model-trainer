from flask import request
from flask_restplus import Resource, Namespace
from app.rest.models.service import create_model
from .dto import TrainDto
from app.rest.models.dto import ModelDto
from .service import train_model

api = TrainDto.api
train_dto = TrainDto.train_item
model_dto = ModelDto.model_item

@api.route('/run')
class TrainModel(Resource):

    # @api.response(201, 'Model successfully trained.')
    @api.expect(train_dto)
    # @api.marshal_with(model_dto)
    def post(self):
        """
        Start training
        """

        model_params = request.json
        model_id = train_model(**model_params)
        return create_model({"name": model_id})
