from flask import request
from flask_restplus import Resource

from .dto import ModelDto
from .service import get_all_models, create_model, get_model, delete_model

api = ModelDto.api
model_dto = ModelDto.model_item

# TODO logging for each ns log = logging.getLogger(__name__)


@api.route('/')
class ModelCollection(Resource):

    @api.marshal_list_with(model_dto)
    def get(self):
        """
        Returns list of models.
        """

        return get_all_models()

@api.route('/<int:id>')
@api.response(404, 'Model not found.')
class ModelItem(Resource):

    @api.marshal_with(model_dto)
    def get(self, id):
        """
        Returns a model
        """
        return get_model(id)

    @api.response(204, 'Model successfully deleted.')
    def delete(self, id):
        """
        Deletes a model.
        """
        delete_model(id)
        return None, 204
