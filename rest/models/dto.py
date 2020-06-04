from flask_restplus import Namespace, fields


class ModelDto():
    api = Namespace('models',
                    description='Operations related to models mgmt')
    model_item = api.model('Model', {
        'id': fields.Integer(readOnly=True, description='The unique identifier of a model'),
        'name': fields.String(required=True, description='Model name')
    })
