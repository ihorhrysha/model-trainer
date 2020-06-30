from flask_restplus import Namespace, fields


class TrainDto():
    api = Namespace('tasks', description='Operations related to models mgmt')

    hgbr_settings_item =  api.model('Task_settings_hgbr', {
            'loss': fields.String(readOnly=True, default = 'least_squares', example = 'least_squares', description='The loss function to use in the boosting process: least_squares, least_absolute_deviation, poisson'),
            'max_iter': fields.Integer(readOnly=True, default = 100, example = 100, description='The maximum number of iterations of the boosting process.'),
            'learning_rate': fields.Float(readOnly=True, default = 1, example = 1, description='The learning rate, also known as shrinkage.'),
            'max_leaf_nodes': fields.Integer(readOnly=True, default = 31, example = 31, description='The maximum number of leaves for each tree.'),
            'max_depth': fields.Integer(readOnly=True, default = 10000, example = 10000, description='The maximum depth of each tree.'),
            'min_samples_leaf': fields.Integer(readOnly=True, default = 20, example = 20, description='The minimum number of samples per leaf.'),
            'l2_regularization': fields.Float(readOnly=True, default = 0, example = 0, description='The L2 regularization parameter.'),
            'max_bins': fields.Integer(readOnly=True, default = 255, example = 255, description='The maximum number of bins to use for non-missing values.'),

    })
    nn_settings_item = api.model('Task_settings_nn', {
            "n_dense1": fields.Integer(readOnly=True, default = 16, example = 16, description='Number of neurons in first dense layer.'),
            "n_dense2": fields.Integer(readOnly=True, default = 8, example = 16, description='Number of neurons in second dense layer.')
    })

    train_item = api.model('Task', {
        'model_type': fields.String(required=True, example = 'lr', description='One from model types: lr, nn, hgbr.'),
        'nn_settings': fields.Nested(nn_settings_item),
        'hgbr_settings': fields.Nested(hgbr_settings_item),
    })
