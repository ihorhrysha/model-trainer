from flask_restplus import Namespace, fields


class TrainDto():
    api = Namespace('tasks', description='Operations related to models mgmt')

    hgbr_settings_item = api.model('Task_settings_hgbr', {
        'loss': fields.String(
            readOnly=True,
            default='least_squares',
            example='least_squares',
            description='The loss function to use in the boosting process:'
            ' least_squares, least_absolute_deviation, poisson'
        ),
        'max_iter': fields.Integer(
            readOnly=True,
            default=100,
            example=100,
            description='The maximum number of iterations of '
            'the boosting process.'
        ),
        'learning_rate': fields.Float(
            readOnly=True,
            default=1,
            example=1,
            description='The learning rate, also known as shrinkage.'
        ),
        'max_leaf_nodes': fields.Integer(
            readOnly=True,
            default=31,
            example=31,
            description='The maximum number of leaves for each tree.'
        ),
        'max_depth': fields.Integer(
            readOnly=True,
            default=10000,
            example=10000,
            description='The maximum depth of each tree.'
        ),
        'min_samples_leaf': fields.Integer(
            readOnly=True,
            default=20,
            example=20,
            description='The minimum number of samples per leaf.'
        ),
        'l2_regularization': fields.Float(
            readOnly=True,
            default=0,
            example=0,
            description='The L2 regularization parameter.'
        ),
        'max_bins': fields.Integer(
            readOnly=True,
            default=255,
            example=255,
            description='The maximum number of bins to use for '
            'non-missing values.'
        ),
    })

    train_item = api.model('Task', {
        'model_type': fields.String(
            required=True,
            example='lr',
            description='One from model types: lr, nn, hgbr.'
        ),
        'hgbr_settings': fields.Nested(hgbr_settings_item),
    })

    task_item = api.model('Task_item', {
        'job_id': fields.String(
            description='The id of redis job'
        ),
        'name': fields.String(
            example='Model training task',
            description='The name of the task'
        ),
        'status': fields.String(
            example='Started',
            description='The status of the task'
        ),
        'info': fields.String(
            description='The information about the task'
        ),
        'begin_date': fields.DateTime(
            description='The date of starting the task'
        ),
        'finish_date': fields.DateTime(
            description='The date of starting the task'
        ),
        'model_id': fields.String(
            description='The id of created model if it was created'
        ),

    })
