
import logging
import traceback

from flask import Blueprint
from flask_restplus import Api
from flask import current_app
from sqlalchemy.orm.exc import NoResultFound

from app.rest.models.controller import api as models_namespace
from app.rest.tasks.controller import api as tasks_namespace

log = logging.getLogger(__name__)

# building the API
api = Api(version='1.0', title='Trainer API',
          description='Model Trainer')

api.add_namespace(models_namespace)
api.add_namespace(tasks_namespace)

bp = Blueprint('api', __name__)

api.init_app(bp)


@api.errorhandler
def default_error_handler(e):
    message = 'An unhandled exception occurred.'
    log.exception(message)

    if not current_app.config('FLASK_DEBUG'):
        return {'message': message}, 500


@api.errorhandler(NoResultFound)
def database_not_found_error_handler(e):
    log.warning(traceback.format_exc())
    return {'message': 'A database result was required but none was found.'}, 404
