import os
import logging.config

from flask import Flask, Blueprint
from flask_restplus import Api

import settings
from rest import api

from database import init_database

app = Flask(__name__)

# logging
logging_conf_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__), 'logging.conf'))
logging.config.fileConfig(logging_conf_path)
log = logging.getLogger(__name__)


def configure_app(flask_app):
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = settings.SQLALCHEMY_DATABASE_URI
    flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = settings.SQLALCHEMY_TRACK_MODIFICATIONS
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
    flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP


def initialize_app(flask_app):

    # Config
    configure_app(flask_app)

    # Separate API module
    blueprint = Blueprint('api', __name__, url_prefix='/api')
    api.init_app(blueprint)

    # API
    flask_app.register_blueprint(blueprint)

    # DB
    init_database(flask_app)


initialize_app(app)

if __name__ == "__main__":
    app.run(debug=settings.FLASK_DEBUG)
