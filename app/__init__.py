from flask import Flask, Blueprint
from flask_restplus import Api

from .config import configure_app
from app.rest import api
from .database import db
from flask_sqlalchemy import SQLAlchemy


def create_app():
    app = Flask(__name__)

    # Config
    configure_app(app)

    # Separate API module
    blueprint = Blueprint('api', __name__, url_prefix='/api')
    api.init_app(blueprint)
    app.register_blueprint(blueprint)

    # DB
    db.init_app(app)

    return app