from flask import Flask
from app.config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from redis import Redis
import rq

db = SQLAlchemy()
migrate = Migrate()

def create_app(config_class=Config):
    app = Flask(__name__)

    # Config
    app.config.from_object(config_class)

    # DB
    db.init_app(app)
    migrate.init_app(app, db=db)

    # Separate API module
    from app.rest import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    # Frontend
    from app.front import bp as front_bp
    app.register_blueprint(front_bp)
    
    # redis
    app.redis = Redis.from_url(app.config['REDIS_URL'])
    app.task_queue = rq.Queue('training-tasks', connection=app.redis)

    return app
