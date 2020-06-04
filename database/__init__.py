
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_database(flask_app):
    from database.model import Model
    #from rest.tasks.router import tasks

    db.init_app(flask_app)
    
    with flask_app.app_context():
        db.drop_all()
        db.create_all()
