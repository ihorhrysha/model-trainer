from datetime import datetime

from app.database import db


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))

    # TODO binary resource

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<Model %r>' % self.name
