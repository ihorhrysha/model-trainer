from datetime import datetime

from app.database import db


class Task(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))  # TODO Enums
    status = db.Column(db.String(20))
    info = db.Column(db.Text)
    begin_date = db.Column(db.DateTime)
    finish_date = db.Column(db.DateTime)

    model = db.relationship(
        'Model', backref=db.backref('model', lazy='dynamic'))

    model_id = db.Column(db.Integer, db.ForeignKey('model.id'))

    def __init__(self, name, info, status, model, begin_date=None, finish_date=None):
        self.name = name
        self.info = info
        self.status = status
        self
        if begin_date is None:
            begin_date = datetime.utcnow()
        self.begin_date = begin_date
        self.finish_date = finish_date
        self.model = model

    def __repr__(self):
        return '<Task %r>' % self.title
