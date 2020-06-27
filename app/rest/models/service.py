from app.database.model import Model
from app.database import db


def create_model(data):
    name = data.get('name')
    model = Model(name)

    db.session.add(model)
    db.session.commit()


def delete_model(id):
    model = Model.query.filter(Model.id == id).one()
    db.session.delete(model)
    db.session.commit()


def get_all_models():
    return Model.query.all()


def get_model():
    return Model.query.filter(Model.id == id).one()
