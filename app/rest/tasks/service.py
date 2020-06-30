from app.trainer.trainer import Trainer
from app.database.model import Model
from app.database import db
from flask import abort

def train_model(data):
    model_type = data.get('model_type')
    if model_type == "lr":
        trainer = Trainer(model_type, None)

    elif model_type == "nn":
        trainer = Trainer(model_type, data.get('nn_settings'))

    elif model_type == "hgbr":
        trainer = Trainer(model_type, data.get('hgbr_settings'))

    else:
        abort(400)

    model_name = trainer.train()
    return Model(model_name)#, model_path)
