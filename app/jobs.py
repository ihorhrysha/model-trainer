from app import create_app
from flask import abort
from app.trainer.trainer import LRTrainer, NNTrainer, TreeTrainer

app = create_app()
app.app_context().push()


def train_model(model_type: str = 'lr', **model_params) -> str:
    if model_type == "lr":
        trainer = LRTrainer(model_type)
    elif model_type == "nn":
        trainer = NNTrainer(model_type,
                            **model_params.get('nn_settings', {}))
    elif model_type == "hgbr":
        trainer = TreeTrainer(model_type,
                              **model_params.get('hgbr_settings', {}))
    else:
        trainer = None
        abort(400)
    model_id = trainer.run()
    return model_id