from app.trainer.trainer import LRTrainer, NNTrainer, TreeTrainer
from flask import abort

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
