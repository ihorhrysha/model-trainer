from app.trainer import LRPipeline, NNPipeline, TreePipeline
from flask import abort


def train_model(model_type: str, **model_params) -> str:
    if model_type == "lr":
        trainer = LRPipeline(model_type)
    elif model_type == "nn":
        trainer = NNPipeline(model_type,
                             **model_params.get('nn_settings', {}))
    elif model_type == "hgbr":
        trainer = TreePipeline(model_type,
                               **model_params.get('hgbr_settings', {}))
    else:
        trainer = None
        abort(400)
    model_id = trainer.run()
    return model_id
