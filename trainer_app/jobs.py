from trainer_app import create_app
from flask import abort
from trainer_app.trainer import LRPipeline, NNPipeline, TreePipeline

from rq import get_current_job
from trainer_app.models import Task
from trainer_app import db


app = create_app()
app.app_context().push()


def train_model(model_type: str = 'lr', **model_params) -> str:
    try:
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
            abort(400, "{} model type is not possible. Possible parameters are: 'lr', 'nn', 'hdbr'".format(model_type))
        model_id = trainer.run()
    finally:
        job = get_current_job()
        if job:
            task = Task.query.filter(Task.job_id == job.get_id()).first()
            task.update_task_progress()
            if (job.meta['progress'] == 100):
                task.model_id = model_id
                db.session.commit()
    return model_id
