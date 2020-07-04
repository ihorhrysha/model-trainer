from app.trainer.trainer import LRTrainer, NNTrainer, TreeTrainer
from flask import abort
from flask import current_app
from app.database.task import Task
from app.database import db
import redis

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

def create_task(func, name, info, **data):
    rq_job = current_app.task_queue.enqueue(func, **data)
    rq_job.meta['progress'] = 0
    task = Task( name=name, info=info,
                 status = rq_job.get_status(),
                 job_id = rq_job.get_id(),
                 begin_date=rq_job.started_at,
                 finish_date=rq_job.ended_at)
    db.session.add(task)
    db.session.commit()

#def get_tasks_in_progress():
#    return Task.query.filter_by(Task.status=="started").all()

def get_task(id):
    task = Task.query.filter(Task.id==id).one()
    task.update_task_progress()
    return task

def delete_task(id):
    task = Task.query.filter(Task.id == id).one()
    r = redis.StrictRedis()
    response = r.delete("rq:job:" + task.job_id)
    if response == 1:
        db.session.delete(task)
        db.session.commit()
    else:
        abort(500, "The redis job was not deleted")


def get_all_tasks():
    for task in Task.query.all():
        task.update_task_progress()
    return Task.query.all()
