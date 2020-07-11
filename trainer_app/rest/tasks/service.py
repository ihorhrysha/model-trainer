from flask import abort
from flask import current_app
from trainer_app.models import Task
from trainer_app import db
import redis


def create_task(func, name, info, **data):
    rq_job = current_app.task_queue.enqueue(
        "trainer_app.jobs."+func, job_timeout=-1, **data)
    rq_job.meta['progress'] = 0
    task = Task(name=name, info=info,
                status=rq_job.get_status(),
                job_id=rq_job.get_id(),
                begin_date=rq_job.started_at,
                finish_date=rq_job.ended_at)
    db.session.add(task)
    db.session.commit()

    return task


def get_task(id):
    task = Task.query.filter(Task.id == id).one()
    task.update_task_progress()
    return task


def delete_task(id):
    task = Task.query.filter(Task.id == id).one()
    r = redis.StrictRedis()
    if task.get_rq_job() != None:
        response = r.delete("rq:job:" + task.job_id)
        if response == 1:
            db.session.delete(task)
            db.session.commit()
        else:
            abort(500, "The redis job was not deleted")
    else:
        db.session.delete(task)
        db.session.commit()

def get_all_tasks():
    for task in Task.query.all():
        task.update_task_progress()
    return Task.query.all()
