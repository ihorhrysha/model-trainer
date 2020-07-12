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

    return rq_job.get_id()


def get_task(job_id):
    task = Task.query.filter(Task.job_id == job_id).first()
    if task is None:
        abort(404, "Task {} has not been found".format(job_id))

    task.update_task_progress()
    return task


def delete_task(job_id):
    task = Task.query.filter(Task.job_id == job_id).first()

    # Redis job
    if task is None:
        abort(404, "Task {} has not been found".format(job_id))

    r = redis.StrictRedis()
    response = r.delete("rq:job:" + task.job_id)

    # DB Task
    db.session.delete(task)
    db.session.commit()

    return job_id


def get_all_tasks():
    for task in Task.query.all():
        task.update_task_progress()
    return Task.query.all()
