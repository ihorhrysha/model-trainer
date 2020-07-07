import rq
import redis
from flask import current_app
from datetime import datetime

from app import db

# TODO Do we need local nodel storage


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<Model %r>' % self.name


class Task(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))  # TODO Enums
    status = db.Column(db.String(20))
    info = db.Column(db.Text)
    begin_date = db.Column(db.DateTime)
    finish_date = db.Column(db.DateTime)
    job_id = db.Column(db.String(36))

    model = db.relationship(
        'Model', backref=db.backref('model', lazy='dynamic'))

    model_id = db.Column(db.Integer, db.ForeignKey('model.id'))

    def __init__(self, name, info, status, job_id, begin_date=None, finish_date=None):
        self.name = name
        self.info = info
        self.status = status

        if begin_date is None:
            begin_date = datetime.utcnow()
        self.begin_date = begin_date
        self.finish_date = finish_date
        self.job_id = job_id

    def __repr__(self):
        return '<Task %r>' % self.name

    def get_rq_job(self):
        try:
            rq_job = rq.job.Job.fetch(
                self.job_id, connection=current_app.redis)
        except (redis.exceptions.RedisError, rq.exceptions.NoSuchJobError):
            return None
        return rq_job

    def update_task_progress(self):
        job = self.get_rq_job()

        if (job is not None):

            if (job.result is not None) and (self.status == "finished") and (job.meta['progress'] == 100):
                model = Model(job.result.name)
                db.session.add(model)
                self.model_id = job.result

            self.status = job.get_status()
            self.finish_date = job.ended_at
            db.session.commit()

    def get_task_progress(self):
        self.update_task_progress()
        job = self.get_rq_job()
        return job.meta.get('progress', 0) if job is not None else 100
