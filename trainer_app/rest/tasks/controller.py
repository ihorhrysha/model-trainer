from flask import request
from flask_restplus import Resource, Namespace
from .dto import TrainDto
from .service import create_task, get_all_tasks, get_task, delete_task

api = TrainDto.api
train_dto = TrainDto.train_item
task_dto = TrainDto.task_item


@api.route('/train')
class TaskTrain(Resource):

    @api.expect(train_dto)
    @api.marshal_with(task_dto)
    @api.response(201, 'Model training successfully started.')
    def post(self):
        """
        Start task for model training
        """

        model_params = request.json
        task = create_task("train_model",
                    name = "Model training",
                    info = "{0} model training with params: {1}".format(model_params.get("model_type"), model_params),
                    **model_params)
        return task, 201


@api.route('/')
class TaskCollection(Resource):
    @api.marshal_list_with(task_dto)
    def get(self):
        """
        Returns all tasks
        """
        return get_all_tasks()


@api.route('/<string:job_id>')
@api.response(404, 'Task not found.')
class TaskItem(Resource):

    @api.marshal_with(task_dto)
    def get(self, job_id):
        """
        Returns a task
        """
        return get_task(job_id)

    @api.response(204, 'Model successfully deleted.')
    def delete(self, job_id):
        """
        Deletes a task.
        """
        delete_task(job_id)
        return None, 204

@api.route('/info/<string:job_id>')
@api.response(404, 'Task not found.')
class TaskProgress(Resource):

    def get(self, job_id):
        """
        Returns a progress of the task
        """
        return get_task(job_id).get_task_progress()
