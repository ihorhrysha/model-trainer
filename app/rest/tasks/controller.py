from flask import request
from flask_restplus import Resource, Namespace
from app.rest.models.service import create_model
from .dto import TrainDto
from app.rest.models.dto import ModelDto
from .service import create_task, get_all_tasks, get_task, delete_task

api = TrainDto.api
train_dto = TrainDto.train_item
task_dto = TrainDto.task_item
model_dto = ModelDto.model_item

@api.route('/model')
class TaskModel(Resource):

    @api.expect(train_dto)
    @api.response(201, 'Model training successfully started.')
    def post(self):
        """
        Start task for model training
        """

        model_params = request.json
        job_id = create_task("train_model",
                    name = "Model training",
                    info = "{0} model training with params: {1}".format(model_params.get("model_type"), model_params),
                    **model_params)
        return job_id, 201

@api.route('/')
class TaskCollection(Resource):
    @api.marshal_list_with(task_dto)
    def get(self):
        """
        Returns all tasks
        """
        return get_all_tasks()


@api.route('/<int:id>')
@api.response(404, 'Task not found.')
class TrainModel(Resource):

    @api.marshal_with(task_dto)
    def get(self, id):
        """
        Returns a task
        """
        return get_task(id)

    @api.response(204, 'Model successfully deleted.')
    def delete(self, id):
        """
        Deletes a task.
        """
        delete_task(id)
        return None, 204

@api.route('/progress/<int:id>')
@api.response(404, 'Task not found.')
class TrainModel(Resource):

    def get(self, id):
        """
        Returns a progress of the task
        """
        return get_task(id).get_task_progress()
