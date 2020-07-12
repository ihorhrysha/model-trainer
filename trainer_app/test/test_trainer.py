import unittest
from trainer_app import create_app, db
from trainer_app.config import TestConfig
from trainer_app.models import Task, Model
import os
import time
import json

class TestTrainer(unittest.TestCase):

    # executed before each test
    def setUp(self):
        app = create_app(TestConfig)
        app.app_context().push()
        self.app = app
        self.client = app.test_client()
        db.drop_all()
        db.create_all()

    # executed after each test
    def tearDown(self):
        pass

    def test_empty_db(self):
        response = self.client.get('/api/models', follow_redirects = True)
        self.assertEqual(response.data, b'[]\n')

        response = self.client.get('/api/tasks', follow_redirects = True)
        self.assertEqual(response.data, b'[]\n')


    def test_task_creation(self):
        response = self.client.post('/api/tasks/model', data = json.dumps({"model_type": "elastic net",
                                                                  "hgbr_settings": {
                                                                    "loss": "least_squares",
                                                                    "max_iter": 100,
                                                                    "learning_rate": 1,
                                                                    "max_leaf_nodes": 31,
                                                                    "max_depth": 10000,
                                                                    "min_samples_leaf": 20,
                                                                    "l2_regularization": 0,
                                                                    "max_bins": 255
                                                                  }
                                                                }), follow_redirects = True,
                                                                    content_type='application/json')
        time.sleep(10)
        self.assertNotEqual(Task.query.all(), [])
        self.assertEqual(Task.query.first().id, response.json.get("id"))
        task = Task.query.first()
        task.update_task_progress()
        self.assertEqual(task.status, 'failed')


if __name__ == '__main__':
    unittest.main()
