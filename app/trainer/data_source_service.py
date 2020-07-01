from google.cloud import bigquery
from google.oauth2 import service_account
from flask import current_app
from .main_query import query_string


class DataSource:
    def __init__(self, key_path=None):
        credentials = service_account.Credentials.from_service_account_file(
            (current_app.config["GOOGLE_APPLICATION_CREDENTIALS"] or key_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform",
                    'https://www.googleapis.com/auth/drive'],
        )

        self.client = bigquery.Client(
            credentials=credentials,
            project=credentials.project_id,
        )

    def main_query(self):
        return self.query(query_string)

    def query(self, select):
        return self.client.query(select).to_dataframe()


class ModelSource(DataSource):
    def push_model(
            self,
            model_id,
            model_type=None,
            model_params=None,
            model=None,
            transformer=None,
            metrics=None
    ):
        model_artifacts = {
            'model_id': model_id,
            'model_type': 'null' if model_type is None else str(model_type),
            'model_params': 'null' if model_params is None else str(model_params),
            'model': 'null' if model is None else model,
            'transformer': 'null' if transformer is None else transformer,
            'metrics': 'null'
        }
        insert = """
        insert models.Models (model_id, model_type, model_params, model, transformer, metrics, datetime)
        values('{model_id}', '{model_type}', '{model_params}', {model}, {transformer}, {metrics}, current_datetime())
        """.format(**model_artifacts)
        job = self.client.query(insert)
        result = job.result()
        return result


    def get_model(self, model_id):
        raise NotImplementedError
