import os
from datetime import datetime

from flask import current_app
from google.cloud import bigquery
from google.oauth2 import service_account

from .utils import read_sql_query


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
        query_string = read_sql_query('app/trainer/main_query.sql')
        return self.query(query_string)

    def query(self, select):
        return self.client.query(select).to_dataframe()


class ModelSource(DataSource):
    def push_model(
            self,
            *,
            model_id,
            model_type=None,
            model_params=None,
            model=None,
            transformer=None,
            metrics=None
    ):
        table_id = 'models.Models'
        table = self.client.get_table(table_id)
        rows_to_insert = [(
            model_id,
            model_type,
            str(model_params),
            model,
            transformer,
            metrics,
            datetime.now()
        )]
        errors = self.client.insert_rows(table, rows_to_insert)
        if errors:
            raise RuntimeError(
                'An error occurred while saving model to BigQuery')

    def get_model(self, model_id):
        raise NotImplementedError
