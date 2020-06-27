from google.cloud import bigquery
from google.oauth2 import service_account
from app.config import GOOGLE_APPLICATION_CREDENTIALS


class DataSource():
    def __init__(self, key_path):

        credentials = service_account.Credentials.from_service_account_file(
            (GOOGLE_APPLICATION_CREDENTIALS or key_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        self.client = bigquery.Client(
            credentials=credentials,
            project=credentials.project_id,
        )

    def query(self, select):
        return self.client.query(select).to_dataframe()
