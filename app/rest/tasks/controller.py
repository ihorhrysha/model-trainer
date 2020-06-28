from flask import request
from flask_restplus import Resource, Namespace

from app.trainer.data_source_service import DataSource
from app.trainer.data_preprocessor import DataPreprocessor


api = Namespace('tasks', description='Operations related to models mgmt')


@api.route('/train')
class TrainModel(Resource):

    def get(self):
        """
        Start default training(for testing purposes)
        """

        gbq_ds = DataSource("bigquery.cred.json")

        df = gbq_ds.main_query()

        pd = DataPreprocessor()

        df = pd.preprocess(df)

        return "am trining" + str(df.shape)
