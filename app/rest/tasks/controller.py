from flask import request
from flask_restplus import Resource, Namespace
from .dto import TrainDto
from app.rest.models.dto import ModelDto
# from app.trainer.data_source_service import DataSource
# from app.trainer.data_preprocessor import DataPreprocessor
from .service import train_model

api = TrainDto.api
train_dto = TrainDto.train_item
model_dto = ModelDto.model_item

@api.route('/train')
class TrainModel(Resource):

    #
    # @api.response(201, 'Model successfully trained.')
    @api.expect(train_dto)
    @api.marshal_with(model_dto)
    def post(self):
        """
        Start default training(for testing purposes)
        """

        #gbq_ds = DataSource("bigquery.cred.json")
        #df = gbq_ds.main_query()
        # pd = DataPreprocessor(df)
        # pd.preprocess()

        data = request.json
        return train_model(data)#"am trining" + str(df.shape)
