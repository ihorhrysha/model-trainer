from app.trainer.data_source_service import DataSource
import time

class Trainer():
    def __init__(self, model_type, model_params_dict):
        self.__gbq_ds = DataSource("bigquery.cred.json")
        self._model_type = model_type
        self._model_params_dict = model_params_dict

    def train(self):
        #df = self.__gbq_ds.main_query()
        ### TODO: smart preprocessing and training
        time.sleep(10)
        return "Super-model" #, "Super-model-path.json"

