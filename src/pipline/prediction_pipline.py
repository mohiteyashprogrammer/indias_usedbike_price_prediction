import os
import sys 
import pandas as pd
import numpy as  np 
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object


class PredictPipline:
    def __init__(self):
        pass

    def Predict(self,features):
        try:
            ## This line Of code Work in Any system
            preproccesor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            preprocessor = load_object(preproccesor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
            bike_name:str,
            city:str,
            kms_driven:float,
            owner:int,
            age:float,
            power:float,
            brand:int):

        self.bike_name = bike_name
        self.city = city
        self.kms_driven = kms_driven
        self.owner = owner
        self.age  = age
        self.power = power
        self.brand = brand

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "bike_name":[self.bike_name],
                "city":[self.city],
                "kms_driven":[self.kms_driven],
                "owner":[self.owner],
                "age":[self.age],
                "power":[self.power],
                "brand":[self.brand],
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)
