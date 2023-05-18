import os
import sys 
import pandas as pd
import numpy as  np 
from src.logger import logging
from src.exception import CustomException
from  dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
# Pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

@dataclass
class DataTRansformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTRansformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initated")
            catigorical_features = ['bike_name', 'city']

            numerical_features = ['kms_driven', 'owner', 'age', 'power', 'brand']

            logging.info("Pipline Started")

            # numerical Pipline

            num_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
        ]
    )

            # catigorical pipline
            cato_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse=False,handle_unknown="ignore")),
                ("scaler",StandardScaler(with_mean=False))
        ]
    )


            preprocessor = ColumnTransformer([
            ("num_pipline",num_pipline,numerical_features),
            ("cato_pipline",cato_pipline,catigorical_features)
            ])

            return preprocessor

            logging.info("Pipline Complited")


        except Exception as e:
            logging.info("Error Occured in Data Transformation Class")
            raise CustomException(e,sys)


    def initatie_data_transformation(self,train_path,test_path):
        try:
            ## Read Train and Test Data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning And Test Data Complited")
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')


            logging.info("Obtaining Preprosser object")

            preprocessor_obj = self.get_data_transformation_object()


            target_columns_name = "Price"
            drop_columns = [target_columns_name]

            ## spliting dependent and indipend veriable
            input_features_train_data = train_data.drop(drop_columns,axis=1)
            targer_feature_train_data = train_data[target_columns_name]

            ## spliting dependent and indipend veriable
            input_features_test_data = test_data.drop(drop_columns,axis=1)
            targer_feature_test_data = test_data[target_columns_name]


            ## Apply Transformation object on train and test data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_data)

            logging.info("Apply Preprocessor Object on train and test Data")

            ## Convert in to array to become fast
            train_array = np.c_[input_feature_train_arr,np.array(targer_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(targer_feature_test_data)]


            ## Callling Save object to save preprocessor pkl file
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, 
            obj=preprocessor_obj
            )

            logging.info("Preprocessor Object File is Save")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.info("Error Occured in initate Data Transformation")
            raise CustomException(e,sys)

