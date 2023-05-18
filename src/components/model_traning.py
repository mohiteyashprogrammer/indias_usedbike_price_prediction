import os
import sys 
import pandas as pd
import numpy as  np 
from src.logger import logging
from src.exception import CustomException
from  dataclasses import dataclass

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from src.utils import save_object
from src.utils import model_evalution


@dataclass
class ModelTraningconfig:
    train_model_file_path = os.path.join("artifcats","model.pkl")


class ModelTraning:
    def __init__(self):
        self.model_traner_config = ModelTraningconfig()


    def initaied_model_traning(self,train_array,test_array):
        try:
            logging.info("Spliting Dependent And Indipendent Feature")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                #train_array[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]],
                #train_array[:,6],
                #test_array[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]],
                #test_array[:,6]
            )


            models = {
            "LinearRegression":LinearRegression(),
            "Ridge":Ridge(),
            "Lesso":Lasso(),
            "Elastic":ElasticNet(),
            "LinearSVR":LinearSVR(),
            "DecisionTreeRegressor":DecisionTreeRegressor(),
            "RandomForestRegressor":RandomForestRegressor(),
            "XGBRegressor":XGBRegressor()
            }

            model_report:dict = model_evalution(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n************************************************************\n")
            logging.info(f"Model Report: {model_report}")

            ## To get the best Model Score From dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score : {best_model_score}")
            print("\n***************************************************\n")
            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 Score : {best_model_score}")


            save_object(file_path=self.model_traner_config.train_model_file_path,
            obj=best_model
            )
           
        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)