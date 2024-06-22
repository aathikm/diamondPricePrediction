import pandas as pd
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.append(src_dir)

from logFile.loggingInfo import logging
from exception.exception import customException
from utils.utils import save_object, evaluate_model

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression

@dataclass
class ModelTrainingConfig:
    model_save_path = os.path.join("artifacts", "model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_path = ModelTrainingConfig()
        
    def initialize_model_training(self, train_arr, test_arr):
        try:
            logging.info("model trainer is started")
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            logging.info("X_train, X_test, y_train, y_test was splitted.")
                        
            models = {
                "lasso": Lasso(),
                "ridge": Ridge(),
                "linear": LinearRegression(),
                "randomForest": RandomForestRegressor(),
                "xgbr": XGBRegressor()
            }
            
            model_report:dict = evaluate_model(X_train, X_test, y_train, y_test, models=models)
            
            logging.info(f"model report --- {model_report}")
            
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            save_object(
                file_path = self.model_path.model_save_path,
                obj = best_model
            )
            
            logging.info("model stored as pickle file in artifact folder.")
            logging.info("model trainer was done...")

        
        except Exception as e:
            logging.warn("Error happened at model trainer block")
            raise customException(e, sys)
            
# if __name__ == "__main__":
#     obj = ModelTraining()
#     train_arr_path, test_arr_path = os.path.join("artifacts", "train_arr.pkl"), os.path.join("artifacts", "test_arr.pkl")
#     with open(train_arr_path, "rb") as f:
#         train_arr = pickle.load(f)
    
#     with open(test_arr_path, "rb") as f:
#         test_arr = pickle.load(f)
        
#     obj.initialize_model_training(train_arr, test_arr)
