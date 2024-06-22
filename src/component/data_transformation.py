import pandas as pd
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
import pickle
import sys

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

current_dirName = os.path.dirname(os.path.abspath(__file__))
sys_dirName = os.path.join(current_dirName, "..")
sys.path.append(sys_dirName)
from logFile.loggingInfo import logging
from exception.exception import customException
from utils.utils import save_object

@dataclass
class DataTransformationConfig:
    model_preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")
    train_arr_file = os.path.join("artifacts", "train_arr.pkl")
    test_arr_file = os.path.join("artifacts", "test_arr.pkl")

class DataTransformation:
    def __init__(self):
        self.modelPreprocessor = DataTransformationConfig
        
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
        
        except Exception as e:
            logging.info("get data transformation error")
            raise customException(e, sys)
    
    def process_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("model transformation -- read the data frame")
            logging.info(f"train df: {train_df.head().to_string()}")
            logging.info(f"test df: {test_df.head().to_string()}")
            
            preprocessor_obj = self.get_data_transformation()
            
            target_cols = "price"
            drop_columns = [target_cols, "Unnamed: 0"]
            
            input_features_train_df = train_df.drop(drop_columns, axis=1)
            target_features_train_df = train_df[target_cols]
            
            input_features_test_df = test_df.drop(drop_columns, axis=1)
            target_features_test_df = test_df[target_cols]
            
            input_features_train_preprocessed_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_preprocessed_arr = preprocessor_obj.fit_transform(input_features_test_df)
            
            train_input_arr = np.c_[input_features_train_preprocessed_arr, target_features_train_df]
            test_input_arr = np.c_[input_features_test_preprocessed_arr, target_features_test_df]
            
            logging.info(f"training preprocessed array, {train_input_arr}")
            logging.info("the train and test data preprocessed using our custom preprocess model")
            
            save_object(
                self.modelPreprocessor.model_preprocessor_file_path, 
                preprocessor_obj)
            logging.info("preprocessor model stored as an pickle file in artifact folder")
            
            # with open(self.modelPreprocessor.train_arr_file, "wb") as f:
            #     pickle.dump(train_input_arr, f)
                
            # with open(self.modelPreprocessor.test_arr_file, "wb") as f:
            #     pickle.dump(test_input_arr, f)
            
            return(train_input_arr, test_input_arr)
        
        except Exception as e:
            logging.info("get data transformation error")
            raise customException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    train_path, test_path = os.path.join("artifacts", "train.csv"), os.path.join("artifacts", "test.csv")
    obj.process_data_transformation(train_path=train_path, test_path=test_path)