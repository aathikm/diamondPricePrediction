import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.append(src_dir)

from utils.utils import load_object
from logFile.loggingInfo import logging
from exception.exception import customException
from pipeline.training_pipeline import TrainingPipeline

class PredictionPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self, features):
        try:
            logging.info("training pipeline started in prediction pipeline")
            training_pipe = TrainingPipeline()
            training_pipe.start_training_pipeline()
            logging.info("training pipeline sucessfully ran")
            
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logging.info(f"data: {features}")
            model= load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            data_transformation = preprocessor.transform(features)
            logging.info(f"preprocessed value: {data_transformation}")
            prediction_value = model.predict(data_transformation)
            
            return prediction_value
            
        except Exception as e:
            logging.info("Error occured in prediction pipeline")
            raise customException(e, sys)

class GetCustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color:str,
                 clarity: str
                 ) -> None:
        
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
    
    def get_data(self):
        try:
            custom_data = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            
            df = pd.DataFrame(custom_data)
            logging.info(f"custom data converted as dataframe. And the data is, {df}")
            return(df)
        except Exception as e:
            logging.info("Error occured in prediction pipeline data preparation")
            raise customException(e, sys)

if __name__ == "__main__":
    obj1 = GetCustomData(carat=0.44, depth=59, table=60, x=4.9, y=4.94, z = 2.98,
                         clarity="VVS2", color="E", cut="Premium")
    
    data = obj1.get_data()
    
    obj2 = PredictionPipeline()
    res = obj2.predict(data)
    print(res)
    
    # 0.44,Premium,E,VVS2,60.0,59.0,4.99,4.95,2.98