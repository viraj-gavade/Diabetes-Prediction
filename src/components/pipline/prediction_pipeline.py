from src.exception_handler import CustomException
from src.logger import logging
from src.utils import load_object
import numpy as np
import pandas as pd
import sys
import os


class PredictionPipeline:
    def __init__(self):
        logging.info("Initializing Prediction Pipeline")
        pass


    def predict_result(self,features):
        try:
            logging.info("Started prediction process")
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            model_path = os.path.join(base_dir,'artifacts','model.pkl')
            preprocessor_path = os.path.join(base_dir,'artifcats','processor.pkl')
            
            logging.info(f"Loading model from {model_path}")
            model = load_object(model_path)
            
            logging.info(f"Loading preprocessor from {preprocessor_path}")
            preprocessor = load_object(l=preprocessor_path) 

            logging.info("Transforming features using preprocessor")
            scaled_data = preprocessor.transform(features) 

            logging.info("Making prediction with model")
            prediction = model.predict(scaled_data)
            
            logging.info(f"Prediction completed successfully: {prediction}")
            return prediction

        except Exception as e:
            logging.error(f"Error occurred during prediction: {e}")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,Pregnancies:int,Glucose:int,BloodPressure:int,SkinThickness:int,Insulin:int,BMI:int,DiabetesPedigeeFunction:int,Age:int):
        logging.info("Initializing CustomData object with user inputs")
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigeeFunction = DiabetesPedigeeFunction
        self.Age = Age
        logging.info(f"CustomData initialized with parameters: Pregnancies={Pregnancies}, Glucose={Glucose}, BloodPressure={BloodPressure}, SkinThickness={SkinThickness}, Insulin={Insulin}, BMI={BMI}, DiabetesPedigeeFunction={DiabetesPedigeeFunction}, Age={Age}")


    def get_data_as_dataframe(self):
        try:    
            logging.info("Converting CustomData to DataFrame")
            custom_data_input_dict = {
                'Pregnancies':[self.Pregnancies],
                'Glucose':[self.Glucose],
                'BloodPressure':[self.BloodPressure],
                'SkinThickness':[self.SkinThickness],
                'Insulin':[self.Insulin],
                'BMI':[self.BMI],
                'DiabetesPedigeeFunction':[self.DiabetesPedigeeFunction],
                'Age':[self.Age]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created successfully")
            
            logging.info(f"DataFrame shape: {df.shape}")
            return df 
        except Exception as e:
            logging.error(f"Error in converting to DataFrame: {e}")
            raise CustomException(e,sys)




