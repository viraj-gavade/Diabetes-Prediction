from src.exception_handler import CustomException
from src.logger import logging
from src.utils import load_object
import numpy as np
import pandas as pd
import sys
import os




class PredictionPipeline:
    def __init__(self):
        
        pass


    def predict_result(self,features):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            model_path = os.path.join(base_dir,'artifacts','model.pkl')
            preprocessor_path = os.path.join(base_dir,'artifcats','processor.pkl')


            model = load_object(model_path)
            preprocessor = load_object(l=preprocessor_path) 


            scaled_data = preprocessor.transform(features) 

            prediction = model.predict(scaled_data)

            return prediction

        except Exception as e:
            print(e)
            raise CustomException(e,sys)
        


class CustomData :
    def __init__(self,Pregnancies:int,Glucose:int,BloodPressure:int,SkinThickness:int,Insulin:int,BMI:int,DiabetesPedigeeFunction:int,Age:int):
        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigeeFunction = DiabetesPedigeeFunction
        self.Age = Age


    def get_data_as_dataframe(self):
        try:    
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

            return df 
        except Exception as e :
            print(e)
            raise CustomException(e,sys)


        

