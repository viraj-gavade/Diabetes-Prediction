from dataclasses import dataclass
import os
import numpy as np
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception_handler import CustomException
from src.utils import save_object

@dataclass 
class DataTransformerConfig:
    data_processor_file_obj_path:str = os.path.join('artifacts','processor.pkl')


class DataTranformaer:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()

    def get_data_transformer_object(self):
        try:
            numerical_fearures = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

            numerical_pipeline = Pipeline(
                steps=[
                    ('standardscalar',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
            [
                ('numerical_pipeline',numerical_pipeline,numerical_fearures)
            ]
        )
            
            return preprocessor
        except Exception as e :
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = 'Outcome'

            input_feature_train_df = train_df.drop([target_column],axis=1)
            taregt_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop([target_column],axis=1)
            taregt_feature_test_df = test_df[target_column]

            preoprocessor_object = self.get_data_transformer_object()

            input_feature_train_array = preoprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_array = preoprocessor_object.transform(input_feature_test_df)


            train_array = np.c_[
                input_feature_train_array,np.array(taregt_feature_train_df)
            ]

            test_array = np.c_[
                input_feature_test_array,np.array(taregt_feature_test_df)
            ]


            save_object(self.data_transformer_config.data_processor_file_obj_path,preoprocessor_object)

            return (
                train_array,
                test_array
            )

        except Exception as e:
                print(e)
                raise CustomException(e,sys)