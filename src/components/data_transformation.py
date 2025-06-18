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
from src.logger import logging

@dataclass 
class DataTransformerConfig:
    data_processor_file_obj_path:str = os.path.join('artifacts','processor.pkl')


class DataTranformaer:
    def __init__(self):
        logging.info("Initializing DataTransformer")
        self.data_transformer_config = DataTransformerConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformer object")
            numerical_fearures = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
            logging.info(f"Numerical features: {numerical_fearures}")

            logging.info("Creating numerical pipeline with StandardScaler")
            numerical_pipeline = Pipeline(
                steps=[
                    ('standardscalar',StandardScaler())
                ]
            )

            logging.info("Creating column transformer with numerical pipeline")
            preprocessor = ColumnTransformer(
            [
                ('numerical_pipeline',numerical_pipeline,numerical_fearures)
            ]
        )
            
            logging.info("Data transformer object created successfully")
            return preprocessor
        except Exception as e:
            logging.error(f"Error in creating data transformer object: {str(e)}")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):
        try:
            logging.info(f"Reading training dataset from {train_path}")
            train_df = pd.read_csv(train_path)
            logging.info(f"Training dataset shape: {train_df.shape}")
            
            logging.info(f"Reading test dataset from {test_path}")
            test_df = pd.read_csv(test_path)
            logging.info(f"Test dataset shape: {test_df.shape}")

            target_column = 'Outcome'
            logging.info(f"Target column identified: {target_column}")

            logging.info("Separating input and target features for training dataset")
            input_feature_train_df = train_df.drop([target_column],axis=1)
            taregt_feature_train_df = train_df[target_column]

            logging.info("Separating input and target features for testing dataset")
            input_feature_test_df = test_df.drop([target_column],axis=1)
            taregt_feature_test_df = test_df[target_column]

            logging.info("Obtaining the preprocessor object")
            preoprocessor_object = self.get_data_transformer_object()

            logging.info("Applying preprocessing on training data")
            input_feature_train_array = preoprocessor_object.fit_transform(input_feature_train_df)
            logging.info("Applying preprocessing on testing data")
            input_feature_test_array = preoprocessor_object.transform(input_feature_test_df)

            logging.info("Combining preprocessed features with target for training data")
            train_array = np.c_[
                input_feature_train_array,np.array(taregt_feature_train_df)
            ]
            logging.info(f"Training array shape: {train_array.shape}")

            logging.info("Combining preprocessed features with target for testing data")
            test_array = np.c_[
                input_feature_test_array,np.array(taregt_feature_test_df)
            ]
            logging.info(f"Testing array shape: {test_array.shape}")

            logging.info(f"Saving preprocessor object to {self.data_transformer_config.data_processor_file_obj_path}")
            save_object(self.data_transformer_config.data_processor_file_obj_path, preoprocessor_object)
            logging.info("Preprocessor object saved successfully")

            logging.info("Data transformation completed successfully")
            return (
                train_array,
                test_array
            )

        except Exception as e:
            logging.error(f"Exception occurred during data transformation: {str(e)}")
            print(e)
            raise CustomException(e,sys)