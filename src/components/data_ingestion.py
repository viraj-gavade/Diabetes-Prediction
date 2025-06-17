## Data Ingestion Pipeline
import os
import sys
import pandas as pd
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)



from src.logger import logging
from src.exception_handler import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


from src.components.data_transformation import DataTranformaer,DataTransformerConfig

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw.csv')



class DataIngestion:
    def __init__(self):
        self.data_ingestion_config_path = DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('Notebook/diabetes.csv')
            
            os.makedirs(os.path.dirname(self.data_ingestion_config_path.train_data_path),exist_ok=True)


            df.to_csv(self.data_ingestion_config_path.raw_data_path)


            train_set , test_set = train_test_split(df,train_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion_config_path.train_data_path,index=False,header=True)


            test_set.to_csv(self.data_ingestion_config_path.test_data_path,index=False,header=True)  



            return (
                self.data_ingestion_config_path.train_data_path,
                self.data_ingestion_config_path.test_data_path,
                
            )



        except Exception as e:
            print(e)
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Train Data Path: {train_data}")
    print(f"Test Data Path: {test_data}")
    print("Data ingestion completed successfully.")
    logging.info("Data ingestion completed successfully.")
    data_transformer = DataTranformaer()
    data_transformer.initiate_data_transformation(train_data,test_data)
