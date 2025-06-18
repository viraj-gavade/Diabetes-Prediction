import sys
from src.exception_handler import CustomException
from src.utils import evaluate_model
from src.utils import save_object

from src.logger import logging
from dataclasses import dataclass
import os
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier



@dataclass
class ModelTrainerConfig :
    trained_model_path : str = os.path.join('artifacts','model.pkl')


class ModelTrainer :
    def __init__(self):
        self.trained_model_config_path = ModelTrainerConfig()



    def initiate_model_training(self,train_arry,test_arry):
        try:
            X_train, y_train, X_test, y_test = (
            train_arry[:, :-1],
             train_arry[:, -1],
        test_arry[:, :-1],
        test_arry[:, -1])




            models = {
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Support Vector Classifier": SVC(probability=True),
                "CatBoost": CatBoostClassifier(verbose=0),
                "XGBoost": XGBClassifier()
                }

            model_report : dict = evaluate_model(X_train,y_train,X_test,y_test,models)

            best_model_name = None
            best_model_score = 0

            best_model_name = None
            best_model_score = 0

            for model_name in model_report:
                train_score = model_report[model_name]['Train Accuracy Score']
                if train_score > best_model_score:
                    best_model_score = train_score
                    best_model_name = model_name



            logging.info(f'Best model score found: {best_model_score}')
            logging.info(f'Best performing model: {best_model_name}')

            best_model = models[best_model_name]

            logging.info(f'Saving the best model to {self.trained_model_config_path.trained_model_path}')
            save_object(filepath=self.trained_model_config_path.trained_model_path, obj=best_model)
            logging.info(f'Best model saved successfully')

            

        except Exception as e:
            print(e)
            raise CustomException(e,sys)