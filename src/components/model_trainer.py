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
                logging.info(f"Model: {model_name}, Train Accuracy: {train_score:.4f}, Test Accuracy: {model_report[model_name]['Test Accuracy Score']:.4f}")
                if train_score > best_model_score:
                    best_model_score = train_score
                    best_model_name = model_name
                    logging.debug(f"Found better model: {best_model_name} with score: {best_model_score:.4f}")

            logging.info(f"Best model evaluation complete")
            logging.info(f"Best model score found: {best_model_score:.4f}")
            logging.info(f"Best performing model: {best_model_name}")

            best_model = models[best_model_name]
            
            # Get test accuracy of the best model for reporting
            test_accuracy = model_report[best_model_name]['Test Accuracy Score']
            logging.info(f"Best model test accuracy: {test_accuracy:.4f}")
            
            # Save model file path
            model_path = self.trained_model_config_path.trained_model_path
            logging.info(f"Saving the best model ({best_model_name}) to {model_path}")
            
            try:
                save_object(filepath=model_path, obj=best_model)
                logging.info(f"Best model saved successfully")
                
                # Return model details for reporting
                return {
                    "best_model_name": best_model_name,
                    "train_accuracy": best_model_score,
                    "test_accuracy": test_accuracy,
                    "model_path": model_path
                }
            except Exception as e:
                logging.error(f"Error saving model: {str(e)}")
                raise CustomException(f"Model saving failed: {str(e)}", sys)

        except Exception as e:
            logging.error(f"Exception occurred during model training: {str(e)}")
            print(e)
            raise CustomException(e,sys)