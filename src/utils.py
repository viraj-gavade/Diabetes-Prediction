import os
import dill
from src.exception_handler import CustomException
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import sys


def save_object(filepath,obj):
    try:  
        dir_path = os.path.dirname(filepath)

        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_model(X_train,y_train,X_test,y_test,models):
    model_report = {}

    for i in range(len(list(models))):
        model = list(models.values())[i]

        print("Sample y_train:", y_train[:5])
        print("y_train dtype:", y_train.dtype)

        model.fit(X_train,y_train)

        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)


        cm_train = confusion_matrix(y_train,y_pred_train)
        cm_test = confusion_matrix(y_test,y_pred_test)

        clf_report_test = classification_report(y_test,y_pred_test)
        clf_report_train = classification_report(y_train,y_pred_train)

        train_accuracy = accuracy_score(y_train,y_pred_train)
        test_accuracy = accuracy_score(y_test,y_pred_test)


        model_report[list(models.keys())[i]] = {
    'Train Confusion Matrix': cm_train,
    'Test Confusion Matrix': cm_test,
    'Train Classification Report': clf_report_train,
    'Test Classification Report': clf_report_test,
    'Train Accuracy Score': train_accuracy,
    'Test Accuracy Score': test_accuracy,
}


        return model_report

def load_object(filepath):
    try:  
        with open(filepath,'rb') as file_obj:
          return  dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)