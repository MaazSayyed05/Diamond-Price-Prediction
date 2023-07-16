import os,pickle,sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.logger import logging
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj) # file_obj as file_path 

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            # Predict Testing Data
            y_pred = model.predict(X_test)

            # Get r2 score for train and test data
            # train_model_score = r2_score(y_test,y_pred)
            test_model_score = r2_score(y_pred=y_pred,y_true=y_test)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logging.info("Error occured at Model Evaluation")
        return CustomException(e,sys)