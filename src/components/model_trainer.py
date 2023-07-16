# After training_pipeline.py
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
import sys,os,pickle
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent from train and test arrays")
            X_train, X_test, y_train, y_test = (
            train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso'           : Lasso(),
                'ElasticNet'      : ElasticNet(),
                'Ridge'           : Ridge()
            }

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("\n")
            print("="*40)

            logging.info(f"Model Report :{model_report}")

            # To get best model score from dict.
            best_model_score = max(sorted(model_report.values()))

            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            print(f"Best Model Found, Model Name:{best_model_name}, R2 Score: {best_model_score}")
            print("="*40)
            logging.info(f"Best Model Found, Model Name:{best_model_name}, R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


        except Exception as e:
            logging.info("Error occured at Model Training")
            CustomException(e,sys)



