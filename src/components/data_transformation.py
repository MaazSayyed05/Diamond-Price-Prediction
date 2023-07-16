
from src.logger import logging
from src.exception import CustomException

# Data Transformation
import pandas as pd
import numpy as np
import sys,os
from dataclasses import dataclass
from sklearn.model_selection import  train_test_split

# Data Transformation
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.impute import SimpleImputer # Handling Missing Values

from sklearn.pipeline import  Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from sklearn.metrics import  r2_score,mean_squared_error,mean_absolute_error

from src.utils import save_object

# Data Transformation Config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


# Data Transformation  Class
class DataTransformation:
        def __init__(self):
            self.data_transformation_config = DataTransformationconfig()

        def get_data_transformation_object(self):#pickle file
            try:
                logging.info("Data Transformation Initiated")
                # Define which columns should be ordinal-encoded and scaled
                categorical_cols = ['cut','color','clarity']
                numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
                
                # Define custome ranking for each ordinal variable (1,2,3,4,...)
                cut_categories = ["Fair","Good","Very Good","Premium","Ideal"]
                color_categories = ["D","E","F","G","H","I","J"]
                clarity_categories = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]


                logging.info("Pipeline Initiated")
                # Numerical Pipeline
                num_pipeline= Pipeline(
                    steps=[
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler',StandardScaler())
                    ]
                    
                )
                # logging.info("Pipeline Initiated: numerical")

                # Categorical Pipeline
                cat_pipeline = Pipeline(
                    steps=[
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                        ('scaler',StandardScaler())
                    ]
                )
                # logging.info("Pipeline Initiated: categorical")

                preprocessor = ColumnTransformer([
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',cat_pipeline,categorical_cols)

                ])

                # logging.info("Pipeline Initiated: combine")
                logging.info("Pipeline Completed")
                return preprocessor




            except Exception  as e:
                logging.info("Error in Data Transformation")
                raise CustomException(e,sys)
            
        def initiate_data_transformation(self,train_data_path,test_data_path):
            try:
                # Reading train and test data
                train_df = pd.read_csv(train_data_path)
                test_df = pd.read_csv(test_data_path)

                logging.info("Read train and test data completed")
                logging.info(f"Train DataFrame Head: {train_df.head().to_string()}")
                logging.info(f"Test DataFrame Head: {test_df.head().to_string()}")

                logging.info("Obtaining preprocessing object")

                preprocessing_obj = self.get_data_transformation_object()

                logging.info("preprocessing object acquired")
                target_column_name = 'price'
                drop_columns = [target_column_name,'id']

                # Independent and Dependent Features 
                # logging.info("Dependent and Independent")

                input_feature_train_df = train_df.drop(columns=drop_columns,axis=1) # axis=1
                # logging.info("Dependent and Independent2....1")
                target_feature_train_df = train_df[target_column_name]
                # logging.info("Dependent and Independent2....2")

                
                input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)# axis=1
                # logging.info("Dependent and Independent3.....1")
                target_feature_test_df = test_df[target_column_name]
                # logging.info("Dependent and Independent3.....2")

                # Apply teh transformation
                # logging.info("Dependent and Independent:4 ")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr  = preprocessing_obj.transform(input_feature_test_df)

                logging.info("Applying preprocessing object on training and testing datasets")
                train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)] # np [] as ()
                test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
                
                # logging.info("Before save file pickle")
                save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessing_obj
                )

                logging.info("Pickle file created successfully.")

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
            except Exception as e:
                CustomException(e,sys)














