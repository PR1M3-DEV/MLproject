import os
import sys

import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates the preprocessing pipeline.
        """

        try:
            # Numerical columns
            numerical_columns = [
                "writing score",
                "reading score"
            ]

            # Categorical columns
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    ),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical pipeline created")
            logging.info("Categorical pipeline created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessing object created")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):
        """
        Reads train/test CSVs, applies preprocessing,
        saves the preprocessor,
        and returns transformed train/test arrays.
        """

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test datasets loaded")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"

            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and testing datasets."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)