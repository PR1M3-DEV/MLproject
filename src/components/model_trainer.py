import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    save_object,
    evaluate_models,
)


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
    ):

        try:

            logging.info("=" * 60)
            logging.info("Starting Model Training")
            logging.info("=" * 60)

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Linear Regression":
                    LinearRegression(),

                "Decision Tree":
                    DecisionTreeRegressor(
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestRegressor(
                        random_state=42
                    ),

                "Gradient Boosting":
                    GradientBoostingRegressor(
                        random_state=42
                    ),

                "AdaBoost":
                    AdaBoostRegressor(
                        random_state=42
                    ),

                "XGBoost":
                    XGBRegressor(
                        random_state=42,
                        verbosity=0
                    ),

                "CatBoost":
                    CatBoostRegressor(
                        random_state=42,
                        verbose=False,
                        allow_writing_files=False   
                    ),
            }

            params = {

                "Linear Regression": {},

                "Decision Tree": {

                    "criterion": [
                        "squared_error",
                        "absolute_error"
                    ],

                    "max_depth": [
                        None,
                        5,
                        10,
                        20
                    ]
                },

                "Random Forest": {

                    "n_estimators": [
                        100,
                        200
                    ],

                    "max_depth": [
                        None,
                        10,
                        20
                    ]
                },

                "Gradient Boosting": {

                    "learning_rate": [
                        0.01,
                        0.1
                    ],

                    "n_estimators": [
                        100,
                        200
                    ],

                    "max_depth": [
                        3,
                        5
                    ]
                },

                "AdaBoost": {

                    "learning_rate": [
                        0.01,
                        0.1,
                        1.0
                    ],

                    "n_estimators": [
                        50,
                        100
                    ]
                },

                "XGBoost": {

                    "learning_rate": [
                        0.01,
                        0.1
                    ],

                    "n_estimators": [
                        100,
                        200
                    ],

                    "max_depth": [
                        3,
                        6
                    ]
                },

                "CatBoost": {

                    "depth": [
                        4,
                        6,
                        8
                    ],

                    "learning_rate": [
                        0.01,
                        0.1
                    ],

                    "iterations": [
                        100,
                        200
                    ]
                }

            }

            model_report = evaluate_models(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models,
                param=params,
            )

            logging.info("=" * 60)
            logging.info("Complete Model Report")
            logging.info(model_report)
            logging.info("=" * 60)

            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["test_score"]
            )

            best_model_info = model_report[
                best_model_name
            ]

            best_model = best_model_info[
                "model"
            ]

            best_model_score = best_model_info[
                "test_score"
            ]

            best_params = best_model_info[
                "best_params"
            ]

            logging.info("=" * 60)
            logging.info("Best Model Summary")
            logging.info(f"Model : {best_model_name}")
            logging.info(
                f"Train R² : {best_model_info['train_score']:.4f}"
            )
            logging.info(
                f"Test R² : {best_model_score:.4f}"
            )
            logging.info(
                f"Best Parameters : {best_params}"
            )
            logging.info("=" * 60)

            if best_model_score < 0.60:

                raise CustomException(

                    f"""

No suitable model found.

Best Model:
{best_model_name}

Training R²:
{best_model_info['train_score']:.4f}

Testing R²:
{best_model_score:.4f}

Best Hyperparameters:
{best_params}

Minimum Required Test R²:
0.60

""",

                    sys

                )

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model

            )

            prediction = best_model.predict(
                X_test
            )

            r2_square = r2_score(
                y_test,
                prediction
            )

            logging.info("=" * 60)
            logging.info("Training Completed Successfully")
            logging.info(f"Selected Model : {best_model_name}")
            logging.info(f"Final Test R² : {r2_square:.4f}")
            logging.info(
                f"Model saved to : {self.model_trainer_config.trained_model_file_path}"
            )
            logging.info("=" * 60)

            return r2_square

        except Exception as e:

            raise CustomException(e, sys)