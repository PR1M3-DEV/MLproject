"""
Utility functions for the End-to-End Machine Learning Project.

This module provides helper functions for:
1. Saving Python objects
2. Loading Python objects
3. Evaluating Machine Learning models
"""

import os
import sys

import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


# ==========================================================
# Save Object
# ==========================================================
def save_object(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ==========================================================
# Load Object
# ==========================================================
def load_object(file_path):
    try:

        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ==========================================================
# Evaluate Models
# ==========================================================
def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param,
):
    """
    Train and evaluate multiple machine learning models.

    Returns
    -------
    dict
        Dictionary containing detailed information for every model.
    """

    try:

        report = {}

        logging.info("=" * 60)
        logging.info("Beginning Model Evaluation")
        logging.info("=" * 60)

        for model_name, model in models.items():

            logging.info(f"Training {model_name}")

            para = param.get(model_name, {})

            if para:

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    n_jobs=-1
                )

                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)

                best_params = gs.best_params_

            else:

                best_params = {}

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(
                y_train,
                y_train_pred
            )

            test_score = r2_score(
                y_test,
                y_test_pred
            )

            report[model_name] = {

                "train_score": train_score,

                "test_score": test_score,

                "best_params": best_params,

                "model": model

            }

            logging.info(f"Model : {model_name}")
            logging.info(f"Train R² : {train_score:.4f}")
            logging.info(f"Test  R² : {test_score:.4f}")
            logging.info(f"Best Parameters : {best_params}")
            logging.info("-" * 60)

        logging.info("=" * 60)
        logging.info("Completed Model Evaluation")
        logging.info("=" * 60)

        return report

    except Exception as e:
        raise CustomException(e, sys)