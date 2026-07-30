"""
Utility functions for the End-to-End Machine Learning Project.

This module provides helper functions for:
1. Saving Python objects
2. Loading Python objects
3. Evaluating Machine Learning models
"""

import os
import sys
import time

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
    Train, tune and evaluate multiple machine learning models.

    Returns
    -------
    dict
        Dictionary containing detailed evaluation information
        for every trained model.
    """

    try:

        report = {}

        LINE = "=" * 60
        SUBLINE = "-" * 60

        logging.info(LINE)
        logging.info("Beginning Model Evaluation")
        logging.info(LINE)

        for model_name, model in models.items():

            logging.info(f"Training Model : {model_name}")

            start_time = time.time()

            para = param.get(model_name, {})

            best_params = {}
            cv_score = None

            if len(para) > 0:

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    scoring="r2",
                    cv=5,
                    n_jobs=-1,
                    refit=True,
                    return_train_score=True,
                )

                gs.fit(X_train, y_train)

                model = gs.best_estimator_

                best_params = gs.best_params_

                cv_score = gs.best_score_

            else:

                model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(
                y_train,
                y_train_pred,
            )

            test_score = r2_score(
                y_test,
                y_test_pred,
            )

            elapsed_time = time.time() - start_time

            report[model_name] = {

                "model": model,

                "train_score": train_score,

                "test_score": test_score,

                "cv_score": cv_score,

                "best_params": best_params,

                "training_time": elapsed_time,

            }

            logging.info(f"Train R²        : {train_score:.4f}")
            logging.info(f"Test  R²        : {test_score:.4f}")

            if cv_score is not None:
                logging.info(
                    f"Best CV R²      : {cv_score:.4f}"
                )

            logging.info(
                f"Best Parameters : {best_params}"
            )

            logging.info(
                f"Training Time   : {elapsed_time:.2f} seconds"
            )

            logging.info(SUBLINE)

        logging.info(LINE)
        logging.info("Completed Model Evaluation")
        logging.info(LINE)

        return report

    except Exception as e:

        raise CustomException(e, sys)