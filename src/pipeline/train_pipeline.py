import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    """
    Executes the complete machine learning training pipeline.

    Pipeline Flow:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training
    """

    def __init__(self):
        pass

    def run_pipeline(self):

        try:

            logging.info("=" * 80)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("=" * 80)

            ####################################################
            # STEP 1 - DATA INGESTION
            ####################################################

            logging.info("STEP 1 : DATA INGESTION")

            ingestion = DataIngestion()

            train_path, test_path = (
                ingestion.initiate_data_ingestion()
            )

            logging.info(f"Training file : {train_path}")
            logging.info(f"Testing file  : {test_path}")

            ####################################################
            # STEP 2 - DATA TRANSFORMATION
            ####################################################

            logging.info("STEP 2 : DATA TRANSFORMATION")

            transformation = DataTransformation()

            train_arr, test_arr, _ = (
                transformation.initiate_data_transformation(
                    train_path,
                    test_path
                )
            )

            logging.info("Data transformation completed successfully")

            ####################################################
            # STEP 3 - MODEL TRAINING
            ####################################################

            logging.info("STEP 3 : MODEL TRAINING")

            trainer = ModelTrainer()

            r2_score = trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info("=" * 80)
            logging.info("PIPELINE COMPLETED SUCCESSFULLY")
            logging.info(f"Final Test R² Score : {r2_score:.4f}")
            logging.info("=" * 80)

            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.run_pipeline()