import os
import sys

from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            # Read the dataset
            df = pd.read_csv("notebook/data/stud.csv")

            logging.info("Dataset read successfully as a pandas DataFrame")

            # Create artifacts directory if it doesn't exist
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Save raw dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train-test split initiated")

            # Split dataset
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logging.info("Train-test split completed")

            # Save training dataset
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save testing dataset
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    print("=" * 60)
    print("STEP 1 : DATA INGESTION")
    print("=" * 60)

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    print(f"Training Data : {train_path}")
    print(f"Testing Data  : {test_path}")

    print("\n" + "=" * 60)
    print("STEP 2 : DATA TRANSFORMATION")
    print("=" * 60)

    transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    print(f"Train Shape        : {train_arr.shape}")
    print(f"Test Shape         : {test_arr.shape}")
    print(f"Preprocessor Saved : {preprocessor_path}")

    print("\n" + "=" * 60)
    print("STEP 3 : MODEL TRAINING")
    print("=" * 60)

    trainer = ModelTrainer()

    r2_score = trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    print("\n")
    print("=" * 60)
    print("PROJECT EXECUTED SUCCESSFULLY")
    print("=" * 60)
    print(f"Final Test R² Score : {r2_score:.4f}")
    print("=" * 60)