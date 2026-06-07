import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):

    try:
        logging.info(f"Saving object at: {file_path}")

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Object saved successfully")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a pickle object from file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        from sklearn.metrics import r2_score

        report = {}

        for name, model in models.items():
            logging.info(f"Training model: {name}")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = r2_score(y_test, y_pred)

            report[name] = score

            logging.info(f"{name} R2 Score: {score}")

        return report

    except Exception as e:
        raise CustomException(e, sys)