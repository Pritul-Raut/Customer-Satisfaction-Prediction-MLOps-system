import logging

import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train:pd.DataFrame,
    y_train:pd.DataFrame,
    X_test:pd.DataFrame,
    y_test:pd.DataFrame,
    config:ModelNameConfig,
                )->RegressorMixin:
    """ 
    Trains The Model On the ingested data

    Args:
        X_train (pd.DataFrame): _description_
        y_train (pd.DataFrame): _description_
        X_test (pd.DataFrame): _description_
        y_test (pd.DataFrame): _description_

    Returns:
        LinearRegressionModel: _description_
    """
    model = None
    try:
        if config.model_name=="LinearRegression":
            model=LinearRegressionModel()
            trained_model=model.train(X_train,y_train)
            return trained_model
        else :
            raise ValueError("Model {config.model_name} not supported")
    except Exception as e:
        logging.error("Error in training model : {e}")
        raise e

