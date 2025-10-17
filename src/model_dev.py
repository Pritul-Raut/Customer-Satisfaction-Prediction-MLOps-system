import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    abstract class for all models 

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def train(self,X_train,y_train):
        """_summary_
        trains the model 

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model

    Args:
        Model (_type_): _description_
    """
    def train(self,X_train,y_train,**kwargs):
        """
        train the model

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        try:
            reg=LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model  training completed ")
            return reg
        except Exception as e:
            logging.error("Error in Training Model: {e}")
            raise e