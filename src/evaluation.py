import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error
class Evaluation(ABC):
    """_summary_
    abstract class definig startegy for evalution our models  
    Args:
        ABC (_type_): _description_
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """_summary_
        calculates the score for the model

        Args:
            y_true (np.ndarray): _description_
            y_pred (np.array): _description_
        """
        
        pass
    
class MSE(Evaluation):
    """_summary_
    Evalutuion strategy that uses Mean Squared Error
    Args:
        Evaluation (_type_): _description_
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.error("Calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("MSE : {mse}")
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {e}")
            raise e

class R2(Evaluation):
    """_summary_

    Args:
        Evaluation (_type_): _description_
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.error("Calculating r2")
            r2 = r2_score(y_true,y_pred)
            logging.info("r2 : {r2}")
            return r2
        except Exception as e:
            logging.error("Error in calculating r2: {e}")
            raise e

class RMSE(Evaluation):
    """_summary_

    Args:
        Evaluation (_type_): _description_
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.error("Calculating RMSE")
            rmse = root_mean_squared_error(y_true,y_pred)
            
            logging.info("rmse : {rmse}")
            return rmse
        except Exception as e:
            logging.error("Error in calculating rmse: {e}")
            raise e