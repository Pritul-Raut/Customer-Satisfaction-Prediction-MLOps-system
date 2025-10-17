import logging
from abc import ABC,abstractmethod

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from typing import Union
class DataStrategy(ABC):
    """
    Abstract class definig strategy for handling data

    Args:
        ABC (_type_): _description_
    """
    def handle_data(self,data:pd.DataFrame)-> Union[pd.DataFrame ,pd.Series]:
        pass
    
    
class DataPreProcessStrategy(DataStrategy):
    """_summary_

    Strategy for Data Pre Processing 
    Args:
        DataStrategy (_type_): _description_
    """
    def handle_data(self, data):
        """
        PreProcess Data

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ],axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median())
            data["product_length_cm"].fillna(data["product_length_cm"].median())
            data["product_height_cm"].fillna(data["product_height_cm"].median())
            data["product_width_cm"].fillna(data["product_width_cm"].median())
            data["review_comment_message"].fillna("No review")
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix","order_item_id"]
            data=data.drop(cols_to_drop,axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing:{e}")
            raise e
        
class DataDivideStrategy:
          
    def handle_data (self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Strategy Divide Data Into Train and test

        Args:
            data (_type_): _description_

        Returns:
            Union[pd.DataFrame,pd.series]: _description_
        """
        try:
            X = data.drop(["review_score"],axis=1)
            y=data["review_score"]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e


class DataCleaning:
    """
    Class for cleaning data using a specified strategy.
    Processes the input DataFrame and divides it into train and test sets.
    """

    def __init__(self, data: pd.DataFrame, strategy: 'DataStrategy'):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling Data: {e}")
            raise e
        
# if __name__=="__main__":
#     data=""
#     data_cleaning=DataCleaning(data,DataPreProcessStrategy())
#     data_cleaning.handle_data()