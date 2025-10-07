import logging
import pandas as pd 
from zenml import step
from src.data_cleaning import DataCleaning,DataPreProcessStrategy,DataDivideStrategy
from typing import Annotated,Tuple
@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]
]:
    """_summary_
    cleans the data and divides it into train and test
    Args:
        df:Raw data
    Returns:
    X_train:
    X_test:
    y_train:
    y_test:

    Raises:
        e: _description_
    """
    
    
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df,process_strategy)
        processed_data=data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
    except Exception as e:
        logging.error("Error in cleaning data : {e} ")
        raise e