import logging
import pandas as pd
from zenml import step

class ingestData:
    """_summary_
    
    Args:
        data_path (str): _description_
        
    Ingesting The Data From The Data_Path
    """
    def __init__(self,data_path):
        self.data_path=data_path
    
    def get_data(self):
        logging.info(f'Ingesting data from {self.data_path}')
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path:str)->pd.DataFrame:
    """_summary_
    Ingesting The Data From The Data Path

    Args:
        data_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    try:
        ingest_data = ingestData(data_path)
        df = ingest_data.get_data()
        return df 
    except Exception as e:
        logging.error(f"Error while Ingesting Data : {e}")
        return e