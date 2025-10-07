# from zenml import pipelines
from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_Data import clean_data
from steps.model_train import train_model
from steps.evalution import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    df=ingest_data(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)
    
    