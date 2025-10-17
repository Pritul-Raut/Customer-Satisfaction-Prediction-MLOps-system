# from zenml import pipelines
from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_Data import clean_data
from steps.model_train import train_model
from steps.evalution import evaluate_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=True)
def train_pipeline(data_path:str):
    df=ingest_data(data_path)
    X_train,X_test,y_train,y_test = clean_data(df)
    model_config = ModelNameConfig(model_name="LinearRegression")
    model = train_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,config=model_config)
    r2_score,rmse = evaluate_model(model=model,X_test=X_test,y_test=y_test)
    
    