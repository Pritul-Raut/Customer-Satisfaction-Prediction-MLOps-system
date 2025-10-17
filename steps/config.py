from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """model config 

    Args:
        BaseParameters (_type_): _description_
    """
    model_name:str="LinearRegression"
    