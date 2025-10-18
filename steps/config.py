from pydantic import BaseModel, Field

class ModelNameConfig(BaseModel):
    """
    Configuration for model selection.

    Attributes:
        model_name (str): Name of the model to use. Defaults to 'LinearRegression'.
    """
    model_name: str = Field(default="LinearRegression", description="Name of the model to use")