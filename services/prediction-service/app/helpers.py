from pydantic import BaseModel
from typing import Union
import datetime
import pandas as pd


class PredictionRequest(BaseModel): # This structure is more easy to expand if need, recommend keep it.
    batch_path: str


def create_prediction_input(batch_path: str):
    """Load dataframe from path"""
    batch = pd.read_parquet(batch_path)
    return batch
