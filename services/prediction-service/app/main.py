import time
import datetime
import contextlib
import pandas as pd
import logging
import pprint
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from handlers.my_mlflow import MLflowHandler
from helpers import PredictionRequest, create_prediction_input
from fastapi.middleware.cors import CORSMiddleware
import mlflow

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

handlers = {}
MODEL_BASE_NAME = "churn_prediction"  # check the name in steps/config.py or mlflow web ui

def get_model():
    global models
    model_name = MODEL_BASE_NAME
    model, model_name, model_version = handlers["mlflow"].get_production_model(
        model_name=model_name
    )
    return model, model_name, model_version


async def get_service_handlers():
    global handlers
    mlflow_handler = MLflowHandler()
    handlers["mlflow"] = mlflow_handler
    logging.info("Retrieving mlflow handler...")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await get_service_handlers()
    yield


app = FastAPI(lifespan=lifespan)

# for local testing calls from JS
origins = [
    "http://localhost",
    "http://localhost:6969",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=200)
async def health_check():
    return {
        "serviceStatus": "OK",
        "modelTrackingHealth": handlers["mlflow"].check_mlflow_health(),
    }



@app.post("/prediction", status_code=200)
def prediction(prediction_request: List[PredictionRequest]):
    """
    Prediction steps:
    1. Load model production for prediction task by get_model().
    2. Create input for model prediction by create_prediction_input(PredictionRequest.batch_path) that is a List[PredictionRequest]
    this fuction will load batch data from the path, then return that batch input.
    3. Return a dictionary object that records prediction results.
    """
    logging.info(f"Succeed to entry post with request body as {prediction_request}")
    start = time.time()
    predictions = []
    for item in prediction_request: # Each item is a batch path, this process can be used for multiple list of batch data
        logging.info(
            f"Getting the production model with base name as {MODEL_BASE_NAME}"
        )
        # Step 1:
        model, model_name, model_version = get_model()
        logging.info(f"Got the model {model_name} version {model_version}")
        # Step 2:
        prediction_input = create_prediction_input(batch_path=item.batch_path)
        logging.info(f"prediction_input: {prediction_input}")
        # Step 3: 
        prediction_result = {}
        prediction_result["request"] = item.dict()
        model_pred = model.predict(prediction_input)
        prediction_result["prediction"] = model_pred.tolist()
        prediction_result["api"] = {
            "model_name": model_name,
            "model_version": model_version,
        }
        predictions.append(prediction_result)
    logging.info(
        f"Making predictions for {len(prediction_request)} entries took {time.time() - start:.4f} s"
    )
    logging.info(f"The prediction results are: {predictions}")
    return predictions
