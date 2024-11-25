from sqlalchemy import create_engine
import os
import logging
import time
from collections import defaultdict
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from db_utils import Base, InferencePredictAttritionTable


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5050")
# Initialize MLflow client
client = MlflowClient()
# Registered model name 
registered_model_name = "churn_prediction" # must be consistency with steps/config.py


def build_prediction_request_body(batch_path: str):
    predict_requests = []
    user_request = {
        "batch_path": batch_path,
        # TODO: can improve
    }
    predict_requests.append(user_request)
    return predict_requests

def df_from_prediction_response(predition_response):
    data = defaultdict(list)
    for result in predition_response:
        predictions = result["prediction"]
        request = result["request"]
        api = result["api"]
        for prediction in predictions:
            data["batch_path"].append(request["batch_path"])
            data["prediction"].append(prediction)
    prediction_df = pd.DataFrame(data) # match to Inference Prediction Results table in postgres db
    return prediction_df


def save_inference_results_to_db(predition_response):
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', f'postgresql://prediction_user:password@postgres:{POSTGRES_PORT}/prediction_pg_db')
    # Connect to DB
    engine = create_engine(DB_CONNECTION_URL)
    # Drop if exsist and create table
    if engine.has_table(InferencePredictAttritionTable.__tablename__):
        Base.metadata.drop_all(engine, tables=[InferencePredictAttritionTable.__table__])
    engine = create_engine(DB_CONNECTION_URL)
    Base.metadata.create_all(engine)

    logging.info('Inserting dataframe to database...')
    start = time.time()
    df_prediction_result = df_from_prediction_response(predition_response)
    df_prediction_result.to_sql(InferencePredictAttritionTable.__tablename__, engine, if_exists='append', index=False)
    logging.info(f'Putting df to postgres took {time.time()-start:.3f} s')
    return "SUCCESS"

# Function to retrieve list of parquet files in subdirectories
def list_parquet_files(data_dir):
    logging.info(f"Enter list_parquet_files(data_dir) with parameter data_dir={data_dir}")
    parquet_files = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            logging.info(f"File: {file}")
            if file.endswith(".parquet"):
                # Build full path and relative path
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, data_dir)
                parquet_files[relative_path] = full_path
    logging.info(f"List of parquet files{parquet_files}")
    return parquet_files

# Retrieve all registered models
def get_registered_models_info():
    logging.info(f"Enter get_registered_models_info()")
    models_info = []
    registered_models = []
    filter_string = f"name='{registered_model_name}'"
    logging.info(f"filter_string: {filter_string}")
    for model in client.search_registered_models(filter_string=filter_string, max_results=5):
        model_name = model.name
        for model_version in model.latest_versions:
            version_number = model_version.version
            run_id = model_version.run_id
            metrics = client.get_run(run_id).data.metrics
            params = client.get_run(run_id).data.params
            tags = model_version.tags
            model_info = {
                "Model Name": model_name,
                "Version": version_number,
                "Run ID": run_id,
                "Metrics": metrics,
                "Parameters": params,
                "Tags": tags
            }
            model = mlflow.sklearn.load_model(model_uri=f"models:/{registered_model_name}/{version_number}")
            models_info.append(model_info)
            registered_models.append(model)  # For plot ROC
    logging.info(f"Models info: {models_info}")
    return models_info, registered_models


def load_model(registered_model_name = registered_model_name):
    """Load model from model registry.

    Args:
        registered_model_name (str): Name

    Returns:
        Model artifact
    """
    models = mlflow.search_registered_models(filter_string=f"name = '{registered_model_name}'")
    logging.info(f"Models in the model registry: {models}")
    if models:
        latest_model_version = models[0].latest_versions[-1].version
        tag_key = "model"
        tag_value = models[0].latest_versions[-1].tags[tag_key]
        logging.info(f"Latest model version in the model registry used for prediction: {latest_model_version}")
        model = mlflow.sklearn.load_model(model_uri=f"models:/{registered_model_name}/{latest_model_version}")
        return model, tag_value
    else:
        logging.warning(f"No model in the model registry under the name: {registered_model_name}.")
        return

def load_batch_df(batch_path: Path) -> pd.DataFrame:
    """Load dataframe from path"""
    batch = pd.read_parquet(batch_path)
    # Log column names and their respective data types
    logging.info(f"Batch columns and data types:\n{batch.dtypes}")
    return batch

