import requests
import os
import logging


MLFLOW_HEALTH_ENDPOINT_URL = "http://prediction-service:4242/health"
def get_mlflow_health():
    logging.info(f"Request get from {MLFLOW_HEALTH_ENDPOINT_URL}")
    res = requests.get(MLFLOW_HEALTH_ENDPOINT_URL)
    res = res.json()
    print(res)
    logging.info("Success get mlflow health:", res)
    return res
