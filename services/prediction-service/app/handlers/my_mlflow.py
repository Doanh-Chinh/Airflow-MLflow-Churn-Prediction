import os
import mlflow
from typing import Tuple
from pprint import pprint
from mlflow.client import MlflowClient
from mlflow.pyfunc import PyFuncModel


class MLflowHandler:
    def __init__(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def check_mlflow_health(self) -> None:
        try:
            experiments = mlflow.search_experiments()
            for rm in experiments:
                pprint(dict(rm), indent=4)
                return "Service returning experiments"
        except:
            return "Error calling MLflow"

    def get_production_model(self, model_name: str) -> Tuple[PyFuncModel, str, str]:
        # mlflow.pyfunc.get_model_dependencies(model_uri=f"models:/{model_name}/production")
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/production")
        latest_versions_metadata = self.client.get_latest_versions(name=model_name, stages=["production"])
        model_version = latest_versions_metadata[0].version
        latest_model_version_metadata = self.client.get_model_version(
            name=model_name, version=model_version
        )
        latest_model_run_id = latest_model_version_metadata.run_id
        roc_auc = self.client.get_metric_history(run_id=latest_model_run_id, key="roc_auc")
        # if this the case meaning that the model is not good enough
        # we can implement the logic to rollback or trigger the retraining
        print("roc_auc of the Retrieved model:", roc_auc)
        roc_auc_value = roc_auc[0].value
        if roc_auc_value < 0.5:
            print(f"This model roc_auc = {roc_auc_value} is too low, did not pass the criteria")
            print("Do something...")
        return model, model_name, model_version
