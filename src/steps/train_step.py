from typing import Dict, Any
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import pandas as pd
import mlflow

from steps.config import TrainerConfig, MlFlowConfig

from sqlalchemy import create_engine
from utils.db_tables import Base, PredictAttritionTable
import os
import logging
import time

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

PREDICTION_RESULTS_TABLE_NAME = os.getenv('PREDICTION_RESULTS_TABLE_NAME', 'prediction_results')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', f'postgresql://prediction_user:password@postgres:{POSTGRES_PORT}/prediction_pg_db')

class TrainStep:
    """Training step tracking experiments with MLFlow.
    In this case, GradientBoostingClassifier/LogisticRegression has been picked, and the chosen metrics are:
    * precision
    * recall
    * roc_auc
    
    Args:
        estimator_name (str): The name of estimator or model is used for training process."""

    def __init__(
            self,
            estimator_name: str
    ) -> None:
        self.params = TrainerConfig.estimator[estimator_name]["params"]  # Parameters of the model. Have to match the model arguments.
        self.model_name = TrainerConfig.estimator[estimator_name]["model_name"] # Additional information for experiments tracking.

    def __call__(
            self,
            train_path: Path,
            test_path: Path,
            target: str
        ) -> None:

        mlflow.set_tracking_uri(MlFlowConfig.uri)
        mlflow.set_experiment(MlFlowConfig.experiment_name)
        
        with mlflow.start_run():

            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            
            # Train
            if self.model_name == "gradient-boosting":
                estimator = GradientBoostingClassifier(
                    random_state=TrainerConfig.random_state,
                    verbose=True,
                    **self.params
                )

            elif self.model_name == "logistic-regression":
                estimator = LogisticRegression(
                    random_state=TrainerConfig.random_state,
                    verbose=True,
                    **self.params
                )

            model = estimator.fit(
                train_df.drop(target, axis=1),
                train_df[target]
            )

            # Evaluate
            X_test = test_df.drop(target, axis=1)
            y_test = test_df[target]
            y_pred = model.predict(X_test)
            
            # Create prediction df
            predictions = {"actual": y_test, "prediction": y_pred}
            df = pd.DataFrame(predictions)
            print(df.head(10))
            # Connect to DB
            engine = create_engine(DB_CONNECTION_URL)
            # drop if exsist and create table
            if engine.has_table(PredictAttritionTable.__tablename__):
                Base.metadata.drop_all(engine, tables=[PredictAttritionTable.__table__])
            engine = create_engine(DB_CONNECTION_URL)
            Base.metadata.create_all(engine)

            logging.info('Inserting dataframe to database...')
            start = time.time()
            df.to_sql(PredictAttritionTable.__tablename__, engine, if_exists='append', index=False)
            logging.info(f'Putting df to postgres took {time.time()-start:.3f} s')

            # Metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            y_score = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_true=y_test, y_score=y_score) # y_score contains probabilities (not class predictions like 0 or 1). https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.roc_auc_score.html
            print(classification_report(y_test, y_pred))

            metrics = {
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc
            }

            # Mlflow
            mlflow.log_params(self.params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag(key=MlFlowConfig.tag_key, value=self.model_name)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=MlFlowConfig.artifact_path,      
            )

            return {"mlflow_run_id": mlflow.active_run().info.run_id}
