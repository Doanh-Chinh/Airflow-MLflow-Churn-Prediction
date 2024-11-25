# Only for testing purpose
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from steps.testing.push_data_in_postgres import push_data_in_postgres
from steps.testing.pull_data_in_postgres import pull_data_in_postgres
from steps.testing.reate_prediction_pg_tables import create_prediction_pg_tables
from steps.testing.prediction_requests import get_mlflow_health



default_args = {
    "owner": "user",                     # user's name
    "depends_on_past": False,            # keeps a task from getting triggered if the previous schedule for the task hasn’t succeeded.
    "retries": 0,                        # Number of retries for a dag 
    "catchup": False,                    # Run the dag from the start_date to today in respect to the trigger frequency 
}

with DAG(
    "data_load",                        # Dag name
    default_args=default_args,           # Default dag's arguments that can be share accross dags 
    start_date=datetime(2024, 11, 22),   # Reference date for the scheduler (mandatory)
    tags=["data_load"],                   # tags
    schedule=None,                       # No repetition
) as dag:
    # Importance! All below tasks are optional, only for testing purpose!
    create_prediction_tables_task = PythonOperator(
        task_id="create_prediction_tables",
        python_callable=create_prediction_pg_tables,
    ) # create inference prediction results table in postgres

    get_mlflow_health_task = PythonOperator(
        task_id="get_mlflow_health",
        python_callable=get_mlflow_health,
    ) # get status for testing API from prediction-service
    push_data_task = PythonOperator(
        task_id="push_data",
        python_callable=push_data_in_postgres,
    ) # create and push data to store on postgres
    pull_data_task = PythonOperator(
        task_id="pull_data",
        python_callable=pull_data_in_postgres,
    ) # pull for testing push_data_task ran correctly
    create_prediction_tables_task >> get_mlflow_health_task>> push_data_task >> pull_data_task
