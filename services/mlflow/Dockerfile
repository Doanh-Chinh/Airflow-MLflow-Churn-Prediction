FROM python:3.9.17-slim

USER root

ARG MLFLOW_PORT=$MLFLOW_PORT
ARG POSTGRES_PORT=$POSTGRES_PORT
ARG MLFLOW_ARTIFACT_ROOT=$MLFLOW_ARTIFACT_ROOT


COPY requirements.txt .

RUN pip install -r requirements.txt

# Set up the SQLite database URI as an environment variable
# ENV BACKEND_URI sqlite:///mlflow.db
RUN echo $MLFLOW_PORT
RUN echo $POSTGRES_PORT
RUN echo $MLFLOW_ARTIFACT_ROOT
ENV BACKEND_URI postgresql://mlflow_user:password@postgres:$POSTGRES_PORT/mlflow_pg_db

EXPOSE $MLFLOW_PORT

# MLFLOW_ARTIFACT_ROOT must be mounted/accessible for all mlflow server and clients
CMD mlflow server \
    --host 0.0.0.0 --port $MLFLOW_PORT \
    --backend-store-uri $BACKEND_URI \
    --default-artifact-root file://$MLFLOW_ARTIFACT_ROOT \
    --serve-artifacts