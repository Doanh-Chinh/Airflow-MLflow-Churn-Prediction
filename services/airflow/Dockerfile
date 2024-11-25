FROM apache/airflow:2.8.3-python3.9

USER root

ARG MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT}

# Config path dir and permission for mlflow artifacts storage
# cuz mlflow also stores artifacts to client's container
# mkdir
# Changes the ownership to user airflow
# allowing the group (g) and others (o) to read and execute files, recursively
RUN echo "Starting"
RUN echo ${AIRFLOW_HOME}
RUN echo $MLFLOW_ARTIFACT_ROOT
RUN mkdir -p $MLFLOW_ARTIFACT_ROOT && \
    chown -R airflow $MLFLOW_ARTIFACT_ROOT && \ 
    chmod -R go+rX $MLFLOW_ARTIFACT_ROOT

# Install Airflow & Python packages
USER airflow

COPY ./requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# # Migrate DB & create account
# RUN airflow db migrate \
#     && airflow users create \
#     --username admin \
#     --password admin \
#     --firstname Admin \
#     --lastname Admin \
#     --role Admin \
#     --email admin@localhost

# # Init webserver & scheduler
# ENTRYPOINT ["airflow", "standalone"]

# Initialize the database and create the default user
ENTRYPOINT ["/bin/bash", "-c", "\
    airflow db migrate && \
    airflow users create \
        --username admin \
        --password admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@localhost && \
    airflow standalone"]
