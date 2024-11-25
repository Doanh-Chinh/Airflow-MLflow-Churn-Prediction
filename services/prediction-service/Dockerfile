FROM python:3.9.17-slim

ARG PREDICTION_SERVICE_PORT=$PREDICTION_SERVICE_PORT

COPY requirements.txt /service/requirements.txt

RUN pip install -r /service/requirements.txt

COPY app/ /service/app/

RUN echo $PREDICTION_SERVICE_PORT

EXPOSE $PREDICTION_SERVICE_PORT

WORKDIR /service/app

RUN pwd

CMD gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PREDICTION_SERVICE_PORT}