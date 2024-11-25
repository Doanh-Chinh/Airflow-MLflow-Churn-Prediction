import os
import logging
import time
import pandas as pd
from sqlalchemy import create_engine
from utils.db_tables import Base, InferencePredictAttritionTable


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', f'postgresql://prediction_user:password@postgres:{POSTGRES_PORT}/prediction_pg_db')
INFERENCE_PREDICTION_RESULTS_TABLE_NAME = os.getenv('INFERENCE_PREDICTION_RESULTS_TABLE_NAME', 'inference_prediction_results')

def create_prediction_pg_tables():
    logging.info('Connecting to database...')
    start = time.time()
    engine = create_engine(DB_CONNECTION_URL)
    # drop if exsist and create table
    if engine.has_table(InferencePredictAttritionTable.__tablename__):
        Base.metadata.drop_all(engine, tables=[InferencePredictAttritionTable.__table__])
    engine = create_engine(DB_CONNECTION_URL)
    Base.metadata.create_all(engine)

    logging.info(f'Create {INFERENCE_PREDICTION_RESULTS_TABLE_NAME} table took {time.time()-start:.3f} s')

if __name__ == "__main__":
    create_prediction_pg_tables()