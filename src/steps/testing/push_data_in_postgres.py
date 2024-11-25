import os
import logging
import time
import pandas as pd
from sqlalchemy import create_engine
from utils.db_tables import Base, CustomerAttritionTable

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

CUS_ATTRITION_TABLE_NAME = os.getenv('CUS_ATTRITION_TABLE_NAME', 'customer_attrition')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', f'postgresql://prediction_user:password@postgres:{POSTGRES_PORT}/prediction_pg_db')

def push_data_in_postgres():
    # Can load data after preprocessing in ./data/preprocessed
    # Below, we process data from the original data
    logging.info('Reading data from csv...')
    df = pd.read_csv('data/BankChurners.csv')
    ori_cols_order = df.columns
    print("ori_cols_order:", ori_cols_order)

    logging.info('Processing data...')
    # Remove unnecessary columns
    df.drop(labels=['CLIENTNUM', df.columns[-1], df.columns[-2]], axis=1, inplace=True)
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    # Rename df columns from camelcase to lowercase that fit to CustomerAttritionTable schema
    df.columns = df.columns.str.lower()
    logging.info('Connecting to database...')
    engine = create_engine(DB_CONNECTION_URL)
    # drop if exsist and create table
    if engine.has_table(CustomerAttritionTable.__tablename__):
        Base.metadata.drop_all(engine, tables=[CustomerAttritionTable.__table__])
    engine = create_engine(DB_CONNECTION_URL)
    Base.metadata.create_all(engine)

    logging.info('Inserting dataframe to database...')
    start = time.time()
    df.to_sql(CUS_ATTRITION_TABLE_NAME, engine, if_exists='append', index=False)
    logging.info(f'Putting df to postgres took {time.time()-start:.3f} s')


if __name__ == "__main__":
    push_data_in_postgres()