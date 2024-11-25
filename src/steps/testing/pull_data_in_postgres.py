# a script for manually test if the data insertion work correctly
import os
import pandas as pd
from collections import defaultdict
import sqlalchemy
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from utils.db_tables import CustomerAttritionTable

POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', f'postgresql://prediction_user:password@postgres:{POSTGRES_PORT}/prediction_pg_db')

def open_db_session(engine: sqlalchemy.engine) -> sqlalchemy.orm.Session:
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def query_top_rows(session, table, id_col='id', top_n=None):
    q = session.query(table)
    table_id = getattr(table, id_col)
    if top_n:
        q = q.filter(
                table_id <= top_n
            )
        ret = q.all()
    else:
        ret = q.order_by(table_id.asc()).all()
    return ret

def df_from_query(sql_ret, use_cols) -> pd.DataFrame:
    data = defaultdict(list)
    for row in sql_ret:
        for col in use_cols:
            data[col].append(getattr(row, col))
    # df = pd.DataFrame(data).set_index('id')
    df = pd.DataFrame(data)
    return df

def pull_data_in_postgres():
    engine = create_engine(DB_CONNECTION_URL)
    session = open_db_session(engine)

    ret = query_top_rows(session, CustomerAttritionTable, top_n=10)
    all_cols = [column.name for column in CustomerAttritionTable.__table__.columns]
    ret_df = df_from_query(ret, all_cols)
    ret_df = ret_df.sort_values('id')
    print(ret_df.tail())
    # ret_df.to_csv('data/dev.csv', index=False)