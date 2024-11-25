import sqlalchemy
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Any
from collections import defaultdict
from sqlalchemy import func, and_
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from sqlalchemy import Column, Integer, String, Float

INFERENCE_PREDICTION_RESULTS_TABLE_NAME = os.getenv('PREDICTION_RESULTS_TABLE_NAME', 'inference_prediction_results')
Base = declarative_base()

class InferencePredictAttritionTable(Base):
    __tablename__ = INFERENCE_PREDICTION_RESULTS_TABLE_NAME
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_path = Column(String)
    prediction = Column(Integer)

def open_db_session(engine: sqlalchemy.engine) -> sqlalchemy.orm.Session:
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def unique_list_from_col(
    session: sqlalchemy.orm.Session, table, column: str
) -> List[Any]:
    unique_rets = session.query(getattr(table, column)).distinct().all()
    unique_list = [ret[0] for ret in unique_rets]
    return unique_list


def get_table_from_engine(engine: sqlalchemy.engine, table_name: str):
    Base = automap_base()
    Base.prepare(autoload_with=engine)
    table_obj = getattr(Base.classes, table_name)
    return table_obj


def df_from_query(sql_ret, use_cols) -> pd.DataFrame:
    data = defaultdict(list)
    for row in sql_ret:
        for col in use_cols:
            data[col].append(getattr(row, col))
    df = pd.DataFrame(data).set_index("id")
    return df
