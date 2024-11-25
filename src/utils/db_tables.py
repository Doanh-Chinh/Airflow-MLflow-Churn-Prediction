import os
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import declarative_base

CUS_ATTRITION_TABLE_NAME = os.getenv('CUS_ATTRITION_TABLE_NAME', 'customer_attrition') # data table
PREDICTION_RESULTS_TABLE_NAME = os.getenv('PREDICTION_RESULTS_TABLE_NAME', 'prediction_results') # prediction results table
INFERENCE_PREDICTION_RESULTS_TABLE_NAME = os.getenv('PREDICTION_RESULTS_TABLE_NAME', 'inference_prediction_results') # inference prediction results table



Base = declarative_base()

class CustomerAttritionTable(Base):
    __tablename__ = CUS_ATTRITION_TABLE_NAME
    id = Column(Integer, primary_key=True, autoincrement=True)
    attrition_flag = Column(String)
    customer_age = Column(Integer)
    gender = Column(String)
    dependent_count = Column(Integer)
    education_level = Column(String)
    marital_status = Column(String)
    income_category = Column(String)
    card_category = Column(String)
    months_on_book = Column(Integer)
    total_relationship_count = Column(Integer)
    months_inactive_12_mon = Column(Integer)
    contacts_count_12_mon = Column(Integer)
    credit_limit = Column(Float)
    total_revolving_bal = Column(Integer)
    avg_open_to_buy = Column(Float)
    total_amt_chng_q4_q1 = Column(Float)
    total_trans_amt = Column(Integer)
    total_trans_ct = Column(Integer)
    total_ct_chng_q4_q1 = Column(Float)
    avg_utilization_ratio = Column(Float)

class PredictAttritionTable(Base):
    __tablename__ = PREDICTION_RESULTS_TABLE_NAME
    id = Column(Integer, primary_key=True, autoincrement=True)
    actual = Column(Integer)
    prediction = Column(Integer)

class InferencePredictAttritionTable(Base):
    __tablename__ = INFERENCE_PREDICTION_RESULTS_TABLE_NAME
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_path = Column(String)
    prediction = Column(Integer)

