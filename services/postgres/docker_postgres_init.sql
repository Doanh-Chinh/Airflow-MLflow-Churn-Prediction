CREATE USER mlflow_user WITH PASSWORD 'password';
CREATE DATABASE mlflow_pg_db
    WITH 
    OWNER = mlflow_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;
GRANT ALL PRIVILEGES ON DATABASE mlflow_pg_db TO mlflow_user;
-- PostgreSQL 15 requires additional privileges:
GRANT ALL ON SCHEMA public TO mlflow_user;

CREATE USER airflow_user WITH PASSWORD 'password';
CREATE DATABASE airflow_pg_db
    WITH 
    OWNER = airflow_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;
GRANT ALL PRIVILEGES ON DATABASE airflow_pg_db TO airflow_user;
GRANT ALL ON SCHEMA public TO airflow_user;

CREATE USER prediction_user WITH PASSWORD 'password';
CREATE DATABASE prediction_pg_db
    WITH 
    OWNER = prediction_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;
GRANT ALL PRIVILEGES ON DATABASE prediction_pg_db TO prediction_user;
GRANT ALL ON SCHEMA public TO prediction_user;