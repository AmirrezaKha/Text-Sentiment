-- Create a database for MLflow
CREATE DATABASE mlflow;

-- Create the user for MLflow
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';

-- Grant privileges to mlflow_user on the mlflow database
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;

-- Ensure user 'metaflow' has necessary access to its own default database (optional, based on your needs)
GRANT ALL PRIVILEGES ON DATABASE metadatabase TO metadatabase_user;
