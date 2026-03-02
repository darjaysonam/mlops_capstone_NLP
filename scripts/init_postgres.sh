#!/bin/bash

set -e

echo "Initializing PostgreSQL databases..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE mlflow_db;
    CREATE DATABASE airflow_db;
EOSQL

echo "Databases created successfully."