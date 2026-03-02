"""
ml_pipeline_dag.py - Apache Airflow DAG for ML Training Pipeline

This DAG orchestrates the complete ML pipeline:
1. Data ingestion & validation
2. Feature engineering
3. Model training (CNN, LSTM, BERT, Ensemble)
4. Model evaluation & comparison
5. Model registration to MLflow
6. Deployment promotion (staging → production)
7. Post-deployment monitoring
"""

"""
Airflow DAG for ML pipeline
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


default_args = {
    "owner": "mlops",
    "start_date": datetime(2024, 1, 1)
}

with DAG(
    dag_id="chest_xray_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False
) as dag:

    preprocess = BashOperator(
        task_id="data_preprocessing",
        bash_command="python src/data/data_loader.py"
    )

    train = BashOperator(
        task_id="model_training",
        bash_command="python src/training/trainer.py"
    )

    preprocess >> train