from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from rag import DataBaseCollector

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 10, 26),
    "retries": 1,
}

dag = DAG(
    "update_database",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
)


def update_database():
    collector = DataBaseCollector(
        host="localhost",
        port=27017,
        database="confluence",
        collection_name="pages",
        context_size=200
    )
    collector.update_database()


update_database_task = PythonOperator(
    task_id="update_database",
    python_callable=update_database,
    dag=dag,
)

update_database_task