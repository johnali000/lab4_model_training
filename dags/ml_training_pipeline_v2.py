import json
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import airflow.utils.helpers
import joblib
from utils.s3 import S3

from ml_pipeline import data

# Default arguments
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'ml_training_pipeline_v2',
    default_args=default_args,
    description='ML model training pipeline',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

def train_model():
    dataset = datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    # We keep evaluation in the evaluate_model task to avoid passing sklearn objects via XCom
    joblib.dump(model, "model/model.pkl")


def evaluate_model(model, X_test, y_test):
    model = joblib.load("model/model.pkl")
    accuracy = accuracy_score(y_test, model.predict(X_test))
    model_performance = {'accuracy': accuracy}

    file_path = os.path.join('model', 'metric.json')
    with open(file_path, 'w') as f:
        json.dump(model_performance, f, indent=4)

def promote_model(**context):
    model_accuracy = json.load(open(os.path.join('model', 'metric.json'))).get('accuracy')
    if model_accuracy is None:
        raise ValueError('model accuracy not found in model/metric.json')

    threshold = 0.94
    if model_accuracy < threshold:
        raise ValueError(f'model accuracy {model_accuracy} is below threshold {threshold}')

    bucket = os.environ.get('MODEL_REPOSITORY_S3_BUCKET')
    if not bucket:
        raise ValueError('MODEL_REPOSITORY_S3_BUCKET is required for artifact upload')

    s3_client = S3()
    artifacts = ['model/model.pkl', 'model/metric.json', 'model/metadata.json']
    uploaded = []

    for artifact in artifacts:
        if not os.path.exists(artifact):
            raise FileNotFoundError(f'Expected artifact missing: {artifact}')
        object_key = f'models/{os.path.basename(artifact)}'
        s3_client.upload_file(artifact, bucket, object_key)
        uploaded.append(object_key)

    return {'promoted': True, 'accuracy': model_accuracy, 'uploaded': uploaded}



def version_model():
    run_id = airflow.utils.helpers.get_dagrun().run_id if airflow.utils.helpers.get_dagrun() else 'unknown'
    execution_date = airflow.utils.helpers.get_dagrun().execution_date.isoformat() if airflow.utils.helpers.get_dagrun() else 'unknown'
    timestamp = datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')
    version = f"v{timestamp}_{run_id}_{execution_date}"
    model_accuracy =  json.load(open(os.path.join('model', 'metric.json'))).get('accuracy')
    metadata = {
        'version': version,
        'model_type': 'logistic_regression',
        'dataset': 'breast_cancer',
        'accuracy': model_accuracy
    }
    file_path = os.path.join("model", "metadata.json")
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=4)
# Create task instances
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

promote_task = PythonOperator(
    task_id='promote_model',
    python_callable=promote_model,
    dag=dag,
)

# Set task dependencies
train_task >> evaluate_task >> promote_task