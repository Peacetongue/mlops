import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL')


mlflow.run(
    uri='.',
    entry_point='neuro_third',
    env_manager='local',
    experiment_name='neuro_experiment',
    run_name='neuro_third',
    parameters={'epochs': 15, 'batch_size': 64},
)