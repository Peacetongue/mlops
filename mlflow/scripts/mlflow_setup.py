import mlflow

def configure_mlflow(experiment_name="GliomaSegmentation"):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)