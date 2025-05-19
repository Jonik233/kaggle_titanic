import os

import mlflow
import numpy as np
from typing import Dict
from mlflow.models import infer_signature


def run_mlflow_tracking(
    model,
    model_name: str,
    inputs: np.ndarray,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    tags: Dict[str, str],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    plot_path: str
):

    # Setting mlflow tracking uri to project root directory
    mlflow.set_tracking_uri(tracking_uri)

    # Creating an experiment
    mlflow.set_experiment(experiment_name)

    # Starting a run
    with mlflow.start_run(run_name=run_name):

        # Creating tags
        mlflow.set_tags(tags)

        # Logging Train metrics
        mlflow.log_metrics(train_metrics)

        # Logging Val metrics
        mlflow.log_metrics(val_metrics)

        # Logging matplotlib metrics plot
        mlflow.log_artifact(plot_path)
        os.remove(plot_path)

        # Logging hyperparameters
        mlflow.log_params(model.get_params())

        # Creating model signature
        signature = infer_signature(inputs, model.predict(inputs))

        # Logging sklearn model
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path=model_name, signature=signature
        )
