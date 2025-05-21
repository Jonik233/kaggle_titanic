import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import dotenv_values
from config import ENV_FILE_PATH

from preprocessing import preprocess_data
from plotting_utils import plot_learning_curves
from mlflow_tracking import run_mlflow_tracking
from scores import get_train_scores, get_val_scores

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

np.random.seed(42)


def train(plot=False, mlflow_tracking=False):

    # Loading data
    env_config = dotenv_values(ENV_FILE_PATH)
    df = pd.read_csv(env_config["DATASET_PATH"])

    # Preprocessing data
    train_inputs, train_labels = preprocess_data(df=df, split=True)

    # Model initialization
    model = RandomForestClassifier(n_estimators=140,
                                   max_depth=5,
                                   min_samples_split=17,
                                   min_samples_leaf=11,
                                   max_leaf_nodes=24,
                                   criterion="gini",
                                   max_features=0.8,
                                   random_state=42,
                                   n_jobs=-1)

    # Fetching validation metrics using cross validation
    val_metrics = get_val_scores(model, train_inputs, train_labels)

    print("Validation" + "\n" + "-" * 30)
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")

    # Plotting and saving metrics if plot is True
    if plot:
        save_plot = True if mlflow_tracking else False
        plot_learning_curves(model, train_inputs, train_labels, 4, save_plot)

    # Training model and loading it into the dump
    model.fit(train_inputs, train_labels)
    model_path = Path(env_config["MODEL_DUMP_PATH"])

    if model_path.exists():
        os.remove(model_path)

    joblib.dump(model, env_config["MODEL_DUMP_PATH"])

    # Fetching train metrics
    train_metrics = get_train_scores(model, train_inputs, train_labels)

    print("\nTraining" + "\n" + "-" * 30)
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")

    # Running mlflow tracking in case mlflow tracking is True
    if mlflow_tracking:
        tags = {
            "Model": "RandomForestClassifier",
            "Branch": "dev1",
        }
        run_mlflow_tracking(
            model=model,
            model_name=tags["Model"],
            inputs=train_inputs,
            tracking_uri=env_config["MLFLOW_RUNS_PATH"],
            experiment_name="Preprocessing V1",
            run_name=tags["Model"],
            tags=tags,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            plot_path="plot.png"
        )


if __name__ == "__main__":
    train(plot=True, mlflow_tracking=False)
