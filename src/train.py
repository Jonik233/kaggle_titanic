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
from scores import get_scores

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
    model = XGBClassifier(max_depth=4,
                          learning_rate=0.8,
                          objective='binary:logistic',
                          n_estimators=340,
                          gamma=1.1,
                          reg_alpha=1.7,
                          reg_lambda=3.0,
                          colsample_bytree=0.8,
                          min_child_weight=4,
                          random_state=42,
                          n_jobs=-1)

    # Fetching metrics using cross validation
    train_metrics, val_metrics = get_scores(model, train_inputs, train_labels)

    print("Training" + "\n" + "-" * 30)
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nValidation" + "\n" + "-" * 30)
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

    # Running mlflow tracking in case mlflow tracking is True
    if mlflow_tracking:
        tags = {
            "Model": "XGBClassifier",
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
    train(plot=True, mlflow_tracking=True)
