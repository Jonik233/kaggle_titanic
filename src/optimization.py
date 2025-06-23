import optuna
import numpy as np
import pandas as pd
from dotenv import dotenv_values

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from optuna.visualization.matplotlib import plot_param_importances

from xgboost import XGBClassifier

np.random.seed(42)


def ml_objective(trial, trial_inputs, trial_labels):

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 350, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 12, step=1),
        "min_child_weight": trial.suggest_int("min_child_weight", 13, 20, step=1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 2, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 2.5, step=0.1),
        "gamma": trial.suggest_float("gamma", 0.5, 1.5, step=0.1),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 1, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, step=0.1),
    }

    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    cv_predicts = np.empty(4)

    for idx, (train_idx, test_idx) in enumerate(cv.split(trial_inputs, trial_labels)):
        train_inputs, test_inputs = trial_inputs[train_idx], trial_inputs[test_idx]
        train_labels, test_labels = trial_labels[train_idx], trial_labels[test_idx]

        model = XGBClassifier(**param_grid, objective='binary:logistic', random_state=42, n_jobs=-1)

        model.fit(train_inputs, train_labels)
        preds = model.predict(test_inputs)
        score = f1_score(test_labels, preds)
        cv_predicts[idx] = score

        trial.report(score, idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(cv_predicts)


if __name__ == "__main__":
    env_config = dotenv_values("./.env")
    df = pd.read_csv(env_config["DATASET_PATH"])

    inputs, labels = preprocess_data(df=df, split=True)

    func = lambda trial: ml_objective(trial, inputs, labels)
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.HyperbandPruner()
    )

    try:
        study.optimize(func, n_trials=1000, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted. Saving progress...")

    study_df = study.trials_dataframe()
    study_df.to_csv("optuna_data/optuna_study.csv", index=False)

    plot_param_importances(study)
    plt.show()

    print("\n\n" + "-" * 50)
    print(study.best_value)
    print(study.best_params)
