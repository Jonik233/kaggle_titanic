import optuna
import numpy as np
import pandas as pd
from dotenv import dotenv_values

# from models import Model
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# from train_val import train, val_test
from transforms import TransformerPipeline
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_param_importances

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

np.random.seed(42)


def ml_objective(trial, trial_inputs, trial_labels):

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=10),
        "max_depth": trial.suggest_int("max_depth", 1, 3, step=1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 3, step=1),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 1, step=0.1),
        "gamma": trial.suggest_float("gamma", 0.1, 1, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1, step=0.1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, step=0.1)
    }

    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    cv_predicts = np.empty(4)

    for idx, (train_idx, test_idx) in enumerate(cv.split(trial_inputs, trial_labels)):
        train_inputs, test_inputs = trial_inputs[train_idx], trial_inputs[test_idx]
        train_labels, test_labels = trial_labels[train_idx], trial_labels[test_idx]

        model = XGBClassifier(**param_grid, objective='binary:logistic', random_seed=42, n_jobs=-1)

        model.fit(train_inputs, train_labels)
        preds = model.predict(test_inputs)
        score = f1_score(test_labels, preds)
        cv_predicts[idx] = score

        trial.report(score, idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(cv_predicts)


# def define_model(trial):
#     n_layers = trial.suggest_int("n_layers", 1, 10)
#
#     layers = []
#     in_features = 10
#     out_features = trial.suggest_int(f"out_features", 10, 250, step=10)
#     for i in range(n_layers):
#         layers.append(nn.Linear(in_features, out_features))
#         layers.append(nn.BatchNorm1d(out_features))
#         layers.append(nn.Tanh())
#         in_features = out_features
#
#     layers.append(nn.Linear(in_features, 2))
#     return nn.Sequential(*layers)
#
#
# def dp_objective(trial, device, epochs, train_dataloader, val_dataloader):
#     model = define_model(trial).to(device=device)
#     loss_fn = nn.CrossEntropyLoss()
#
#     lr = trial.suggest_float("lr", 8e-4, 8e-2)
#     optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop"])
#     optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
#
#     best_val_fscore = -1
#
#     for _ in range(epochs):
#         train_loss, train_acc, train_recall, train_precision, train_fscore = train(model, device, optimizer, loss_fn, train_dataloader)
#         val_loss, val_acc, val_recall, val_precision, val_fscore = val_test(model, device, val_dataloader, loss_fn)
#
#         if val_fscore > best_val_fscore:
#             best_val_fscore = val_fscore
#
#     return best_val_fscore


if __name__ == "__main__":
    env_config = dotenv_values("./.env")
    df = pd.read_csv(env_config["DATASET_PATH"])

    pipeline = TransformerPipeline()
    inputs, labels = pipeline.transform(df, split_attr=True)

    func = lambda trial: ml_objective(trial, inputs, labels)
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.HyperbandPruner()
    )

    try:
        study.optimize(func, n_trials=800, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted. Saving progress...")

    study_df = study.trials_dataframe()
    study_df.to_csv("optuna_data/optuna_study.csv", index=False)

    plot_param_importances(study)
    plt.show()

    print("\n\n" + "-" * 50)
    print(study.best_value)
    print(study.best_params)
