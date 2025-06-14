import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import cross_validate, KFold


def get_scores(
    model,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    n_splits: int = 4,
    shuffle: bool = True,
    random_state: int = 42,
    scores: Tuple[str] = ("f1", "accuracy", "neg_log_loss", "roc_auc"),
) -> Tuple[Dict[str, float], Dict[str, float]]:

    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = cross_validate(
        model,
        train_inputs,
        train_labels,
        scoring=scores,
        cv=cv,
        return_train_score=True,
    )

    train_metrics = {
        "Train F1 score": scores["train_f1"].mean(),
        "Train Accuracy": scores["train_accuracy"].mean(),
        "Train ROC AUC": scores["train_roc_auc"].mean(),
        "Train Loss": -scores["train_neg_log_loss"].mean(),
    }

    val_metrics = {
        "Val F1 score": scores["test_f1"].mean(),
        "Val Accuracy": scores["test_accuracy"].mean(),
        "Val ROC AUC": scores["test_roc_auc"].mean(),
        "Val Loss": -scores["test_neg_log_loss"].mean(),
    }

    return train_metrics, val_metrics
