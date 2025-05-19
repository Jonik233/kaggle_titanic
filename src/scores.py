import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import f1_score, accuracy_score, log_loss, roc_auc_score


def get_val_scores(
    model,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    n_splits: int = 4,
    shuffle: bool = True,
    random_state: int = 42,
    scores: Tuple[str] = ("f1", "accuracy", "neg_log_loss", "roc_auc"),
) -> Dict[str, float]:

    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = cross_validate(model, train_inputs, train_labels, scoring=scores, cv=cv)

    f_score = scores["test_f1"].mean()
    accuracy = scores["test_accuracy"].mean()
    roc_auc = scores["test_roc_auc"].mean()
    loss = -scores["test_neg_log_loss"].mean()

    val_metrics = {
        "Val F1 score": f_score,
        "Val Accuracy": accuracy,
        "Val ROC AUC": roc_auc,
        "Val Loss": loss,
    }

    return val_metrics


def get_train_scores(
    model, train_inputs: np.ndarray, train_labels: np.ndarray
) -> Dict[str, float]:

    predicted = model.predict(train_inputs)
    probs = model.predict_proba(train_inputs)
    predicted_scores = model.predict_proba(train_inputs)[:, 1]

    f_score = f1_score(train_labels, predicted)
    accuracy = accuracy_score(train_labels, predicted)
    roc_auc = roc_auc_score(train_labels, predicted_scores)
    loss = log_loss(train_labels, probs)

    train_metrics = {
        "Train F1 score": f_score,
        "Train Accuracy": accuracy,
        "Train ROC AUC": roc_auc,
        "Train Loss": loss,
    }

    return train_metrics
