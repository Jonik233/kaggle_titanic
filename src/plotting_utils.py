import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, train_data: np.ndarray, train_labels: np.ndarray, cv: int, save_plot: bool = False):
    train_sizes, train_acc, val_acc = learning_curve(model, train_data, train_labels, shuffle=True, cv=cv, scoring="accuracy")
    train_sizes, train_losses, val_losses = learning_curve(model, train_data, train_labels, shuffle=True, cv=cv, scoring="neg_log_loss")
    train_sizes, train_roc_auc, val_roc_auc = learning_curve(model, train_data, train_labels, shuffle=True, cv=cv, scoring="roc_auc")
    
    _, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].plot(train_sizes, np.mean(train_acc, axis=1), label="Train Score")
    axes[0].plot(train_sizes, np.mean(val_acc, axis=1), label="Validation Score")
    axes[0].set_xlabel("Training Samples")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Learning Curve - Accuracy")
    axes[0].legend()

    axes[1].plot(train_sizes, np.mean(-train_losses, axis=1), label="Train Loss")
    axes[1].plot(train_sizes, np.mean(-val_losses, axis=1), label="Validation Loss")
    axes[1].set_xlabel("Training Samples")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Learning Curve - Loss")
    axes[1].legend()
    
    axes[2].plot(train_sizes, np.mean(train_roc_auc, axis=1), label="Train ROC AUC")
    axes[2].plot(train_sizes, np.mean(val_roc_auc, axis=1), label="Validation ROC AUC")
    axes[2].set_xlabel("Training Samples")
    axes[2].set_ylabel("ROC_AUC")
    axes[2].set_title("Learning Curve - ROC_AUC")
    axes[2].legend()
    
    plt.tight_layout()

    if save_plot:
        plt.savefig("plot.png")

    plt.show()