import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

def get_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "classification_report": classification_report(y_true, y_pred)
    }

def plot_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))

    cm = np.array([[TN, FP],
                   [FN, TP]])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    # Axis labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    # Annotate each cell
    labels = [["TN", "FP"],
              ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center")

    plt.title("Confusion Matrix")
    plt.tight_layout()

    return fig


def plot_roc_curve(y_true, y_prob):
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # Compute AUC
    roc_auc = roc_auc_score(y_true, y_prob)

    # Create plot
    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--', label="Random Classifier")

    # Labels and title
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve")

    ax.legend(loc="lower right")
    plt.tight_layout()

    return fig


def plot_threshold_tuning(y_true, y_prob):

    thresholds = np.arange(0.01, 1.00, 0.01)

    precisions = []
    recalls = []
    f1_scores = []

    for t in thresholds:
        y_pred = (np.array(y_prob) >= t).astype(int)

        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    # Best threshold (max F1)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Plot
    fig, ax = plt.subplots()

    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1_scores, label="F1 Score")

    # Mark best threshold
    ax.scatter(best_threshold, best_f1)
    ax.axvline(best_threshold, linestyle='--', label=f"Best Threshold = {best_threshold:.2f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning")

    ax.legend()
    plt.tight_layout()

    return fig, best_threshold