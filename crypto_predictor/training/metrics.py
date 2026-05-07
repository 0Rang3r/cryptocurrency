import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        logger.warning("AUC 计算失败，设为 0.0")
        auc = 0.0

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "AUC": round(auc, 4),
    }


def log_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    prefix_str = f"[{prefix}] " if prefix else ""
    msg = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    logger.info(f"{prefix_str}{msg}")