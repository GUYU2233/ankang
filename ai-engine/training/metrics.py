"""评测指标：混淆矩阵、准确率、精确率、召回率、F1。"""
from __future__ import annotations

import numpy as np


def compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    precision = []
    recall = []
    f1 = []
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        precision.append(round(p, 4))
        recall.append(round(r, 4))
        f1.append(round(f, 4))

    total = int(cm.sum())
    accuracy = round(float(np.trace(cm)) / total, 4) if total else 0.0
    macro_f1 = round(float(np.mean(f1)), 4)
    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_metrics(metrics: dict, class_names: list[str]) -> str:
    cm = np.array(metrics["confusion_matrix"])
    lines = [
        f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}",
        "confusion_matrix (rows=true, cols=pred):",
    ]
    header = " " * 12 + "".join(f"{n:>10s}" for n in class_names)
    lines.append(header)
    for i, name in enumerate(class_names):
        lines.append(f"{name:>10s}  " + "".join(f"{int(cm[i, j]):>10d}" for j in range(len(class_names))))
    lines.append("per-class  precision  recall  f1")
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:>10s}  {metrics['precision'][i]:>8.4f}  {metrics['recall'][i]:>6.4f}  {metrics['f1'][i]:>6.4f}"
        )
    return "\n".join(lines)
