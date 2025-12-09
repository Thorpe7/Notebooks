"""Evaluation utilities for classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    class_names,
    device: torch.device,
    results_dir: Path,
    prefix: str = "val",
) -> Dict[str, float]:
    """Run evaluation and save reports/plots to results_dir."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for xb, yb in dataloaders[prefix]:
            xb = xb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    results_dir.mkdir(parents=True, exist_ok=True)

    report_path = results_dir / f"{prefix}_classification_report.txt"
    with report_path.open("w") as f:
        for label, metrics in report.items():
            f.write(f"{label}: {metrics}
")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig_path = results_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(fig_path)
    plt.close(fig)

    return {
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
    }
