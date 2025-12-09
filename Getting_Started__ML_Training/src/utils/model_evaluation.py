"""
Model evaluation visualizations for classification tasks.

Provides functions for:
- ROC curves (per-class and macro-averaged)
- Precision-Recall curves
- Per-class metrics bar charts
- Prediction confidence distributions
- Grad-CAM heatmaps for model interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot ROC curves for each class and macro-averaged ROC.

    Args:
        y_true: Ground truth labels (n_samples,)
        y_probs: Prediction probabilities (n_samples, n_classes)
        class_names: List of class names
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    n_classes = len(class_names)

    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot macro-average ROC
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro-average (AUC = {roc_auc['macro']:.3f})",
        color="navy",
        linestyle="--",
        linewidth=2,
    )

    # Plot per-class ROC curves
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
            linewidth=1.5,
        )

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot Precision-Recall curves for each class.

    Args:
        y_true: Ground truth labels (n_samples,)
        y_probs: Prediction probabilities (n_samples, n_classes)
        class_names: List of class names
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    n_classes = len(class_names)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Compute PR curve for each class
    precision = {}
    recall = {}
    avg_precision = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_probs[:, i]
        )
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(
            recall[i],
            precision[i],
            color=color,
            label=f"{class_names[i]} (AP = {avg_precision[i]:.3f})",
            linewidth=1.5,
        )

    # Compute and plot macro-average
    macro_ap = np.mean(list(avg_precision.values()))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves (Macro AP = {macro_ap:.3f})")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_class_metrics_bar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot per-class F1, Precision, and Recall as grouped bar chart.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    n_classes = len(class_names)

    # Compute per-class metrics
    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Ensure we have scores for all classes
    if len(f1_scores) < n_classes:
        f1_scores = np.zeros(n_classes)
        precisions = np.zeros(n_classes)
        recalls = np.zeros(n_classes)
        for i, cls in enumerate(sorted(set(y_true) | set(y_pred))):
            if cls < n_classes:
                mask_true = y_true == cls
                mask_pred = y_pred == cls
                tp = np.sum(mask_true & mask_pred)
                fp = np.sum(~mask_true & mask_pred)
                fn = np.sum(mask_true & ~mask_pred)
                precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
                recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_scores[cls] = 2 * precisions[cls] * recalls[cls] / (precisions[cls] + recalls[cls]) if (precisions[cls] + recalls[cls]) > 0 else 0

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_classes)
    width = 0.25

    bars1 = ax.bar(x - width, f1_scores, width, label="F1-Score", color="#2ecc71")
    bars2 = ax.bar(x, precisions, width, label="Precision", color="#3498db")
    bars3 = ax.bar(x + width, recalls, width, label="Recall", color="#e74c3c")

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot distribution of prediction confidence for correct vs incorrect predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (n_samples, n_classes)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Get confidence (max probability) for each prediction
    confidences = np.max(y_probs, axis=1)

    # Split by correct/incorrect
    correct_mask = y_true == y_pred
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram of confidences
    ax1 = axes[0]
    bins = np.linspace(0, 1, 21)

    ax1.hist(
        correct_conf,
        bins=bins,
        alpha=0.7,
        label=f"Correct (n={len(correct_conf)})",
        color="#2ecc71",
        edgecolor="black",
    )
    ax1.hist(
        incorrect_conf,
        bins=bins,
        alpha=0.7,
        label=f"Incorrect (n={len(incorrect_conf)})",
        color="#e74c3c",
        edgecolor="black",
    )

    ax1.set_xlabel("Prediction Confidence")
    ax1.set_ylabel("Count")
    ax1.set_title("Confidence Distribution: Correct vs Incorrect")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot comparison
    ax2 = axes[1]
    data_to_plot = [correct_conf, incorrect_conf]
    bp = ax2.boxplot(
        data_to_plot,
        labels=["Correct", "Incorrect"],
        patch_artist=True,
    )

    colors = ["#2ecc71", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel("Prediction Confidence")
    ax2.set_title("Confidence Comparison")
    ax2.grid(True, alpha=0.3)

    # Add mean annotations
    for i, (data, color) in enumerate(zip(data_to_plot, colors)):
        if len(data) > 0:
            mean_val = np.mean(data)
            ax2.annotate(
                f"Mean: {mean_val:.3f}",
                xy=(i + 1, mean_val),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=9,
            )

    plt.tight_layout()
    return fig


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN interpretability.

    Generates heatmaps showing which regions of the input image
    the model focuses on for making predictions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize GradCAM.

        Args:
            model: The CNN model
            target_layer: The convolutional layer to compute CAM from
                         (typically the last conv layer before pooling)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Class to generate CAM for. If None, uses predicted class.

        Returns:
            CAM heatmap as numpy array (H, W)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def plot_gradcam_grid(
    model: torch.nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
    target_layer: torch.nn.Module,
    n_samples: int = 8,
    figsize: Tuple[int, int] = (16, 8),
    device: str = "cuda",
) -> plt.Figure:
    """
    Plot a grid of images with their Grad-CAM heatmaps.

    Args:
        model: The trained model
        images: Validation images (n_samples, C, H, W)
        labels: Ground truth labels
        preds: Predicted labels
        class_names: List of class names
        target_layer: Target layer for Grad-CAM
        n_samples: Number of samples to display
        figsize: Figure size
        device: Device to run model on

    Returns:
        matplotlib Figure object
    """
    import cv2

    gradcam = GradCAM(model, target_layer)

    # Select diverse samples (mix of correct and incorrect)
    correct_mask = labels == preds
    incorrect_indices = np.where(~correct_mask)[0]
    correct_indices = np.where(correct_mask)[0]

    # Try to get half correct, half incorrect
    n_incorrect = min(n_samples // 2, len(incorrect_indices))
    n_correct = min(n_samples - n_incorrect, len(correct_indices))

    selected_indices = []
    if n_incorrect > 0:
        selected_indices.extend(
            np.random.choice(incorrect_indices, n_incorrect, replace=False)
        )
    if n_correct > 0:
        selected_indices.extend(
            np.random.choice(correct_indices, n_correct, replace=False)
        )

    n_actual = len(selected_indices)
    if n_actual == 0:
        print("No samples available for Grad-CAM visualization")
        return None

    # Create figure
    fig, axes = plt.subplots(2, n_actual, figsize=figsize)
    if n_actual == 1:
        axes = axes.reshape(2, 1)

    for idx, sample_idx in enumerate(selected_indices):
        img = images[sample_idx]
        true_label = labels[sample_idx]
        pred_label = preds[sample_idx]

        # Prepare input tensor
        input_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)

        # Generate CAM
        cam = gradcam.generate_cam(input_tensor, target_class=pred_label)

        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (img.shape[2], img.shape[1]))

        # Prepare image for display (denormalize)
        img_display = img.transpose(1, 2, 0)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255

        # Blend with original image
        overlay = 0.6 * img_display + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)

        # Plot original image
        axes[0, idx].imshow(img_display)
        axes[0, idx].axis("off")

        # Color title based on correct/incorrect
        is_correct = true_label == pred_label
        color = "green" if is_correct else "red"
        axes[0, idx].set_title(
            f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
            fontsize=9,
            color=color,
        )

        # Plot Grad-CAM overlay
        axes[1, idx].imshow(overlay)
        axes[1, idx].axis("off")
        axes[1, idx].set_title("Grad-CAM", fontsize=9)

    plt.suptitle("Grad-CAM Visualizations (Top: Original, Bottom: Attention Map)", fontsize=12)
    plt.tight_layout()
    return fig


def get_efficientnet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Get the appropriate target layer for Grad-CAM from an EfficientNet model.

    Args:
        model: EfficientNet model

    Returns:
        The last convolutional layer suitable for Grad-CAM
    """
    # For EfficientNet, the last conv layer is in features[-1]
    # Access through the backbone
    if hasattr(model, "backbone"):
        return model.backbone.features[-1]
    elif hasattr(model, "features"):
        return model.features[-1]
    else:
        raise ValueError("Could not find features layer in model")
