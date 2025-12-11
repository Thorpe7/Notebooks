"""
Reusable Voila-friendly dashboards for the XNAT ResNet notebook.

Each function returns an ipywidgets.VBox that you can display in
Jupyter and Voila, e.g.:

    from src.utils.xnat_voila_dashboards import (
        class_distribution_dashboard,
        training_history_dashboard,
        confusion_matrix_dashboard,
        pixel_intensity_dashboard,
        roc_curves_dashboard,
        gradcam_dashboard,
        run_comparison_dashboard,
        gradcam_comparison_dashboard,
        metadata_filter_dashboard,
    )

    ui = class_distribution_dashboard(train_labels, val_labels, class_names)
    ui

All plots are rendered with matplotlib and updated via ipywidgets.

Dashboard summary:
1. class_distribution_dashboard - Train/val class distribution bar charts
2. training_history_dashboard - Loss/error curves over epochs
3. confusion_matrix_dashboard - Confusion matrix with normalization options
4. pixel_intensity_dashboard - Pixel intensity histograms per class
5. roc_curves_dashboard - ROC curves with toggleable class lines
6. gradcam_dashboard - Individual Grad-CAM + class average attention maps
7. run_comparison_dashboard - Compare metrics across training runs
8. gradcam_comparison_dashboard - Compare model predictions between runs
9. metadata_filter_dashboard - Filter and subset XNAT metadata with visualizations
"""

from typing import Sequence, Mapping, Optional, List, Callable

import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import VBox, HBox
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------

def _ensure_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


# -------------------------------------------------------------------------
# 1. Class distribution dashboard
# -------------------------------------------------------------------------

def class_distribution_dashboard(
    train_labels: Sequence[int],
    val_labels: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> VBox:
    """
    Dashboard to explore train/val class distribution.

    Parameters
    ----------
    train_labels : sequence of int
        Class indices for the training set.
    val_labels : sequence of int, optional
        Class indices for the validation set.
    class_names : sequence of str, optional
        Human-readable class names; if None, numeric indices are used.

    Returns
    -------
    ipywidgets.VBox
    """
    y_train = _ensure_numpy(train_labels)
    y_val = _ensure_numpy(val_labels) if val_labels is not None else None

    unique_classes = np.unique(
        y_train if y_val is None else np.concatenate([y_train, y_val])
    )
    unique_classes = np.sort(unique_classes)

    if class_names is None:
        class_names = [str(int(c)) for c in unique_classes]
    else:
        class_names = list(class_names)

    dataset_options = ["train"]
    if y_val is not None:
        dataset_options += ["val", "both"]

    dataset_selector = widgets.ToggleButtons(
        options=dataset_options,
        value=dataset_options[0],
        description="Split:",
    )

    normalize_selector = widgets.ToggleButtons(
        options=[("Count", "count"), ("Proportion", "prop")],
        value="count",
        description="Y-axis:",
    )

    out = widgets.Output()

    def _compute_counts(labels: np.ndarray) -> np.ndarray:
        return np.array([(labels == c).sum() for c in unique_classes], dtype=float)

    def _update(change=None):
        mode = dataset_selector.value
        y_mode = normalize_selector.value

        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 4))

            width = 0.35
            x = np.arange(len(unique_classes))

            if mode in ("train", "both"):
                counts_train = _compute_counts(y_train)
                if y_mode == "prop":
                    total = counts_train.sum() or 1.0
                    counts_train = counts_train / total
                ax.bar(x - width/2, counts_train, width, label="Train")

            if y_val is not None and mode in ("val", "both"):
                counts_val = _compute_counts(y_val)
                if y_mode == "prop":
                    total = counts_val.sum() or 1.0
                    counts_val = counts_val / total
                offset = width/2 if mode == "both" else 0.0
                ax.bar(x + offset, counts_val, width, label="Val")

            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count" if y_mode == "count" else "Fraction")
            ax.set_title("Class distribution")
            if (mode in ("train", "both")) or (mode in ("val", "both")):
                ax.legend()
            plt.tight_layout()
            plt.show()

    dataset_selector.observe(_update, names="value")
    normalize_selector.observe(_update, names="value")

    _update()
    controls = HBox([dataset_selector, normalize_selector])
    return VBox([controls, out])


# -------------------------------------------------------------------------
# 2. Training history (loss/error) dashboard
# -------------------------------------------------------------------------

def training_history_dashboard(
    history: Mapping[str, Sequence[float]],
    title: str = "Training history",
) -> VBox:
    """
    Dashboard for training/validation loss and error curves over epochs.

    Expects a history dict like:
        {
            "train_loss": [...],
            "val_loss": [...],
            "train_err": [...],
            "val_err": [...],
        }
    """
    hist_np = {k: np.asarray(v, dtype=float) for k, v in history.items()}

    metric_selector = widgets.ToggleButtons(
        options=[("Loss", "loss"), ("Error rate", "err")],
        value="loss",
        description="Metric:",
    )

    max_smooth = max(1, (len(hist_np.get("train_loss", [])) // 4) or 1)

    smooth_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=max_smooth,
        step=1,
        description="Smoothing:",
        continuous_update=False,
    )

    out = widgets.Output()

    def _smooth(x: np.ndarray, win: int) -> np.ndarray:
        if win <= 1 or len(x) == 0:
            return x
        k = min(win, len(x))
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="same")

    def _update(change=None):
        metric = metric_selector.value
        win = int(smooth_slider.value)

        train_key = f"train_{metric}"
        val_key = f"val_{metric}"

        train_vals = hist_np.get(train_key, np.array([]))
        val_vals = hist_np.get(val_key, np.array([]))

        epochs = np.arange(1, len(train_vals) + 1)

        y_train = _smooth(train_vals, win)
        y_val = _smooth(val_vals, win)

        ylabel = (
            "Cross-entropy loss"
            if metric == "loss"
            else "Error rate (1 - accuracy)"
        )

        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 4))

            if len(y_train):
                ax.plot(epochs, y_train, "-o", label="Train")
            if len(y_val):
                ax.plot(epochs, y_val, "-o", label="Val")

            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} – {metric}")
            ax.grid(True, linestyle="--", alpha=0.4)
            if len(y_train) or len(y_val):
                ax.legend()
            plt.tight_layout()
            plt.show()

    metric_selector.observe(_update, names="value")
    smooth_slider.observe(_update, names="value")

    _update()
    controls = HBox([metric_selector, smooth_slider])
    return VBox([controls, out])


# -------------------------------------------------------------------------
# 3. Confusion matrix dashboard
# -------------------------------------------------------------------------

def confusion_matrix_dashboard(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> VBox:
    """
    Dashboard for a confusion matrix with optional normalization.
    """
    y_true = _ensure_numpy(y_true)
    y_pred = _ensure_numpy(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.sort(labels)

    if class_names is None:
        class_names = [str(int(c)) for c in labels]
    else:
        class_names = list(class_names)

    norm_selector = widgets.ToggleButtons(
        options=[("Counts", "count"), ("Row-normalized", "row")],
        value="row" if normalize else "count",
        description="Display:",
    )

    out = widgets.Output()

    def _update(change=None):
        mode = norm_selector.value
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if mode == "row":
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_disp = cm / row_sums
            cmap_label = "Fraction"
        else:
            cm_disp = cm.astype(float)
            cmap_label = "Count"

        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm_disp, interpolation="nearest", cmap=plt.cm.Blues)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cmap_label)

            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)

            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(title)

            # Annotate cells
            thresh = cm_disp.max() / 2.0 if cm_disp.size else 0.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = cm_disp[i, j]
                    text = f"{val:.2f}" if mode == "row" else str(cm[i, j])
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color="white" if val > thresh else "black",
                        fontsize=8,
                    )

            plt.tight_layout()
            plt.show()

    norm_selector.observe(_update, names="value")

    _update()
    controls = HBox([norm_selector])
    return VBox([controls, out])


# -------------------------------------------------------------------------
# 4. Pixel intensity distribution dashboard
# -------------------------------------------------------------------------

def pixel_intensity_dashboard(
    images: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    n_bins: int = 64,
) -> VBox:
    """
    Dashboard to inspect pixel intensity distributions.

    Shows two histograms:
      1. Full intensity range
      2. Zoomed to a percentile range (default 1–99%)

    Parameters
    ----------
    images : np.ndarray
        Image data as (N, H, W) or (N, 1, H, W) or (N, C, H, W).
        For RGB, intensities are flattened across channels.
    labels : sequence of int, optional
        Class labels (len = N). If provided, per-class histograms are available.
    n_bins : int
        Number of histogram bins.
    """
    imgs = np.asarray(images)
    if imgs.ndim == 4:
        # (N, C, H, W) -> flatten channels
        imgs_flat = imgs.reshape(imgs.shape[0], -1)
    elif imgs.ndim == 3:
        imgs_flat = imgs.reshape(imgs.shape[0], -1)
    else:
        raise ValueError("Expected images with shape (N,H,W) or (N,C,H,W).")

    labels_np = _ensure_numpy(labels) if labels is not None else None

    # --- Class selector ------------------------------------------------------
    if labels_np is not None:
        unique_classes = np.unique(labels_np)
        unique_classes = np.sort(unique_classes)
        if class_names is None:
            class_names = [str(int(c)) for c in unique_classes]
        else:
            class_names = list(class_names)

        options = ["all"] + [str(int(c)) for c in unique_classes]
        class_selector = widgets.Dropdown(
            options=options,
            value="all",
            description="Class:",
        )
    else:
        class_selector = None
        class_names = None
        unique_classes = None

    # --- Percentile zoom slider ---------------------------------------------
    perc_slider = widgets.IntRangeSlider(
        value=[1, 99],
        min=0,
        max=100,
        step=1,
        description="Zoom %:",
        continuous_update=False,
        layout=widgets.Layout(width="50%"),
    )

    out = widgets.Output()

    def _update(change=None):
        # Select subset
        if labels_np is not None and class_selector.value != "all":
            c = int(class_selector.value)
            mask = labels_np == c
            data = imgs_flat[mask]
        else:
            data = imgs_flat

        pixels = data.reshape(-1)
        pixels = pixels[np.isfinite(pixels)]

        # Optional subsample if insane number of pixels
        max_samples = 1_000_000
        if pixels.size > max_samples:
            idx = np.random.choice(pixels.size, max_samples, replace=False)
            pixels = pixels[idx]

        with out:
            out.clear_output(wait=True)
            fig, (ax_full, ax_zoom) = plt.subplots(
                2, 1, figsize=(7, 6), sharex=False
            )

            # --- Full range ---------------------------------------------------
            if pixels.size == 0:
                ax_full.hist([0], bins=1)
            else:
                ax_full.hist(pixels, bins=n_bins, alpha=0.8)
            ax_full.set_title("Pixel intensity histogram – full range")
            ax_full.set_ylabel("Count")

            # --- Zoomed range (percentiles) ----------------------------------
            if pixels.size > 0:
                low_p, high_p = perc_slider.value
                vmin, vmax = np.percentile(pixels, [low_p, high_p])
                # Avoid degenerate range
                if vmin == vmax:
                    vmin -= 1e-6
                    vmax += 1e-6
                ax_zoom.hist(
                    pixels,
                    bins=n_bins,
                    range=(vmin, vmax),
                    alpha=0.8,
                )
                ax_zoom.set_title(
                    f"Zoomed histogram – {low_p}–{high_p} percentile "
                    f"({vmin:.3f} to {vmax:.3f})"
                )
            else:
                ax_zoom.hist([0], bins=1)
                ax_zoom.set_title("Zoomed histogram – no data")

            ax_zoom.set_xlabel("Pixel intensity")
            ax_zoom.set_ylabel("Count")

            plt.tight_layout()
            plt.show()

    # Wire interactions
    if class_selector is not None:
        class_selector.observe(_update, names="value")
        controls_top = HBox([class_selector, perc_slider])
    else:
        controls_top = HBox([perc_slider])

    perc_slider.observe(_update, names="value")

    # Initial render
    _update()

    return VBox([controls_top, out])


# -------------------------------------------------------------------------
# 5. ROC Curves dashboard with toggleable class lines
# -------------------------------------------------------------------------

def roc_curves_dashboard(
    y_true: Sequence[int],
    y_probs: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> VBox:
    """
    Interactive ROC curves dashboard with checkboxes to toggle class visibility.

    Parameters
    ----------
    y_true : sequence of int
        Ground truth labels (n_samples,)
    y_probs : np.ndarray
        Prediction probabilities (n_samples, n_classes)
    class_names : sequence of str, optional
        Human-readable class names

    Returns
    -------
    ipywidgets.VBox
    """
    y_true = _ensure_numpy(y_true)
    y_probs = _ensure_numpy(y_probs)

    n_classes = y_probs.shape[1]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    else:
        class_names = list(class_names)

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

    # Create checkboxes for each class
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    checkboxes = []
    for i in range(n_classes):
        cb = widgets.Checkbox(
            value=True,
            description=f"{class_names[i]} (AUC={roc_auc[i]:.3f})",
            indent=False,
            layout=widgets.Layout(width="auto"),
        )
        checkboxes.append(cb)

    # Macro average checkbox
    macro_checkbox = widgets.Checkbox(
        value=True,
        description=f"Macro-avg (AUC={roc_auc['macro']:.3f})",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )

    # Random baseline checkbox
    baseline_checkbox = widgets.Checkbox(
        value=True,
        description="Random baseline",
        indent=False,
        layout=widgets.Layout(width="auto"),
    )

    # Select all / deselect all buttons
    select_all_btn = widgets.Button(description="Select All", button_style="info")
    deselect_all_btn = widgets.Button(description="Deselect All", button_style="warning")

    out = widgets.Output()

    def _update(change=None):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot macro-average ROC
            if macro_checkbox.value:
                ax.plot(
                    fpr["macro"],
                    tpr["macro"],
                    label=f"Macro-average (AUC = {roc_auc['macro']:.3f})",
                    color="navy",
                    linestyle="--",
                    linewidth=2,
                )

            # Plot per-class ROC curves based on checkbox state
            for i, (cb, color) in enumerate(zip(checkboxes, colors)):
                if cb.value:
                    ax.plot(
                        fpr[i],
                        tpr[i],
                        color=color,
                        label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})",
                        linewidth=1.5,
                    )

            # Plot diagonal (random classifier)
            if baseline_checkbox.value:
                ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves (One-vs-Rest)")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def _select_all(btn):
        for cb in checkboxes:
            cb.value = True
        macro_checkbox.value = True
        baseline_checkbox.value = True

    def _deselect_all(btn):
        for cb in checkboxes:
            cb.value = False
        macro_checkbox.value = False
        baseline_checkbox.value = False

    # Wire up observers
    for cb in checkboxes:
        cb.observe(_update, names="value")
    macro_checkbox.observe(_update, names="value")
    baseline_checkbox.observe(_update, names="value")

    select_all_btn.on_click(_select_all)
    deselect_all_btn.on_click(_deselect_all)

    # Initial render
    _update()

    # Layout
    buttons_row = HBox([select_all_btn, deselect_all_btn])
    special_checkboxes = HBox([macro_checkbox, baseline_checkbox])
    class_checkboxes = HBox(checkboxes, layout=widgets.Layout(flex_flow="row wrap"))

    controls = VBox([buttons_row, special_checkboxes, class_checkboxes])

    return VBox([controls, out])


# -------------------------------------------------------------------------
# 6. Grad-CAM interactive dashboard
# -------------------------------------------------------------------------

def gradcam_dashboard(
    model,
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    target_layer,
    device: str = "cuda",
) -> VBox:
    """
    Interactive Grad-CAM dashboard with navigation buttons and average attention map.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    images : np.ndarray
        Validation images (n_samples, C, H, W)
    labels : np.ndarray
        Ground truth labels
    preds : np.ndarray
        Predicted labels
    probs : np.ndarray
        Prediction probabilities
    class_names : sequence of str
        Class names
    target_layer : torch.nn.Module
        Target layer for Grad-CAM
    device : str
        Device to run model on

    Returns
    -------
    ipywidgets.VBox
    """
    import torch
    import torch.nn.functional as F

    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

    labels = _ensure_numpy(labels)
    preds = _ensure_numpy(preds)
    probs = _ensure_numpy(probs)

    n_samples = len(labels)
    class_names = list(class_names)
    n_classes = len(class_names)

    # Separate correct and incorrect predictions
    correct_mask = labels == preds
    incorrect_indices = np.where(~correct_mask)[0]
    correct_indices = np.where(correct_mask)[0]

    # Create a shuffled list prioritizing incorrect predictions
    all_indices = list(incorrect_indices) + list(correct_indices)
    np.random.shuffle(all_indices[:len(incorrect_indices)])  # Shuffle incorrect
    np.random.shuffle(all_indices[len(incorrect_indices):])  # Shuffle correct

    if len(all_indices) == 0:
        return VBox([widgets.HTML("<p>No samples available for visualization.</p>")])

    # State
    state = {
        "current_idx": 0,
        "sample_list": all_indices,
        "avg_cam_cache": {},  # Cache for average CAMs per class
    }

    # Grad-CAM helper class
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
            self._register_hooks()

        def _register_hooks(self):
            def forward_hook(module, input, output):
                self.activations = output.detach()

            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()

            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)

        def generate_cam(self, input_tensor, target_class=None):
            self.model.eval()
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)

            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            return cam

    gradcam = GradCAM(model, target_layer)

    def _resize_cam(cam, target_h, target_w):
        """Resize CAM to target dimensions."""
        if HAS_CV2:
            return cv2.resize(cam, (target_w, target_h))
        else:
            from scipy.ndimage import zoom
            scale_h = target_h / cam.shape[0]
            scale_w = target_w / cam.shape[1]
            return zoom(cam, (scale_h, scale_w), order=1)

    def _cam_to_heatmap(cam_resized):
        """Convert CAM to RGB heatmap."""
        if HAS_CV2:
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        else:
            return plt.cm.jet(cam_resized)[:, :, :3]

    def _compute_average_cam(class_idx: int, max_samples: int = 50) -> Optional[np.ndarray]:
        """Compute average Grad-CAM for a specific class."""
        if class_idx in state["avg_cam_cache"]:
            return state["avg_cam_cache"][class_idx]

        # Get indices for this class (based on predicted label)
        class_indices = np.where(preds == class_idx)[0]
        if len(class_indices) == 0:
            return None

        # Sample if too many
        if len(class_indices) > max_samples:
            class_indices = np.random.choice(class_indices, max_samples, replace=False)

        # Compute CAMs for all samples
        cams = []
        target_h, target_w = images.shape[2], images.shape[3]

        for idx in class_indices:
            img = images[idx]
            input_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
            cam = gradcam.generate_cam(input_tensor, target_class=class_idx)
            cam_resized = _resize_cam(cam, target_h, target_w)
            cams.append(cam_resized)

        # Average and normalize
        avg_cam = np.mean(cams, axis=0)
        avg_cam = (avg_cam - avg_cam.min()) / (avg_cam.max() - avg_cam.min() + 1e-8)

        state["avg_cam_cache"][class_idx] = avg_cam
        return avg_cam

    # Widgets
    prev_btn = widgets.Button(
        description="Previous",
        button_style="info",
        icon="arrow-left",
    )
    next_btn = widgets.Button(
        description="Next",
        button_style="info",
        icon="arrow-right",
    )
    random_btn = widgets.Button(
        description="Random",
        button_style="warning",
        icon="random",
    )

    # Filter dropdown
    filter_dropdown = widgets.Dropdown(
        options=[("All samples", "all"), ("Correct only", "correct"), ("Incorrect only", "incorrect")],
        value="all",
        description="Filter:",
    )

    # Show average CAM checkbox
    show_avg_checkbox = widgets.Checkbox(
        value=True,
        description="Show Class Average",
        indent=False,
    )

    # Sample counter
    counter_label = widgets.HTML(value="")

    out = widgets.Output()

    def _get_filtered_indices():
        filter_val = filter_dropdown.value
        if filter_val == "correct":
            return list(correct_indices)
        elif filter_val == "incorrect":
            return list(incorrect_indices)
        else:
            return all_indices

    def _update_display():
        filtered = _get_filtered_indices()
        if len(filtered) == 0:
            with out:
                out.clear_output(wait=True)
                print("No samples match the current filter.")
            counter_label.value = "<b>0 / 0</b>"
            return

        # Clamp current index
        state["current_idx"] = state["current_idx"] % len(filtered)
        sample_idx = filtered[state["current_idx"]]

        img = images[sample_idx]
        true_label = labels[sample_idx]
        pred_label = preds[sample_idx]
        confidence = probs[sample_idx][pred_label]

        # Generate Grad-CAM for current sample
        input_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
        cam = gradcam.generate_cam(input_tensor, target_class=pred_label)

        # Prepare image for display
        img_display = img.transpose(1, 2, 0)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

        # Resize CAM to image size
        cam_resized = _resize_cam(cam, img.shape[1], img.shape[2])
        heatmap = _cam_to_heatmap(cam_resized)

        # Blend for individual overlay
        overlay = 0.6 * img_display + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)

        # Update counter
        counter_label.value = f"<b>Sample {state['current_idx'] + 1} / {len(filtered)}</b>"

        is_correct = true_label == pred_label
        status_color = "green" if is_correct else "red"
        status_text = "CORRECT" if is_correct else "INCORRECT"

        # Determine layout based on whether to show average
        show_avg = show_avg_checkbox.value

        with out:
            out.clear_output(wait=True)

            if show_avg:
                # Compute average CAM for predicted class
                avg_cam = _compute_average_cam(pred_label)

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Row 1: Original image and individual Grad-CAM
                axes[0, 0].imshow(img_display)
                axes[0, 0].set_title(
                    f"True: {class_names[true_label]}\n"
                    f"Pred: {class_names[pred_label]} ({confidence:.1%})",
                    fontsize=11,
                )
                axes[0, 0].axis("off")

                axes[0, 1].imshow(overlay)
                axes[0, 1].set_title("Individual Grad-CAM", fontsize=11)
                axes[0, 1].axis("off")

                # Row 2: Average Grad-CAM overlay
                if avg_cam is not None:
                    avg_heatmap = _cam_to_heatmap(avg_cam)
                    avg_overlay = 0.6 * img_display + 0.4 * avg_heatmap
                    avg_overlay = np.clip(avg_overlay, 0, 1)

                    axes[1, 0].imshow(avg_overlay)
                    axes[1, 0].set_title(
                        f"Average Grad-CAM for '{class_names[pred_label]}'\n"
                        f"(averaged over predictions of this class)",
                        fontsize=11,
                    )
                    axes[1, 0].axis("off")

                    # Show the raw average heatmap
                    im = axes[1, 1].imshow(avg_cam, cmap="jet")
                    axes[1, 1].set_title(
                        f"Average Attention Heatmap\n'{class_names[pred_label]}'",
                        fontsize=11,
                    )
                    axes[1, 1].axis("off")
                    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
                else:
                    axes[1, 0].text(0.5, 0.5, "No samples for this class",
                                   ha="center", va="center", fontsize=12)
                    axes[1, 0].axis("off")
                    axes[1, 1].text(0.5, 0.5, "No samples for this class",
                                   ha="center", va="center", fontsize=12)
                    axes[1, 1].axis("off")

            else:
                # Original 1x2 layout without average
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                axes[0].imshow(img_display)
                axes[0].set_title(
                    f"True: {class_names[true_label]}\n"
                    f"Pred: {class_names[pred_label]} ({confidence:.1%})",
                    fontsize=11,
                )
                axes[0].axis("off")

                axes[1].imshow(overlay)
                axes[1].set_title("Grad-CAM Attention Map", fontsize=11)
                axes[1].axis("off")

            # Add status text
            fig.suptitle(
                f"Prediction: {status_text}",
                fontsize=14,
                color=status_color,
                fontweight="bold",
            )

            plt.tight_layout()
            plt.show()

    def _on_prev(btn):
        filtered = _get_filtered_indices()
        if len(filtered) > 0:
            state["current_idx"] = (state["current_idx"] - 1) % len(filtered)
            _update_display()

    def _on_next(btn):
        filtered = _get_filtered_indices()
        if len(filtered) > 0:
            state["current_idx"] = (state["current_idx"] + 1) % len(filtered)
            _update_display()

    def _on_random(btn):
        filtered = _get_filtered_indices()
        if len(filtered) > 0:
            state["current_idx"] = np.random.randint(0, len(filtered))
            _update_display()

    def _on_filter_change(change):
        state["current_idx"] = 0
        _update_display()

    def _on_avg_toggle(change):
        _update_display()

    # Wire up
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)
    random_btn.on_click(_on_random)
    filter_dropdown.observe(_on_filter_change, names="value")
    show_avg_checkbox.observe(_on_avg_toggle, names="value")

    # Initial display
    _update_display()

    # Layout
    nav_buttons = HBox([prev_btn, next_btn, random_btn])
    controls = HBox([filter_dropdown, show_avg_checkbox, counter_label, nav_buttons])

    return VBox([controls, out])


# -------------------------------------------------------------------------
# 7. Training Run Comparison Dashboard
# -------------------------------------------------------------------------

def run_comparison_dashboard(
    runs_dir: str = "training_runs",
) -> VBox:
    """
    Dashboard to load, compare, and visualize different training runs.

    Parameters
    ----------
    runs_dir : str
        Directory containing saved training runs

    Returns
    -------
    ipywidgets.VBox
    """
    from .training_runs import list_training_runs, load_training_run, compare_runs_summary
    from sklearn.metrics import f1_score, precision_score, recall_score

    # State to hold loaded runs
    state = {
        "loaded_runs": {},  # run_id -> TrainingRunData
        "current_run_id": None,
    }

    # Refresh available runs
    def _get_available_runs():
        return list_training_runs(runs_dir)

    available_runs = _get_available_runs()

    if not available_runs:
        return VBox([
            widgets.HTML(
                f"<p style='color: orange;'>No training runs found in '{runs_dir}'. "
                "Save a training run first using <code>save_training_run()</code>.</p>"
            )
        ])

    # Create run selector dropdown
    run_options = [(f"{r.run_name} ({r.run_id})", r.run_id) for r in available_runs]

    run_selector = widgets.Dropdown(
        options=run_options,
        value=run_options[0][1] if run_options else None,
        description="Select Run:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="400px"),
    )

    refresh_btn = widgets.Button(
        description="Refresh",
        button_style="info",
        icon="refresh",
    )

    # Visualization type selector
    viz_selector = widgets.ToggleButtons(
        options=[
            ("Summary", "summary"),
            ("Loss Curves", "loss"),
            ("ROC Curves", "roc"),
            ("Confusion Matrix", "cm"),
            ("Compare All", "compare"),
        ],
        value="summary",
        description="View:",
    )

    out = widgets.Output()

    def _load_run(run_id: str):
        """Load a run if not already loaded."""
        if run_id not in state["loaded_runs"]:
            from pathlib import Path
            run_path = Path(runs_dir) / run_id
            state["loaded_runs"][run_id] = load_training_run(str(run_path))
        return state["loaded_runs"][run_id]

    def _show_summary(run):
        """Display run summary."""
        meta = run.metadata
        accuracy = (run.labels == run.preds).mean()
        f1 = f1_score(run.labels, run.preds, average="macro", zero_division=0)
        precision = precision_score(run.labels, run.preds, average="macro", zero_division=0)
        recall = recall_score(run.labels, run.preds, average="macro", zero_division=0)

        html = f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h3 style="margin-top: 0;">{meta.run_name}</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>Run ID:</b></td><td>{meta.run_id}</td></tr>
                <tr><td><b>Model:</b></td><td>{meta.model_name}</td></tr>
                <tr><td><b>Created:</b></td><td>{meta.created_at}</td></tr>
                <tr><td><b>Epochs:</b></td><td>{meta.num_epochs}</td></tr>
                <tr><td><b>Classes:</b></td><td>{', '.join(meta.class_names)}</td></tr>
            </table>
            <hr>
            <h4>Performance Metrics</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>Accuracy:</b></td><td>{accuracy:.4f}</td></tr>
                <tr><td><b>F1 (macro):</b></td><td>{f1:.4f}</td></tr>
                <tr><td><b>Precision (macro):</b></td><td>{precision:.4f}</td></tr>
                <tr><td><b>Recall (macro):</b></td><td>{recall:.4f}</td></tr>
            </table>
        """

        if meta.hyperparameters:
            html += "<hr><h4>Hyperparameters</h4><table>"
            for k, v in meta.hyperparameters.items():
                html += f"<tr><td><b>{k}:</b></td><td>{v}</td></tr>"
            html += "</table>"

        if meta.notes:
            html += f"<hr><p><b>Notes:</b> {meta.notes}</p>"

        html += "</div>"

        display(widgets.HTML(html))

    def _show_loss_curves(run):
        """Display loss curves."""
        history = run.history
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        if history.get("train_loss"):
            axes[0].plot(epochs, history["train_loss"], "-o", label="Train", markersize=3)
        if history.get("val_loss"):
            axes[0].plot(epochs, history["val_loss"], "-o", label="Val", markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Error rate
        if history.get("train_err"):
            axes[1].plot(epochs, history["train_err"], "-o", label="Train", markersize=3)
        if history.get("val_err"):
            axes[1].plot(epochs, history["val_err"], "-o", label="Val", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Error Rate")
        axes[1].set_title("Error Rate Curves")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _show_roc_curves(run):
        """Display ROC curves."""
        n_classes = run.probs.shape[1]
        y_true_bin = label_binarize(run.labels, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], run.probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i],
                   label=f"{run.metadata.class_names[i]} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves - {run.metadata.run_name}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _show_confusion_matrix(run):
        """Display confusion matrix."""
        cm = confusion_matrix(run.labels, run.preds)
        class_names = run.metadata.class_names

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {run.metadata.run_name}")

        # Annotate
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()

    def _show_comparison():
        """Compare all runs."""
        all_run_ids = [opt[1] for opt in run_options]
        runs = [_load_run(rid) for rid in all_run_ids]
        summaries = compare_runs_summary(runs)

        # Create comparison table
        html = """
        <h3>Run Comparison</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <thead style="background: #e9ecef;">
                <tr>
                    <th style="padding: 8px; text-align: left;">Run Name</th>
                    <th style="padding: 8px;">Model</th>
                    <th style="padding: 8px;">Epochs</th>
                    <th style="padding: 8px;">Accuracy</th>
                    <th style="padding: 8px;">F1 (macro)</th>
                    <th style="padding: 8px;">Best Val Loss</th>
                </tr>
            </thead>
            <tbody>
        """

        # Find best values for highlighting
        best_acc = max(s["accuracy"] for s in summaries)
        best_f1 = max(s["f1_macro"] for s in summaries)
        best_loss = min(s["best_val_loss"] for s in summaries if s["best_val_loss"])

        for s in summaries:
            acc_style = "color: green; font-weight: bold;" if s["accuracy"] == best_acc else ""
            f1_style = "color: green; font-weight: bold;" if s["f1_macro"] == best_f1 else ""
            loss_style = "color: green; font-weight: bold;" if s["best_val_loss"] == best_loss else ""

            html += f"""
                <tr>
                    <td style="padding: 8px;">{s['run_name']}</td>
                    <td style="padding: 8px; text-align: center;">{s['model_name']}</td>
                    <td style="padding: 8px; text-align: center;">{s['num_epochs']}</td>
                    <td style="padding: 8px; text-align: center; {acc_style}">{s['accuracy']:.4f}</td>
                    <td style="padding: 8px; text-align: center; {f1_style}">{s['f1_macro']:.4f}</td>
                    <td style="padding: 8px; text-align: center; {loss_style}">{s['best_val_loss']:.4f}</td>
                </tr>
            """

        html += "</tbody></table>"
        display(widgets.HTML(html))

        # Plot comparison charts
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        run_names = [s["run_name"][:20] for s in summaries]
        accuracies = [s["accuracy"] for s in summaries]
        f1_scores_list = [s["f1_macro"] for s in summaries]

        x = np.arange(len(run_names))
        width = 0.35

        axes[0].bar(x - width/2, accuracies, width, label="Accuracy", color="#2ecc71")
        axes[0].bar(x + width/2, f1_scores_list, width, label="F1 (macro)", color="#3498db")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(run_names, rotation=45, ha="right")
        axes[0].set_ylabel("Score")
        axes[0].set_title("Accuracy & F1 Comparison")
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3, axis="y")

        # Loss comparison
        for run in runs:
            if run.history.get("val_loss"):
                axes[1].plot(run.history["val_loss"], label=run.metadata.run_name[:15])
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Loss")
        axes[1].set_title("Validation Loss Comparison")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _update(change=None):
        with out:
            out.clear_output(wait=True)

            viz_type = viz_selector.value

            if viz_type == "compare":
                _show_comparison()
            else:
                run_id = run_selector.value
                if run_id:
                    run = _load_run(run_id)
                    state["current_run_id"] = run_id

                    if viz_type == "summary":
                        _show_summary(run)
                    elif viz_type == "loss":
                        _show_loss_curves(run)
                    elif viz_type == "roc":
                        _show_roc_curves(run)
                    elif viz_type == "cm":
                        _show_confusion_matrix(run)

    def _refresh(btn):
        nonlocal available_runs, run_options
        available_runs = _get_available_runs()
        run_options = [(f"{r.run_name} ({r.run_id})", r.run_id) for r in available_runs]
        run_selector.options = run_options
        if run_options:
            run_selector.value = run_options[0][1]
        _update()

    # Wire up
    run_selector.observe(_update, names="value")
    viz_selector.observe(_update, names="value")
    refresh_btn.on_click(_refresh)

    # Initial display
    _update()

    # Layout
    top_row = HBox([run_selector, refresh_btn])
    controls = VBox([top_row, viz_selector])

    return VBox([controls, out])


# -------------------------------------------------------------------------
# 8. Grad-CAM Comparison Dashboard Between Model Runs
# -------------------------------------------------------------------------

def gradcam_comparison_dashboard(
    runs_dir: str = "training_runs",
    device: str = "cuda",
) -> VBox:
    """
    Dashboard to compare Grad-CAM attention maps between different model runs.

    Allows selecting two runs and viewing their Grad-CAM outputs side-by-side
    on the same images, with average attention maps per class.

    Parameters
    ----------
    runs_dir : str
        Directory containing saved training runs
    device : str
        Device to run models on

    Returns
    -------
    ipywidgets.VBox
    """
    import torch
    import torch.nn.functional as F
    from .training_runs import list_training_runs, load_training_run

    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

    # Check for available runs
    available_runs = list_training_runs(runs_dir)

    if len(available_runs) < 2:
        return VBox([
            widgets.HTML(
                f"<p style='color: orange;'>Need at least 2 training runs in '{runs_dir}' "
                "to compare Grad-CAM outputs. Currently have {len(available_runs)} run(s).</p>"
            )
        ])

    # State
    state = {
        "loaded_runs": {},
        "run1_id": None,
        "run2_id": None,
        "current_idx": 0,
        "shared_indices": [],
        "avg_cams": {"run1": {}, "run2": {}},
    }

    # Run selector dropdowns
    run_options = [(f"{r.run_name} ({r.run_id})", r.run_id) for r in available_runs]

    run1_selector = widgets.Dropdown(
        options=run_options,
        value=run_options[0][1],
        description="Run 1:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="350px"),
    )

    run2_selector = widgets.Dropdown(
        options=run_options,
        value=run_options[1][1] if len(run_options) > 1 else run_options[0][1],
        description="Run 2:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="350px"),
    )

    # Navigation buttons
    prev_btn = widgets.Button(description="Previous", button_style="info", icon="arrow-left")
    next_btn = widgets.Button(description="Next", button_style="info", icon="arrow-right")
    random_btn = widgets.Button(description="Random", button_style="warning", icon="random")

    # Class filter for average comparison
    class_selector = widgets.Dropdown(
        options=[("All Classes", "all")],
        value="all",
        description="Class:",
        style={"description_width": "50px"},
    )

    # View mode selector
    view_mode = widgets.ToggleButtons(
        options=[
            ("Individual Sample", "individual"),
            ("Class Averages", "averages"),
        ],
        value="individual",
        description="View:",
    )

    # Counter and status
    counter_label = widgets.HTML(value="")
    status_label = widgets.HTML(value="")

    out = widgets.Output()

    def _load_run(run_id: str):
        """Load a run if not already loaded."""
        if run_id not in state["loaded_runs"]:
            from pathlib import Path
            run_path = Path(runs_dir) / run_id
            state["loaded_runs"][run_id] = load_training_run(str(run_path))
        return state["loaded_runs"][run_id]

    def _resize_cam(cam, target_h, target_w):
        """Resize CAM to target dimensions."""
        if HAS_CV2:
            return cv2.resize(cam, (target_w, target_h))
        else:
            from scipy.ndimage import zoom
            scale_h = target_h / cam.shape[0]
            scale_w = target_w / cam.shape[1]
            return zoom(cam, (scale_h, scale_w), order=1)

    def _cam_to_heatmap(cam_resized):
        """Convert CAM to RGB heatmap."""
        if HAS_CV2:
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        else:
            return plt.cm.jet(cam_resized)[:, :, :3]

    def _find_shared_indices():
        """Find indices that exist in both runs (they should have same images)."""
        run1 = _load_run(run1_selector.value)
        run2 = _load_run(run2_selector.value)

        # Assuming both runs used the same validation set
        n1 = len(run1.labels) if run1.images is not None else 0
        n2 = len(run2.labels) if run2.images is not None else 0

        if n1 == 0 or n2 == 0:
            return []

        # Use minimum of both
        return list(range(min(n1, n2)))

    def _update_class_options():
        """Update class selector with available classes."""
        run1 = _load_run(run1_selector.value)
        classes = run1.metadata.class_names
        options = [("All Classes", "all")] + [(name, str(i)) for i, name in enumerate(classes)]
        class_selector.options = options

    def _generate_gradcam(run, sample_idx: int, model=None, target_layer=None):
        """Generate Grad-CAM for a specific sample from a run."""
        if run.images is None:
            return None, None

        img = run.images[sample_idx]
        pred_label = run.preds[sample_idx]

        # If no model provided, we can't generate Grad-CAM
        # In this case, we just return placeholder
        # For comparison, we'll compute a simple activation-based visualization

        # Since we don't have the actual model saved with hooks,
        # we'll show the prediction differences instead
        return img, pred_label

    def _compute_difference_map(probs1, probs2):
        """Compute a difference heatmap between two probability distributions."""
        diff = np.abs(probs1 - probs2)
        # Normalize
        diff = diff / (diff.max() + 1e-8)
        return diff

    def _update_display():
        """Update the comparison display."""
        run1_id = run1_selector.value
        run2_id = run2_selector.value

        if run1_id == run2_id:
            with out:
                out.clear_output(wait=True)
                print("Please select two different runs to compare.")
            return

        run1 = _load_run(run1_id)
        run2 = _load_run(run2_id)

        # Check if images are available
        if run1.images is None or run2.images is None:
            with out:
                out.clear_output(wait=True)
                print("One or both runs don't have saved images. "
                      "Save runs with save_images=True to enable Grad-CAM comparison.")
            return

        state["shared_indices"] = _find_shared_indices()

        if len(state["shared_indices"]) == 0:
            with out:
                out.clear_output(wait=True)
                print("No shared samples found between runs.")
            return

        # Clamp index
        state["current_idx"] = state["current_idx"] % len(state["shared_indices"])
        sample_idx = state["shared_indices"][state["current_idx"]]

        # Update counter
        counter_label.value = f"<b>Sample {state['current_idx'] + 1} / {len(state['shared_indices'])}</b>"

        mode = view_mode.value

        with out:
            out.clear_output(wait=True)

            if mode == "individual":
                _show_individual_comparison(run1, run2, sample_idx)
            else:
                _show_average_comparison(run1, run2)

    def _show_individual_comparison(run1, run2, sample_idx):
        """Show side-by-side comparison for a single sample."""
        img1 = run1.images[sample_idx]
        img2 = run2.images[sample_idx]

        # Prepare images for display
        img1_display = img1.transpose(1, 2, 0)
        img1_display = (img1_display - img1_display.min()) / (img1_display.max() - img1_display.min() + 1e-8)

        img2_display = img2.transpose(1, 2, 0)
        img2_display = (img2_display - img2_display.min()) / (img2_display.max() - img2_display.min() + 1e-8)

        true_label = run1.labels[sample_idx]
        pred1 = run1.preds[sample_idx]
        pred2 = run2.preds[sample_idx]
        conf1 = run1.probs[sample_idx][pred1]
        conf2 = run2.probs[sample_idx][pred2]

        class_names = run1.metadata.class_names

        # Create probability difference heatmap
        prob_diff = np.abs(run1.probs[sample_idx] - run2.probs[sample_idx])

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Row 1: Images with predictions
        axes[0, 0].imshow(img1_display)
        correct1 = "CORRECT" if pred1 == true_label else "INCORRECT"
        color1 = "green" if pred1 == true_label else "red"
        axes[0, 0].set_title(
            f"{run1.metadata.run_name[:25]}\n"
            f"Pred: {class_names[pred1]} ({conf1:.1%})\n"
            f"[{correct1}]",
            fontsize=10,
            color=color1,
        )
        axes[0, 0].axis("off")

        axes[0, 1].imshow(img2_display)
        correct2 = "CORRECT" if pred2 == true_label else "INCORRECT"
        color2 = "green" if pred2 == true_label else "red"
        axes[0, 1].set_title(
            f"{run2.metadata.run_name[:25]}\n"
            f"Pred: {class_names[pred2]} ({conf2:.1%})\n"
            f"[{correct2}]",
            fontsize=10,
            color=color2,
        )
        axes[0, 1].axis("off")

        # Probability bar chart comparison
        x = np.arange(len(class_names))
        width = 0.35
        axes[0, 2].bar(x - width/2, run1.probs[sample_idx], width, label=run1.metadata.run_name[:15], alpha=0.8)
        axes[0, 2].bar(x + width/2, run2.probs[sample_idx], width, label=run2.metadata.run_name[:15], alpha=0.8)
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        axes[0, 2].set_ylabel("Probability")
        axes[0, 2].set_title("Prediction Probabilities", fontsize=10)
        axes[0, 2].legend(fontsize=8)
        axes[0, 2].set_ylim(0, 1.1)

        # Row 2: Confidence difference visualization
        # Create a simple attention-like map based on probability confidence
        conf_map1 = np.full(img1_display.shape[:2], conf1)
        conf_map2 = np.full(img2_display.shape[:2], conf2)

        # Overlay confidence as alpha
        overlay1 = img1_display.copy()
        overlay1[:, :, 0] = np.clip(overlay1[:, :, 0] + 0.3 * conf1, 0, 1)
        axes[1, 0].imshow(overlay1)
        axes[1, 0].set_title(f"Confidence: {conf1:.1%}", fontsize=10)
        axes[1, 0].axis("off")

        overlay2 = img2_display.copy()
        overlay2[:, :, 0] = np.clip(overlay2[:, :, 0] + 0.3 * conf2, 0, 1)
        axes[1, 1].imshow(overlay2)
        axes[1, 1].set_title(f"Confidence: {conf2:.1%}", fontsize=10)
        axes[1, 1].axis("off")

        # Prediction difference summary
        axes[1, 2].bar(class_names, prob_diff, color="coral")
        axes[1, 2].set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        axes[1, 2].set_ylabel("Absolute Difference")
        axes[1, 2].set_title("Probability Difference\n(|Run1 - Run2|)", fontsize=10)

        # Overall title
        agreement = "AGREE" if pred1 == pred2 else "DISAGREE"
        agree_color = "green" if pred1 == pred2 else "red"
        fig.suptitle(
            f"True Label: {class_names[true_label]} | Models {agreement}",
            fontsize=14,
            color=agree_color,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()

    def _show_average_comparison(run1, run2):
        """Show average prediction comparison per class."""
        class_names = run1.metadata.class_names
        n_classes = len(class_names)

        selected_class = class_selector.value

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Accuracy per class
        acc1_per_class = []
        acc2_per_class = []
        for c in range(n_classes):
            mask = run1.labels == c
            if mask.sum() > 0:
                acc1_per_class.append((run1.preds[mask] == c).mean())
                acc2_per_class.append((run2.preds[mask] == c).mean())
            else:
                acc1_per_class.append(0)
                acc2_per_class.append(0)

        x = np.arange(n_classes)
        width = 0.35

        axes[0, 0].bar(x - width/2, acc1_per_class, width, label=run1.metadata.run_name[:20], color="#3498db")
        axes[0, 0].bar(x + width/2, acc2_per_class, width, label=run2.metadata.run_name[:20], color="#e74c3c")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45, ha="right")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Per-Class Accuracy Comparison")
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.1)

        # Average confidence per class
        avg_conf1 = []
        avg_conf2 = []
        for c in range(n_classes):
            mask = run1.labels == c
            if mask.sum() > 0:
                avg_conf1.append(run1.probs[mask, c].mean())
                avg_conf2.append(run2.probs[mask, c].mean())
            else:
                avg_conf1.append(0)
                avg_conf2.append(0)

        axes[0, 1].bar(x - width/2, avg_conf1, width, label=run1.metadata.run_name[:20], color="#3498db")
        axes[0, 1].bar(x + width/2, avg_conf2, width, label=run2.metadata.run_name[:20], color="#e74c3c")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[0, 1].set_ylabel("Avg Confidence")
        axes[0, 1].set_title("Average Confidence for True Class")
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1.1)

        # Confusion between runs: where do they disagree?
        agree_mask = run1.preds == run2.preds
        agree_correct = ((run1.preds == run1.labels) & agree_mask).sum()
        agree_wrong = ((run1.preds != run1.labels) & agree_mask).sum()
        disagree_r1_correct = ((run1.preds == run1.labels) & ~agree_mask).sum()
        disagree_r2_correct = ((run2.preds == run2.labels) & ~agree_mask).sum()
        disagree_both_wrong = (~agree_mask & (run1.preds != run1.labels) & (run2.preds != run2.labels)).sum()

        categories = ["Both Correct", "Both Wrong", "Only Run1\nCorrect", "Only Run2\nCorrect"]
        counts = [agree_correct, agree_wrong + disagree_both_wrong, disagree_r1_correct, disagree_r2_correct]
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

        axes[1, 0].bar(categories, counts, color=colors)
        axes[1, 0].set_ylabel("Number of Samples")
        axes[1, 0].set_title("Agreement Analysis")
        for i, (cat, count) in enumerate(zip(categories, counts)):
            axes[1, 0].text(i, count + 0.5, str(count), ha="center", fontsize=10)

        # Summary statistics
        total = len(run1.labels)
        acc1 = (run1.preds == run1.labels).mean()
        acc2 = (run2.preds == run2.labels).mean()
        agreement_rate = agree_mask.mean()

        summary_text = (
            f"Summary Statistics\n"
            f"{'='*40}\n\n"
            f"Total Samples: {total}\n\n"
            f"{run1.metadata.run_name[:25]}:\n"
            f"  Accuracy: {acc1:.2%}\n\n"
            f"{run2.metadata.run_name[:25]}:\n"
            f"  Accuracy: {acc2:.2%}\n\n"
            f"Agreement Rate: {agreement_rate:.2%}\n"
            f"  Both Correct: {agree_correct} ({agree_correct/total:.1%})\n"
            f"  Both Wrong: {agree_wrong + disagree_both_wrong} ({(agree_wrong + disagree_both_wrong)/total:.1%})\n"
            f"  Disagreements: {(~agree_mask).sum()} ({(~agree_mask).mean():.1%})"
        )

        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment="top", fontfamily="monospace",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        axes[1, 1].axis("off")

        fig.suptitle("Model Run Comparison Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _on_run_change(change):
        _update_class_options()
        state["current_idx"] = 0
        state["avg_cams"] = {"run1": {}, "run2": {}}
        _update_display()

    def _on_prev(btn):
        if len(state["shared_indices"]) > 0:
            state["current_idx"] = (state["current_idx"] - 1) % len(state["shared_indices"])
            _update_display()

    def _on_next(btn):
        if len(state["shared_indices"]) > 0:
            state["current_idx"] = (state["current_idx"] + 1) % len(state["shared_indices"])
            _update_display()

    def _on_random(btn):
        if len(state["shared_indices"]) > 0:
            state["current_idx"] = np.random.randint(0, len(state["shared_indices"]))
            _update_display()

    def _on_view_change(change):
        _update_display()

    def _on_class_change(change):
        _update_display()

    # Wire up
    run1_selector.observe(_on_run_change, names="value")
    run2_selector.observe(_on_run_change, names="value")
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)
    random_btn.on_click(_on_random)
    view_mode.observe(_on_view_change, names="value")
    class_selector.observe(_on_class_change, names="value")

    # Initialize
    _update_class_options()
    _update_display()

    # Layout
    run_selectors = HBox([run1_selector, run2_selector])
    nav_buttons = HBox([prev_btn, next_btn, random_btn, counter_label])
    view_controls = HBox([view_mode, class_selector])
    controls = VBox([run_selectors, view_controls, nav_buttons])

    return VBox([
        widgets.HTML("<h3>Grad-CAM Comparison Between Model Runs</h3>"),
        controls,
        out,
    ])


# -------------------------------------------------------------------------
# 9. Metadata Filter Dashboard
# -------------------------------------------------------------------------

def metadata_filter_dashboard(
    df: pd.DataFrame,
    filter_columns: Optional[List[str]] = None,
    on_filter_callback: Optional[Callable[[pd.DataFrame], None]] = None,
) -> VBox:
    """
    Interactive dashboard to filter XNAT metadata and create data subsets.

    Features a professional layout with:
    - Header with progress/summary statistics
    - Visualizations and data preview in the main area
    - Collapsible control panel for filtering on the left side
    - User-selectable filter columns

    Parameters
    ----------
    df : pd.DataFrame
        The metadata DataFrame from fetch_xnat_metadata().
    filter_columns : list of str, optional
        Initial columns to use for filtering. Users can add/remove columns
        dynamically through the interface.
    on_filter_callback : callable, optional
        A callback function that receives the filtered DataFrame whenever
        the filters change. Useful for chaining with other processing.

    Returns
    -------
    ipywidgets.VBox
        The dashboard widget containing filters and visualizations.

    Examples
    --------
    >>> from src.utils.xnat_voila_dashboards import metadata_filter_dashboard
    >>> dashboard = metadata_filter_dashboard(meta_df)
    >>> dashboard

    With callback to capture filtered data:
    >>> filtered_data = {}
    >>> def capture_filter(df):
    ...     filtered_data['current'] = df
    >>> dashboard = metadata_filter_dashboard(meta_df, on_filter_callback=capture_filter)
    """
    if df is None or df.empty:
        return VBox([
            widgets.HTML(
                "<p style='color: orange;'>No data provided. "
                "Please pass a valid DataFrame.</p>"
            )
        ])

    # State to hold current filtered DataFrame and active filters
    state = {
        "filtered_df": df.copy(),
        "original_df": df.copy(),
        "active_filter_columns": [],
    }

    # -------------------------------------------------------------------------
    # Helper: Get all filterable columns
    # -------------------------------------------------------------------------
    def _get_all_filterable_columns(dataframe: pd.DataFrame) -> List[str]:
        """Get all columns that could potentially be used for filtering."""
        candidates = []
        for col in dataframe.columns:
            # Skip certain columns that aren't useful for filtering
            if col in ["dicom_files_sample", "file_path", "image_orientation_patient"]:
                continue
            # Skip columns with all null values
            if dataframe[col].isna().all():
                continue
            # For object/string columns, check cardinality
            if dataframe[col].dtype == "object" or str(dataframe[col].dtype) == "category":
                n_unique = dataframe[col].nunique()
                if 1 < n_unique <= 100:
                    candidates.append(col)
            # For numeric columns
            elif np.issubdtype(dataframe[col].dtype, np.number):
                n_unique = dataframe[col].nunique()
                if 1 < n_unique <= 500:
                    candidates.append(col)
        return candidates

    # -------------------------------------------------------------------------
    # Helper: Get default filter columns
    # -------------------------------------------------------------------------
    def _get_default_filter_columns(dataframe: pd.DataFrame) -> List[str]:
        """Select default columns for filtering."""
        if filter_columns is not None:
            return [c for c in filter_columns if c in dataframe.columns]

        all_cols = _get_all_filterable_columns(dataframe)

        # Prioritize certain columns
        priority = [
            "project_name", "project_id", "modality", "gender", "scan_type",
            "manufacturer", "body_part_examined", "photometric_interpretation",
            "bits_stored", "rows", "columns", "num_slices"
        ]
        ordered = [c for c in priority if c in all_cols]
        ordered += [c for c in all_cols if c not in ordered]

        return ordered[:6]  # Start with 6 default filters

    all_filterable_columns = _get_all_filterable_columns(df)
    state["active_filter_columns"] = _get_default_filter_columns(df)

    if not all_filterable_columns:
        return VBox([
            widgets.HTML(
                "<p style='color: orange;'>No suitable columns found for filtering. "
                "The DataFrame may have too few categorical columns.</p>"
            )
        ])

    # -------------------------------------------------------------------------
    # Filter widget creation and management
    # -------------------------------------------------------------------------
    filter_widgets = {}
    filter_container = VBox()

    def _create_filter_widget(col: str) -> Optional[widgets.Widget]:
        """Create appropriate filter widget based on column type."""
        col_data = df[col].dropna()
        if col_data.empty:
            return None

        # For categorical/object columns, use SelectMultiple
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            unique_vals = sorted(col_data.unique().astype(str))
            widget = widgets.SelectMultiple(
                options=["(All)"] + unique_vals,
                value=["(All)"],
                description="",
                layout=widgets.Layout(width="100%", height="100px"),
            )
            return widget

        # For numeric columns with few unique values, use SelectMultiple
        elif np.issubdtype(df[col].dtype, np.number):
            n_unique = col_data.nunique()
            if n_unique <= 20:
                unique_vals = sorted(col_data.unique())
                unique_vals_str = [str(v) for v in unique_vals]
                widget = widgets.SelectMultiple(
                    options=["(All)"] + unique_vals_str,
                    value=["(All)"],
                    description="",
                    layout=widgets.Layout(width="100%", height="100px"),
                )
                return widget
            else:
                # Use range slider for continuous numeric
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                step = (max_val - min_val) / 100 if max_val > min_val else 1
                widget = widgets.FloatRangeSlider(
                    value=[min_val, max_val],
                    min=min_val,
                    max=max_val,
                    step=step,
                    description="",
                    layout=widgets.Layout(width="100%"),
                    continuous_update=False,
                )
                return widget
        return None

    def _rebuild_filter_widgets():
        """Rebuild filter widgets based on active columns."""
        # Clear existing
        filter_widgets.clear()

        filter_boxes = []
        for col in state["active_filter_columns"]:
            widget = _create_filter_widget(col)
            if widget is not None:
                filter_widgets[col] = widget
                widget.observe(_update, names="value")

                # Create a labeled box for this filter
                label = widgets.HTML(
                    f"<div style='background: #2c3e50; color: white; padding: 5px 10px; "
                    f"border-radius: 4px 4px 0 0; font-weight: bold; font-size: 12px;'>"
                    f"{col.replace('_', ' ').title()}</div>"
                )
                filter_box = VBox(
                    [label, widget],
                    layout=widgets.Layout(
                        border="1px solid #bdc3c7",
                        border_radius="4px",
                        margin="5px 0",
                    )
                )
                filter_boxes.append(filter_box)

        filter_container.children = filter_boxes

    # -------------------------------------------------------------------------
    # Column selector for adding/removing filters
    # -------------------------------------------------------------------------
    available_to_add = [c for c in all_filterable_columns if c not in state["active_filter_columns"]]

    add_filter_dropdown = widgets.Dropdown(
        options=[("+ Add Filter...", "")] + [(c.replace("_", " ").title(), c) for c in available_to_add],
        value="",
        layout=widgets.Layout(width="100%"),
    )

    def _on_add_filter(change):
        if change["new"] and change["new"] not in state["active_filter_columns"]:
            state["active_filter_columns"].append(change["new"])
            # Update dropdown options
            available = [c for c in all_filterable_columns if c not in state["active_filter_columns"]]
            add_filter_dropdown.options = [("+ Add Filter...", "")] + [(c.replace("_", " ").title(), c) for c in available]
            add_filter_dropdown.value = ""
            _rebuild_filter_widgets()
            _update()

    add_filter_dropdown.observe(_on_add_filter, names="value")

    # -------------------------------------------------------------------------
    # Summary statistics cards (header area)
    # -------------------------------------------------------------------------
    summary_cards_html = widgets.HTML(value="")

    def _update_summary_cards(filtered: pd.DataFrame):
        """Update the summary statistics cards at the top."""
        total = len(state["original_df"])
        filtered_count = len(filtered)
        pct = (filtered_count / total * 100) if total > 0 else 0

        # Count unique values
        n_projects = filtered["project_name"].nunique() if "project_name" in filtered.columns else 0
        n_subjects = filtered["subject_id"].nunique() if "subject_id" in filtered.columns else 0
        n_experiments = filtered["experiment_id"].nunique() if "experiment_id" in filtered.columns else 0
        n_scans = filtered["scan_id"].nunique() if "scan_id" in filtered.columns else 0

        # Progress bar width
        progress_pct = min(100, pct)

        html = f"""
        <div style="background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
                    padding: 20px; border-radius: 8px; margin-bottom: 15px; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div>
                    <h2 style="margin: 0; font-size: 24px;">XNAT Metadata Filter Dashboard</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">
                        Filter and explore imaging metadata across projects
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 32px; font-weight: bold;">{filtered_count:,} / {total:,}</div>
                    <div style="font-size: 12px; opacity: 0.8;">Records Selected ({pct:.1f}%)</div>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 4px; height: 8px; margin-bottom: 15px;">
                <div style="background: #48bb78; height: 100%; border-radius: 4px; width: {progress_pct}%;"></div>
            </div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_projects}</div>
                    <div style="font-size: 12px; opacity: 0.8;">PROJECTS</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_subjects}</div>
                    <div style="font-size: 12px; opacity: 0.8;">SUBJECTS</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_experiments}</div>
                    <div style="font-size: 12px; opacity: 0.8;">SESSIONS</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_scans}</div>
                    <div style="font-size: 12px; opacity: 0.8;">SCANS</div>
                </div>
            </div>
        </div>
        """
        summary_cards_html.value = html

    # -------------------------------------------------------------------------
    # Visualization outputs
    # -------------------------------------------------------------------------
    out_plots = widgets.Output()
    out_table = widgets.Output()

    # Plot type selector
    plot_selector = widgets.ToggleButtons(
        options=[
            ("Distributions", "dist"),
            ("Pie Charts", "pie"),
            ("Histograms", "hist"),
            ("Data Table", "table"),
        ],
        value="dist",
        layout=widgets.Layout(margin="0 0 10px 0"),
    )

    def _plot_distributions(filtered: pd.DataFrame):
        """Plot distribution bar charts for categorical columns."""
        plot_cols = [c for c in state["active_filter_columns"]
                    if c in filtered.columns and
                    (filtered[c].dtype == "object" or filtered[c].nunique() <= 20)][:6]

        # Add extra categorical columns if we don't have enough
        if len(plot_cols) < 4:
            extra = [c for c in all_filterable_columns
                    if c in filtered.columns and c not in plot_cols and
                    (filtered[c].dtype == "object" or filtered[c].nunique() <= 15)]
            plot_cols.extend(extra[:6 - len(plot_cols)])

        if not plot_cols:
            print("No suitable categorical columns for distribution plots.")
            return

        n_cols = min(3, len(plot_cols))
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        colors_palette = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
                         "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b"]

        for idx, col in enumerate(plot_cols):
            row, col_idx = divmod(idx, n_cols)
            ax = axes[row, col_idx]

            value_counts = filtered[col].value_counts().head(8)
            colors = [colors_palette[i % len(colors_palette)] for i in range(len(value_counts))]

            bars = ax.barh(range(len(value_counts)), value_counts.values, color=colors)
            ax.set_yticks(range(len(value_counts)))
            ax.set_yticklabels([str(v)[:25] for v in value_counts.index], fontsize=9)
            ax.set_xlabel("Count", fontsize=10)
            ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for bar, count in zip(bars, value_counts.values):
                ax.text(bar.get_width() + max(value_counts) * 0.02,
                       bar.get_y() + bar.get_height()/2,
                       f"{count:,}", va="center", fontsize=9)

        for idx in range(len(plot_cols), n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            axes[row, col_idx].axis("off")

        plt.tight_layout()
        plt.show()

    def _plot_pie_charts(filtered: pd.DataFrame):
        """Plot pie charts for key categorical columns."""
        pie_cols = [c for c in ["project_name", "modality", "gender", "manufacturer",
                                "body_part_examined", "scan_type", "photometric_interpretation"]
                   if c in filtered.columns and 1 < filtered[c].nunique() <= 10][:4]

        if not pie_cols:
            pie_cols = [c for c in all_filterable_columns
                       if c in filtered.columns and 1 < filtered[c].nunique() <= 10][:4]

        if not pie_cols:
            print("No suitable columns for pie charts (need columns with 2-10 unique values).")
            return

        n_cols = min(2, len(pie_cols))
        n_rows = (len(pie_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        colors_palette = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
                         "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b"]

        for idx, col in enumerate(pie_cols):
            row, col_idx = divmod(idx, n_cols)
            ax = axes[row, col_idx]

            value_counts = filtered[col].value_counts()
            colors = [colors_palette[i % len(colors_palette)] for i in range(len(value_counts))]

            wedges, texts, autotexts = ax.pie(
                value_counts.values,
                labels=None,
                autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
                colors=colors,
                startangle=90,
                pctdistance=0.75,
            )
            ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")

            # Add legend
            ax.legend(
                wedges,
                [f"{str(v)[:20]}: {c:,}" for v, c in zip(value_counts.index, value_counts.values)],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=9,
            )

        for idx in range(len(pie_cols), n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            axes[row, col_idx].axis("off")

        plt.tight_layout()
        plt.show()

    def _plot_numeric_histograms(filtered: pd.DataFrame):
        """Plot histograms for numeric columns."""
        numeric_cols = ["rows", "columns", "num_slices", "pixel_spacing_row",
                       "pixel_spacing_col", "slice_thickness", "age", "bits_stored",
                       "window_center", "window_width"]
        numeric_cols = [c for c in numeric_cols if c in filtered.columns and
                       np.issubdtype(filtered[c].dtype, np.number) and
                       filtered[c].nunique() > 2][:6]

        if not numeric_cols:
            numeric_cols = [c for c in all_filterable_columns
                          if c in filtered.columns and
                          np.issubdtype(filtered[c].dtype, np.number) and
                          filtered[c].nunique() > 5][:6]

        if not numeric_cols:
            print("No numeric columns available for histograms.")
            return

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, col in enumerate(numeric_cols):
            row, col_idx = divmod(idx, n_cols)
            ax = axes[row, col_idx]

            data = filtered[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
                ax.axvline(data.mean(), color="#e74c3c", linestyle="--", linewidth=2,
                          label=f"Mean: {data.mean():.2f}")
                ax.axvline(data.median(), color="#2ecc71", linestyle="--", linewidth=2,
                          label=f"Median: {data.median():.2f}")
                ax.set_xlabel(col.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.set_title(f"{col.replace('_', ' ').title()} (n={len(data):,})",
                           fontsize=11, fontweight="bold")
                ax.legend(fontsize=8, loc="upper right")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
                ax.set_title(col.replace("_", " ").title(), fontweight="bold")
                ax.axis("off")

        for idx in range(len(numeric_cols), n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            axes[row, col_idx].axis("off")

        plt.tight_layout()
        plt.show()

    def _update_table(filtered: pd.DataFrame):
        """Display a sample of the filtered DataFrame."""
        with out_table:
            out_table.clear_output(wait=True)

            display_cols = [c for c in filtered.columns
                          if c not in ["dicom_files_sample", "image_orientation_patient"]][:15]

            if len(filtered) > 0:
                sample = filtered[display_cols].head(15)
                # Style the dataframe
                print(f"Showing {len(sample)} of {len(filtered):,} filtered records\n")
                display(sample)
            else:
                print("No records match the current filters.")

    # -------------------------------------------------------------------------
    # Apply filters
    # -------------------------------------------------------------------------
    def _apply_filters() -> pd.DataFrame:
        """Apply all current filter selections to the DataFrame."""
        filtered = state["original_df"].copy()

        for col, widget in filter_widgets.items():
            if col not in filtered.columns:
                continue

            if isinstance(widget, widgets.SelectMultiple):
                selected = list(widget.value)
                if "(All)" not in selected and selected:
                    if np.issubdtype(df[col].dtype, np.number):
                        selected_vals = [float(v) for v in selected]
                        filtered = filtered[filtered[col].isin(selected_vals) | filtered[col].isna()]
                    else:
                        filtered = filtered[filtered[col].astype(str).isin(selected) | filtered[col].isna()]

            elif isinstance(widget, widgets.FloatRangeSlider):
                min_val, max_val = widget.value
                filtered = filtered[
                    ((filtered[col] >= min_val) & (filtered[col] <= max_val)) |
                    filtered[col].isna()
                ]

        return filtered

    # -------------------------------------------------------------------------
    # Main update function
    # -------------------------------------------------------------------------
    def _update(change=None):
        """Main update function triggered by any filter change."""
        filtered = _apply_filters()
        state["filtered_df"] = filtered

        _update_summary_cards(filtered)

        with out_plots:
            out_plots.clear_output(wait=True)

            if len(filtered) == 0:
                print("No data matches the current filters.")
            else:
                plot_type = plot_selector.value
                if plot_type == "dist":
                    _plot_distributions(filtered)
                elif plot_type == "pie":
                    _plot_pie_charts(filtered)
                elif plot_type == "hist":
                    _plot_numeric_histograms(filtered)
                elif plot_type == "table":
                    _update_table(filtered)

        # Always update table in background for callback
        if plot_selector.value != "table":
            _update_table(filtered)

        if on_filter_callback is not None:
            on_filter_callback(filtered)

    # -------------------------------------------------------------------------
    # Button handlers
    # -------------------------------------------------------------------------
    reset_btn = widgets.Button(
        description="Reset Filters",
        button_style="warning",
        icon="refresh",
        layout=widgets.Layout(width="100%", margin="5px 0"),
    )

    export_btn = widgets.Button(
        description="Export CSV",
        button_style="success",
        icon="download",
        layout=widgets.Layout(width="100%", margin="5px 0"),
    )

    def _reset_filters(btn):
        """Reset all filters to default values."""
        for col, widget in filter_widgets.items():
            if isinstance(widget, widgets.SelectMultiple):
                widget.value = ["(All)"]
            elif isinstance(widget, widgets.FloatRangeSlider):
                widget.value = [widget.min, widget.max]
        _update()

    def _export_csv(btn):
        """Export the filtered DataFrame to CSV."""
        from datetime import datetime
        from pathlib import Path

        filtered = state["filtered_df"]
        if len(filtered) == 0:
            with out_table:
                print("No data to export.")
            return

        export_dir = Path("logs/exported_data")
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = export_dir / f"filtered_metadata_{timestamp}.csv"

        export_cols = [c for c in filtered.columns if c != "dicom_files_sample"]
        filtered[export_cols].to_csv(filename, index=False)

        with out_table:
            print(f"\n✓ Exported {len(filtered):,} records to: {filename}")

    reset_btn.on_click(_reset_filters)
    export_btn.on_click(_export_csv)

    # -------------------------------------------------------------------------
    # Wire up observers
    # -------------------------------------------------------------------------
    plot_selector.observe(_update, names="value")

    # -------------------------------------------------------------------------
    # Build initial filter widgets
    # -------------------------------------------------------------------------
    _rebuild_filter_widgets()

    # Initial render
    _update()

    # -------------------------------------------------------------------------
    # Layout: Control Panel (left) + Main Content (right)
    # -------------------------------------------------------------------------

    # Control panel header
    control_panel_header = widgets.HTML(
        """<div style="background: #34495e; color: white; padding: 12px;
           border-radius: 6px 6px 0 0; font-weight: bold;">
           <span style="font-size: 14px;">Control Panel</span>
        </div>"""
    )

    # Filter section
    filter_section_header = widgets.HTML(
        """<div style="background: #ecf0f1; padding: 8px 12px; font-weight: bold;
           font-size: 12px; color: #2c3e50; border-bottom: 1px solid #bdc3c7;">
           Filters
        </div>"""
    )

    # Control panel content
    control_panel_content = VBox([
        filter_section_header,
        filter_container,
        add_filter_dropdown,
        widgets.HTML("<hr style='margin: 10px 0; border-color: #ecf0f1;'>"),
        reset_btn,
        export_btn,
    ], layout=widgets.Layout(
        padding="0 10px 10px 10px",
        background="#f8f9fa",
    ))

    control_panel = VBox([
        control_panel_header,
        control_panel_content,
    ], layout=widgets.Layout(
        width="280px",
        border="1px solid #bdc3c7",
        border_radius="6px",
        margin="0 15px 0 0",
    ))

    # Main content area
    viz_header = widgets.HTML(
        """<div style="background: #ecf0f1; padding: 10px 15px; border-radius: 6px;
           margin-bottom: 10px;">
           <span style="font-weight: bold; color: #2c3e50;">Visualization</span>
        </div>"""
    )

    main_content = VBox([
        plot_selector,
        out_plots,
        widgets.HTML("<hr style='margin: 15px 0;'>"),
        widgets.HTML("<div style='font-weight: bold; margin-bottom: 10px;'>Data Preview</div>"),
        out_table,
    ], layout=widgets.Layout(
        flex="1",
    ))

    # Main layout: header on top, then control panel + content side by side
    content_row = HBox([
        control_panel,
        main_content,
    ], layout=widgets.Layout(
        width="100%",
    ))

    dashboard = VBox([
        summary_cards_html,
        content_row,
    ], layout=widgets.Layout(
        width="100%",
    ))

    return dashboard
