"""
Reusable Voila-friendly dashboards for the XNAT ResNet notebook.

Each function returns an ipywidgets.VBox that you can display in
Jupyter and Voila, e.g.:

    from src.utils.xnat_voila_dashboards import (
        class_distribution_dashboard,
        training_history_dashboard,
        confusion_matrix_dashboard,
        pixel_intensity_dashboard,
    )

    ui = class_distribution_dashboard(train_labels, val_labels, class_names)
    ui

All plots are rendered with matplotlib and updated via ipywidgets.
"""

from typing import Sequence, Mapping, Optional, List

import numpy as np
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
    Interactive Grad-CAM dashboard with navigation buttons.

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
    state = {"current_idx": 0, "sample_list": all_indices}

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

        # Generate Grad-CAM
        input_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
        cam = gradcam.generate_cam(input_tensor, target_class=pred_label)

        # Prepare image for display
        img_display = img.transpose(1, 2, 0)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

        # Resize CAM to image size
        if HAS_CV2:
            cam_resized = cv2.resize(cam, (img.shape[2], img.shape[1]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        else:
            # Fallback without cv2
            from scipy.ndimage import zoom
            scale_h = img.shape[1] / cam.shape[0]
            scale_w = img.shape[2] / cam.shape[1]
            cam_resized = zoom(cam, (scale_h, scale_w), order=1)
            # Simple colormap
            heatmap = plt.cm.jet(cam_resized)[:, :, :3]

        # Blend
        overlay = 0.6 * img_display + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)

        # Update counter
        counter_label.value = f"<b>Sample {state['current_idx'] + 1} / {len(filtered)}</b>"

        is_correct = true_label == pred_label
        status_color = "green" if is_correct else "red"
        status_text = "CORRECT" if is_correct else "INCORRECT"

        with out:
            out.clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Original image
            axes[0].imshow(img_display)
            axes[0].set_title(
                f"True: {class_names[true_label]}\n"
                f"Pred: {class_names[pred_label]} ({confidence:.1%})",
                fontsize=11,
            )
            axes[0].axis("off")

            # Grad-CAM overlay
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

    # Wire up
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)
    random_btn.on_click(_on_random)
    filter_dropdown.observe(_on_filter_change, names="value")

    # Initial display
    _update_display()

    # Layout
    nav_buttons = HBox([prev_btn, next_btn, random_btn])
    controls = HBox([filter_dropdown, counter_label, nav_buttons])

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
