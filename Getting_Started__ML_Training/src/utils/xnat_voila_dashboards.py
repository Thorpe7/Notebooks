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

from typing import Sequence, Mapping, Optional

import numpy as np
import ipywidgets as widgets
from ipywidgets import VBox, HBox
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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
