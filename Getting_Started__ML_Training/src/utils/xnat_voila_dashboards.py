"""
Reusable Voila-friendly dashboards for the XNAT ResNet notebook.

Each function returns an ipywidgets container (VBox) that you can display
directly in Jupyter/Voila, e.g.:

    from xnat_voila_dashboards import (
        class_distribution_dashboard,
        training_history_dashboard,
        confusion_matrix_dashboard,
        pixel_intensity_dashboard,
    )

    class_dist_ui = class_distribution_dashboard(train_labels, val_labels, class_names)
    class_dist_ui

These are intentionally small, focused dashboards rather than one giant UI.
"""

from typing import Sequence, Mapping, Optional

import numpy as np
import ipywidgets as widgets
from ipywidgets import VBox, HBox
from sklearn.metrics import confusion_matrix

import plotly.graph_objs as go


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _ensure_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # torch tensors, lists, etc.
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


# -----------------------------------------------------------------------------
# 1. Class distribution dashboard
# -----------------------------------------------------------------------------

def class_distribution_dashboard(
    train_labels: Sequence[int],
    val_labels: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> VBox:
    """
    Small dashboard to explore train/val class distribution.

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
        A widget containing controls + a Plotly bar chart.
    """
    y_train = _ensure_numpy(train_labels)
    y_val = _ensure_numpy(val_labels) if val_labels is not None else None

    unique_classes = np.unique(y_train if y_val is None else np.concatenate([y_train, y_val]))
    unique_classes = np.sort(unique_classes)

    if class_names is None:
        class_names = [str(int(c)) for c in unique_classes]
    else:
        class_names = list(class_names)

    # Controls
    dataset_options = ["train"]
    if y_val is not None:
        dataset_options.append("val")
        dataset_options.append("both")

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

    fig = go.FigureWidget()

    def _compute_counts(labels: np.ndarray) -> np.ndarray:
        counts = np.array([(labels == c).sum() for c in unique_classes], dtype=float)
        return counts

    def _update(*_):
        mode = dataset_selector.value
        y_mode = normalize_selector.value

        with fig.batch_update():
            fig.data = []

            if mode in ("train", "both"):
                counts_train = _compute_counts(y_train)
                if y_mode == "prop":
                    total = counts_train.sum() or 1.0
                    counts_train = counts_train / total
                fig.add_bar(
                    x=class_names,
                    y=counts_train,
                    name="Train",
                )

            if y_val is not None and mode in ("val", "both"):
                counts_val = _compute_counts(y_val)
                if y_mode == "prop":
                    total = counts_val.sum() or 1.0
                    counts_val = counts_val / total
                fig.add_bar(
                    x=class_names,
                    y=counts_val,
                    name="Val",
                )

            fig.update_layout(
                barmode="group",
                xaxis_title="Class",
                yaxis_title="Count" if y_mode == "count" else "Fraction",
                title="Class distribution",
                margin=dict(l=40, r=10, t=40, b=40),
            )

    dataset_selector.observe(_update, "value")
    normalize_selector.observe(_update, "value")

    _update()

    controls = HBox([dataset_selector, normalize_selector])
    return VBox([controls, fig])


# -----------------------------------------------------------------------------
# 2. Training history (loss/error) dashboard
# -----------------------------------------------------------------------------

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

    Parameters
    ----------
    history : mapping
        Dict with lists/arrays of metrics per epoch.
    title : str
        Title prefix for the figure.

    Returns
    -------
    ipywidgets.VBox
    """
    # Convert to numpy for convenience
    hist_np = {k: np.asarray(v, dtype=float) for k, v in history.items()}

    metric_selector = widgets.ToggleButtons(
        options=[("Loss", "loss"), ("Error rate", "err")],
        value="loss",
        description="Metric:",
    )

    smooth_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=max(1, len(hist_np.get("train_loss", [])) // 4) or 1,
        step=1,
        description="Smoothing:",
        continuous_update=False,
    )

    fig = go.FigureWidget()

    def _smooth(x: np.ndarray, win: int) -> np.ndarray:
        if win <= 1 or len(x) == 0:
            return x
        k = min(win, len(x))
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="same")

    def _update(*_):
        metric = metric_selector.value
        win = int(smooth_slider.value)

        train_key = f"train_{metric}"
        val_key = f"val_{metric}"

        train_vals = hist_np.get(train_key, np.array([]))
        val_vals = hist_np.get(val_key, np.array([]))

        epochs = np.arange(1, len(train_vals) + 1)

        y_train = _smooth(train_vals, win)
        y_val = _smooth(val_vals, win)

        ylabel = "Cross-entropy loss" if metric == "loss" else "Error rate (1 - accuracy)"

        with fig.batch_update():
            fig.data = []
            if len(y_train):
                fig.add_scatter(x=epochs, y=y_train, mode="lines+markers", name="Train")
            if len(y_val):
                fig.add_scatter(x=epochs, y=y_val, mode="lines+markers", name="Val")

            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title=ylabel,
                title=f"{title} – {metric}",
                margin=dict(l=40, r=10, t=40, b=40),
            )

    metric_selector.observe(_update, "value")
    smooth_slider.observe(_update, "value")

    _update()

    controls = HBox([metric_selector, smooth_slider])
    return VBox([controls, fig])


# -----------------------------------------------------------------------------
# 3. Confusion matrix dashboard
# -----------------------------------------------------------------------------

def confusion_matrix_dashboard(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> VBox:
    """
    Dashboard for a confusion matrix with optional normalization.

    Parameters
    ----------
    y_true : sequence of int
        Ground-truth labels.
    y_pred : sequence of int
        Predicted labels.
    class_names : sequence of str, optional
        Class names; if None, numeric indices are used.
    normalize : bool
        If True, start with row-normalized CM.
    title : str
        Title prefix for the figure.

    Returns
    -------
    ipywidgets.VBox
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

    fig = go.FigureWidget()

    def _update(*_):
        mode = norm_selector.value
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if mode == "row":
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_disp = cm / row_sums
            z_text = np.round(cm_disp, 3)
            colorbar_title = "Fraction"
        else:
            cm_disp = cm.astype(float)
            z_text = cm
            colorbar_title = "Count"

        with fig.batch_update():
            fig.data = []
            fig.add_heatmap(
                z=cm_disp,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                colorbar=dict(title=colorbar_title),
                hovertemplate="True=%{y}<br>Pred=%{x}<br>Value=%{z}<extra></extra>",
            )
            fig.update_layout(
                xaxis_title="Predicted label",
                yaxis_title="True label",
                title=title,
                margin=dict(l=60, r=10, t=40, b=60),
            )

    norm_selector.observe(_update, "value")
    _update()

    controls = HBox([norm_selector])
    return VBox([controls, fig])


# -----------------------------------------------------------------------------
# 4. Pixel intensity distribution dashboard
# -----------------------------------------------------------------------------

def pixel_intensity_dashboard(
    images: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    n_bins: int = 64,
) -> VBox:
    """
    Dashboard to inspect pixel intensity distributions.

    Parameters
    ----------
    images : np.ndarray
        Image data as (N, H, W) or (N, 1, H, W) or (N, C, H, W).
        For RGB, intensities are flattened across channels.
    labels : sequence of int, optional
        Class labels (len = N). If provided, per-class histograms are available.
    class_names : sequence of str, optional
        Class names; if None, numeric indices are used.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    ipywidgets.VBox
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

    fig = go.FigureWidget()

    def _update(*_):
        if labels_np is not None and class_selector.value != "all":
            c = int(class_selector.value)
            mask = labels_np == c
            data = imgs_flat[mask]
        else:
            data = imgs_flat

        # Flatten all pixels
        pixels = data.reshape(-1)
        pixels = pixels[np.isfinite(pixels)]

        if pixels.size == 0:
            hist_y = np.array([0])
            hist_x = np.array([0])
        else:
            hist_y, bin_edges = np.histogram(pixels, bins=n_bins)
            hist_x = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        with fig.batch_update():
            fig.data = []
            fig.add_bar(x=hist_x, y=hist_y)
            if labels_np is not None and class_selector.value != "all":
                idx = int(class_selector.value)
                cname = class_names[unique_classes.tolist().index(idx)]
                title = f"Pixel intensity histogram – class {cname}"
            else:
                title = "Pixel intensity histogram – all images"

            fig.update_layout(
                xaxis_title="Pixel intensity",
                yaxis_title="Count",
                title=title,
                margin=dict(l=40, r=10, t=40, b=40),
            )

    if class_selector is not None:
        class_selector.observe(_update, "value")
        controls = HBox([class_selector])
    else:
        controls = HBox([])

    _update()
    return VBox([controls, fig])
