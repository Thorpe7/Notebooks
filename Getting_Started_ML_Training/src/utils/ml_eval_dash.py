"""
ML Evaluation Dashboard for comparing training runs.

This module provides an interactive dashboard for evaluating and comparing
ML training runs with visualizations including loss curves, confusion matrices,
ROC curves, confidence distributions, and Grad-CAM comparisons.
"""

from typing import List

import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import VBox, HBox
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


class MLEvaluationDashboard:
    """
    Container class for the ML Evaluation Dashboard that provides access
    to training runs and comparison data.

    Attributes
    ----------
    widget : VBox
        The dashboard widget to display.
    selected_runs : list
        List of currently selected run IDs.

    Methods
    -------
    get_selected_runs()
        Returns list of selected run IDs.
    get_loaded_runs()
        Returns dict of loaded TrainingRunData objects.
    """

    def __init__(self):
        self.widget = None
        self.selected_runs = []
        self._state = None

    def get_selected_runs(self) -> List[str]:
        """Return list of selected run IDs."""
        return self.selected_runs

    def get_loaded_runs(self) -> dict:
        """Return dict of loaded runs."""
        if self._state is not None:
            return self._state.get("loaded_runs", {})
        return {}

    def _repr_mimebundle_(self, **kwargs):
        if self.widget is not None:
            return self.widget._repr_mimebundle_(**kwargs)
        return {"text/plain": "Dashboard not initialized"}

    def _ipython_display_(self):
        if self.widget is not None:
            from IPython.display import display
            display(self.widget)


def ml_evaluation_dashboard(
    runs_dir: str = "training_runs",
    device: str = "cuda",
) -> MLEvaluationDashboard:
    """
    Unified ML Evaluation Dashboard for comparing training runs.

    Features a professional layout with:
    - Header showing number of training runs and key statistics
    - Side-by-side comparison of metrics, loss curves, ROC curves
    - Grad-CAM heatmap comparison for individual images
    - Average Grad-CAM heatmap comparison across runs
    - Dataset CSV tracking per run

    Parameters
    ----------
    runs_dir : str
        Directory containing saved training runs.
    device : str
        Device for Grad-CAM computation ('cuda' or 'cpu').

    Returns
    -------
    MLEvaluationDashboard
        Dashboard object with widget and data access methods.

    Examples
    --------
    >>> from src.utils.ml_eval_dash import ml_evaluation_dashboard
    >>> dashboard = ml_evaluation_dashboard("training_runs")
    >>> dashboard  # displays the dashboard
    """
    import torch
    import torch.nn.functional as F
    from .training_runs import list_training_runs, load_training_run, compare_runs_summary

    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

    # Create dashboard container
    dashboard_obj = MLEvaluationDashboard()

    # Check for available runs
    available_runs = list_training_runs(runs_dir)

    if not available_runs:
        dashboard_obj.widget = VBox([
            widgets.HTML(
                f"""<div style="background: linear-gradient(to right, #1a365d 0%, #2c5282 25%, #d69e2e 50%, #48bb78 75%, #276749 100%);
                    padding: 20px; border-radius: 8px; margin-bottom: 15px; color: white;">
                    <h2 style="margin: 0;">ML Evaluation Dashboard</h2>
                    <p style="opacity: 0.8;">No training runs found in '{runs_dir}'</p>
                </div>
                <p style='color: orange; padding: 20px;'>
                    Save a training run first using <code>save_training_run()</code>.
                </p>"""
            )
        ])
        return dashboard_obj

    # State
    state = {
        "loaded_runs": {},
        "selected_run_ids": [],
        "current_sample_idx": 0,
        "avg_cam_cache": {},
    }
    dashboard_obj._state = state

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------
    def _load_run(run_id: str):
        """Load a run if not already loaded."""
        if run_id not in state["loaded_runs"]:
            from pathlib import Path
            run_path = Path(runs_dir) / run_id
            state["loaded_runs"][run_id] = load_training_run(str(run_path))
        return state["loaded_runs"][run_id]

    def _resize_cam(cam, target_h, target_w):
        if HAS_CV2:
            return cv2.resize(cam, (target_w, target_h))
        else:
            from scipy.ndimage import zoom
            scale_h = target_h / cam.shape[0]
            scale_w = target_w / cam.shape[1]
            return zoom(cam, (scale_h, scale_w), order=1)

    def _cam_to_heatmap(cam_resized):
        if HAS_CV2:
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        else:
            return plt.cm.jet(cam_resized)[:, :, :3]

    def _get_model_and_layer(run):
        """Try to load model from run and get target layer for Grad-CAM."""
        if run.model_state_dict is None:
            return None, None

        model_name = run.metadata.model_name.lower()

        try:
            # Try to create the model based on model_name
            if "efficientnet" in model_name:
                from ..models.efficientnet import create_efficientnet_b0
                num_classes = len(run.metadata.class_names)
                model = create_efficientnet_b0(num_classes=num_classes)
                model.load_state_dict(run.model_state_dict)
                model.to(device)
                model.eval()
                # Get target layer for EfficientNet
                target_layer = model.model.features[-1]
                return model, target_layer
            elif "resnet" in model_name:
                from torchvision import models
                num_classes = len(run.metadata.class_names)
                model = models.resnet50(weights=None)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(run.model_state_dict)
                model.to(device)
                model.eval()
                target_layer = model.layer4[-1]
                return model, target_layer
        except Exception as e:
            print(f"Could not load model: {e}")
            return None, None

        return None, None

    def _compute_gradcam(model, target_layer, img_tensor, target_class=None):
        """Compute Grad-CAM for a single image."""
        if model is None or target_layer is None:
            return None

        # Storage for activations and gradients
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            model.zero_grad()
            img_batch = img_tensor.unsqueeze(0).to(device)
            output = model(img_batch)

            # Get target class (predicted class if not specified)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Backward pass
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot)

            # Compute Grad-CAM
            if activations and gradients:
                activation = activations[0]
                gradient = gradients[0]

                # Global average pooling of gradients
                weights = gradient.mean(dim=(2, 3), keepdim=True)

                # Weighted combination of activation maps
                cam = (weights * activation).sum(dim=1, keepdim=True)
                cam = F.relu(cam)

                # Normalize
                cam = cam.squeeze().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                return cam
        except Exception as e:
            print(f"Grad-CAM computation error: {e}")
        finally:
            forward_handle.remove()
            backward_handle.remove()

        return None

    def _get_dataset_info(run):
        """Extract dataset CSV info from run metadata."""
        hp = run.metadata.hyperparameters or {}
        dataset_csv = hp.get("dataset_csv", hp.get("data_csv", hp.get("csv_file", None)))
        return dataset_csv if dataset_csv else "None"

    # -------------------------------------------------------------------------
    # Summary header HTML
    # -------------------------------------------------------------------------
    summary_header = widgets.HTML(value="")

    def _update_header():
        """Update the summary header with current statistics."""
        n_runs = len(available_runs)
        selected = state["selected_run_ids"]
        n_selected = len(selected)

        # Compute aggregate stats if runs are selected
        if n_selected > 0:
            runs_data = [_load_run(rid) for rid in selected]
            total_samples = sum(len(r.labels) for r in runs_data)
            avg_accuracy = np.mean([(r.preds == r.labels).mean() for r in runs_data])
            models_used = list(set(r.metadata.model_name for r in runs_data))
        else:
            total_samples = 0
            avg_accuracy = 0
            models_used = []

        html = f"""
        <div style="background: linear-gradient(to right, #1a365d 0%, #2c5282 25%, #d69e2e 50%, #48bb78 75%, #276749 100%);
                    padding: 20px; border-radius: 8px; margin-bottom: 15px; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div>
                    <h2 style="margin: 0; font-size: 24px;">ML Evaluation Dashboard</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">
                        Compare training runs, metrics, and Grad-CAM attention maps
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 32px; font-weight: bold;">{n_selected} / {n_runs}</div>
                    <div style="font-size: 12px; opacity: 0.8;">Runs Selected</div>
                </div>
            </div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_runs}</div>
                    <div style="font-size: 12px; opacity: 0.8;">TOTAL RUNS</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{total_samples:,}</div>
                    <div style="font-size: 12px; opacity: 0.8;">SAMPLES</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{avg_accuracy:.1%}</div>
                    <div style="font-size: 12px; opacity: 0.8;">AVG ACCURACY</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{len(models_used)}</div>
                    <div style="font-size: 12px; opacity: 0.8;">MODELS</div>
                </div>
            </div>
        </div>
        """
        summary_header.value = html

    # -------------------------------------------------------------------------
    # Run selection widgets
    # -------------------------------------------------------------------------
    run_options = [(f"{r.run_name} ({r.run_id})", r.run_id) for r in available_runs]

    run_selector = widgets.SelectMultiple(
        options=run_options,
        value=[run_options[0][1]] if run_options else [],
        description="",
        layout=widgets.Layout(width="100%", height="150px"),
    )

    # Quick select buttons
    select_all_btn = widgets.Button(description="Select All", button_style="info",
                                     layout=widgets.Layout(width="100px"))
    clear_btn = widgets.Button(description="Clear", button_style="warning",
                               layout=widgets.Layout(width="100px"))

    # -------------------------------------------------------------------------
    # View selector
    # -------------------------------------------------------------------------
    view_selector = widgets.ToggleButtons(
        options=[
            ("Summary", "summary"),
            ("Loss Curves", "loss"),
            ("Confusion Matrix", "cm"),
            ("ROC Curves", "roc"),
            ("Confidence", "confidence"),
            ("Grad-CAM", "gradcam"),
            ("Comparison", "compare"),
        ],
        value="summary",
        layout=widgets.Layout(margin="0 0 10px 0"),
    )

    # -------------------------------------------------------------------------
    # Grad-CAM specific controls
    # -------------------------------------------------------------------------
    gradcam_mode = widgets.ToggleButtons(
        options=[
            ("Individual Sample", "individual"),
            ("Class Averages", "averages"),
        ],
        value="individual",
    )

    sample_slider = widgets.IntSlider(
        value=0, min=0, max=100, step=1,
        description="Sample:",
        continuous_update=False,
        layout=widgets.Layout(width="300px"),
    )

    prev_btn = widgets.Button(description="Prev", button_style="info", icon="arrow-left")
    next_btn = widgets.Button(description="Next", button_style="info", icon="arrow-right")
    random_btn = widgets.Button(description="Random", button_style="warning", icon="random")

    class_filter = widgets.Dropdown(
        options=[("All Classes", "all")],
        value="all",
        description="Class:",
    )

    prediction_filter = widgets.Dropdown(
        options=[("All", "all"), ("Correct Only", "correct"), ("Incorrect Only", "incorrect")],
        value="all",
        description="Filter:",
    )

    # -------------------------------------------------------------------------
    # Output areas
    # -------------------------------------------------------------------------
    out_main = widgets.Output()
    out_table = widgets.Output()

    # -------------------------------------------------------------------------
    # Visualization functions
    # -------------------------------------------------------------------------
    def _show_summary():
        """Show summary cards for selected runs."""
        selected = list(run_selector.value)
        if not selected:
            print("No runs selected. Select one or more runs from the control panel.")
            return

        runs = [_load_run(rid) for rid in selected]

        # Create summary table
        html = """
        <style>
            .run-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            .run-table th, .run-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            .run-table th { background: #34495e; color: white; }
            .run-table tr:hover { background: #f5f5f5; }
            .metric-good { color: #27ae60; font-weight: bold; }
            .metric-bad { color: #e74c3c; }
        </style>
        <table class="run-table">
            <thead>
                <tr>
                    <th>Run Name</th>
                    <th>Model</th>
                    <th>Epochs</th>
                    <th>Accuracy</th>
                    <th>F1 (macro)</th>
                    <th>Best Val Loss</th>
                    <th>Dataset CSV</th>
                    <th>Created</th>
                </tr>
            </thead>
            <tbody>
        """

        summaries = compare_runs_summary(runs)
        best_acc = max(s["accuracy"] for s in summaries)
        best_f1 = max(s["f1_macro"] for s in summaries)

        for s, run in zip(summaries, runs):
            acc_class = "metric-good" if s["accuracy"] == best_acc else ""
            f1_class = "metric-good" if s["f1_macro"] == best_f1 else ""
            dataset_csv = _get_dataset_info(run)
            created = s["created_at"][:10] if s["created_at"] else "N/A"
            best_val_loss_str = f"{s['best_val_loss']:.4f}" if s['best_val_loss'] else "N/A"

            html += f"""
                <tr>
                    <td><b>{s['run_name']}</b></td>
                    <td>{s['model_name']}</td>
                    <td>{s['num_epochs']}</td>
                    <td class="{acc_class}">{s['accuracy']:.4f}</td>
                    <td class="{f1_class}">{s['f1_macro']:.4f}</td>
                    <td>{best_val_loss_str}</td>
                    <td><code>{dataset_csv}</code></td>
                    <td>{created}</td>
                </tr>
            """

        html += "</tbody></table>"
        display(widgets.HTML(html))

        # Show hyperparameters if available
        for run in runs:
            if run.metadata.hyperparameters:
                hp_html = f"<details><summary><b>{run.metadata.run_name} - Hyperparameters</b></summary><ul>"
                for k, v in run.metadata.hyperparameters.items():
                    hp_html += f"<li><b>{k}:</b> {v}</li>"
                hp_html += "</ul></details>"
                display(widgets.HTML(hp_html))

    def _show_loss_curves():
        """Show loss curves for selected runs side by side."""
        selected = list(run_selector.value)
        if not selected:
            print("No runs selected.")
            return

        runs = [_load_run(rid) for rid in selected]
        n_runs = len(runs)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

        for run, color in zip(runs, colors):
            history = run.history
            epochs = range(1, len(history.get("train_loss", [])) + 1)
            label = run.metadata.run_name[:20]

            if history.get("train_loss"):
                axes[0].plot(epochs, history["train_loss"], "-", color=color,
                           label=f"{label} (train)", alpha=0.7)
            if history.get("val_loss"):
                axes[0].plot(epochs, history["val_loss"], "--", color=color,
                           label=f"{label} (val)", linewidth=2)

            if history.get("train_err"):
                axes[1].plot(epochs, history["train_err"], "-", color=color,
                           label=f"{label} (train)", alpha=0.7)
            if history.get("val_err"):
                axes[1].plot(epochs, history["val_err"], "--", color=color,
                           label=f"{label} (val)", linewidth=2)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss", fontweight="bold")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Error Rate")
        axes[1].set_title("Training & Validation Error", fontweight="bold")
        axes[1].legend(fontsize=8, loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _show_confusion_matrices():
        """Show confusion matrices for selected runs."""
        selected = list(run_selector.value)
        if not selected:
            print("No runs selected.")
            return

        runs = [_load_run(rid) for rid in selected]
        n_runs = len(runs)

        n_cols = min(3, n_runs)
        n_rows = (n_runs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_runs == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, run in enumerate(runs):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]

            cm = confusion_matrix(run.labels, run.preds)
            class_names = run.metadata.class_names

            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(class_names, fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            accuracy = (run.preds == run.labels).mean()
            ax.set_title(f"{run.metadata.run_name[:25]}\nAcc: {accuracy:.2%}", fontweight="bold")

            # Annotate
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black", fontsize=9)

        # Hide empty subplots
        for idx in range(n_runs, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()

    def _show_roc_curves():
        """Show ROC curves for selected runs."""
        selected = list(run_selector.value)
        if not selected:
            print("No runs selected.")
            return

        runs = [_load_run(rid) for rid in selected]
        n_runs = len(runs)

        # If single run, show per-class ROC
        # If multiple runs, show macro-average comparison
        if n_runs == 1:
            run = runs[0]
            n_classes = run.probs.shape[1]
            y_true_bin = label_binarize(run.labels, classes=list(range(n_classes)))

            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], run.probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i],
                       label=f"{run.metadata.class_names[i]} (AUC={roc_auc:.3f})")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curves - {run.metadata.run_name}", fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)

        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

            for run, color in zip(runs, colors):
                n_classes = run.probs.shape[1]
                y_true_bin = label_binarize(run.labels, classes=list(range(n_classes)))

                # Compute macro-average
                all_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], run.probs[:, i])
                    mean_tpr += np.interp(all_fpr, fpr, tpr)
                mean_tpr /= n_classes

                macro_auc = auc(all_fpr, mean_tpr)
                ax.plot(all_fpr, mean_tpr, color=color, linewidth=2,
                       label=f"{run.metadata.run_name[:25]} (AUC={macro_auc:.3f})")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Macro-Average ROC Curves Comparison", fontweight="bold")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _show_confidence_distribution():
        """Show confidence distribution for selected runs."""
        selected = list(run_selector.value)
        if not selected:
            print("No runs selected.")
            return

        runs = [_load_run(rid) for rid in selected]
        n_runs = len(runs)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

        # Plot 1: Overall confidence distribution
        for run, color in zip(runs, colors):
            # Get max confidence (predicted class probability) for each sample
            max_conf = run.probs.max(axis=1)
            axes[0, 0].hist(max_conf, bins=20, alpha=0.5, color=color,
                           label=run.metadata.run_name[:20], density=True, edgecolor='black', linewidth=0.5)

        axes[0, 0].set_xlabel("Prediction Confidence")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Overall Confidence Distribution", fontweight="bold")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Random guess")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Confidence for correct vs incorrect predictions
        for run, color in zip(runs, colors):
            max_conf = run.probs.max(axis=1)
            correct_mask = run.preds == run.labels

            correct_conf = max_conf[correct_mask]
            incorrect_conf = max_conf[~correct_mask]

            if len(correct_conf) > 0:
                axes[0, 1].hist(correct_conf, bins=15, alpha=0.4, color=color,
                               label=f"{run.metadata.run_name[:15]} (correct)", density=True,
                               edgecolor='black', linewidth=0.5)
            if len(incorrect_conf) > 0:
                axes[0, 1].hist(incorrect_conf, bins=15, alpha=0.4, color=color,
                               hatch='//', label=f"{run.metadata.run_name[:15]} (incorrect)", density=True,
                               edgecolor='black', linewidth=0.5)

        axes[0, 1].set_xlabel("Prediction Confidence")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_title("Confidence: Correct vs Incorrect", fontweight="bold")
        axes[0, 1].legend(fontsize=7, loc="upper left")
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Mean confidence per class (bar chart)
        if n_runs > 0:
            run = runs[0]  # Use first run for class names
            class_names = run.metadata.class_names
            n_classes = len(class_names)
            x = np.arange(n_classes)
            width = 0.8 / n_runs

            for i, run in enumerate(runs):
                mean_conf_per_class = []
                for c in range(n_classes):
                    mask = run.labels == c
                    if mask.sum() > 0:
                        # Mean confidence for the TRUE class
                        mean_conf_per_class.append(run.probs[mask, c].mean())
                    else:
                        mean_conf_per_class.append(0)

                offset = (i - n_runs/2 + 0.5) * width
                axes[1, 0].bar(x + offset, mean_conf_per_class, width,
                              label=run.metadata.run_name[:20], alpha=0.8)

            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(class_names, rotation=45, ha="right")
            axes[1, 0].set_ylabel("Mean Confidence")
            axes[1, 0].set_title("Mean Confidence for True Class", fontweight="bold")
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].set_ylim(0, 1.1)
            axes[1, 0].axhline(y=1/n_classes, color="gray", linestyle="--", alpha=0.5)
            axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Plot 4: Confidence calibration (reliability diagram)
        for run, color in zip(runs, colors):
            max_conf = run.probs.max(axis=1)
            correct_mask = run.preds == run.labels

            # Bin predictions by confidence
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            bin_accuracies = []
            bin_counts = []

            for j in range(n_bins):
                mask = (max_conf >= bin_edges[j]) & (max_conf < bin_edges[j + 1])
                if mask.sum() > 0:
                    bin_accuracies.append(correct_mask[mask].mean())
                    bin_counts.append(mask.sum())
                else:
                    bin_accuracies.append(np.nan)
                    bin_counts.append(0)

            # Plot reliability diagram
            valid_mask = ~np.isnan(bin_accuracies)
            axes[1, 1].plot(np.array(bin_centers)[valid_mask], np.array(bin_accuracies)[valid_mask],
                           'o-', color=color, label=run.metadata.run_name[:20], linewidth=2, markersize=6)

        # Perfect calibration line
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Perfect calibration")
        axes[1, 1].set_xlabel("Mean Predicted Confidence")
        axes[1, 1].set_ylabel("Fraction of Correct Predictions")
        axes[1, 1].set_title("Calibration Diagram (Reliability)", fontweight="bold")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect('equal')

        plt.tight_layout()
        plt.show()

        # Print calibration statistics
        print("\nCalibration Statistics:")
        print("-" * 60)
        for run in runs:
            max_conf = run.probs.max(axis=1)
            correct_mask = run.preds == run.labels

            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0
            total_samples = len(max_conf)

            for j in range(n_bins):
                mask = (max_conf >= bin_edges[j]) & (max_conf < bin_edges[j + 1])
                if mask.sum() > 0:
                    bin_acc = correct_mask[mask].mean()
                    bin_conf = max_conf[mask].mean()
                    ece += (mask.sum() / total_samples) * abs(bin_acc - bin_conf)

            avg_conf = max_conf.mean()
            accuracy = correct_mask.mean()

            print(f"{run.metadata.run_name[:30]}")
            print(f"  Avg Confidence: {avg_conf:.3f} | Accuracy: {accuracy:.3f} | ECE: {ece:.4f}")

    def _show_gradcam():
        """Show Grad-CAM comparison."""
        selected = list(run_selector.value)
        if not selected:
            print("No runs selected. Select runs to compare Grad-CAM visualizations.")
            return

        runs = [_load_run(rid) for rid in selected]

        # Check if images are available
        runs_with_images = [r for r in runs if r.images is not None]
        if not runs_with_images:
            print("None of the selected runs have saved images.")
            print("Save runs with save_images=True to enable Grad-CAM visualization.")
            return

        mode = gradcam_mode.value

        if mode == "individual":
            _show_gradcam_individual(runs_with_images)
        else:
            _show_gradcam_averages(runs_with_images)

    def _show_gradcam_individual(runs):
        """Show individual sample comparison with Grad-CAM heatmaps."""
        # Use first run to determine sample count
        n_samples = min(len(r.images) for r in runs)
        sample_slider.max = n_samples - 1

        # Apply prediction filter
        filter_val = prediction_filter.value
        if filter_val == "correct":
            valid_indices = [i for i in range(n_samples)
                           if all(r.preds[i] == r.labels[i] for r in runs)]
        elif filter_val == "incorrect":
            valid_indices = [i for i in range(n_samples)
                           if any(r.preds[i] != r.labels[i] for r in runs)]
        else:
            valid_indices = list(range(n_samples))

        if not valid_indices:
            print(f"No samples match filter: {filter_val}")
            return

        # Get current sample
        idx = sample_slider.value
        if idx >= len(valid_indices):
            idx = 0
            sample_slider.value = 0

        sample_idx = valid_indices[idx] if idx < len(valid_indices) else valid_indices[0]

        n_runs = len(runs)
        # Create 2 rows: original images and Grad-CAM overlays
        n_cols = min(3, n_runs)

        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        true_label = runs[0].labels[sample_idx]
        class_names = runs[0].metadata.class_names

        # Track if any model could compute Grad-CAM
        any_gradcam = False

        for run_idx, run in enumerate(runs):
            if run_idx >= n_cols:
                break  # Only show up to n_cols runs

            img = run.images[sample_idx]
            pred = run.preds[sample_idx]
            conf = run.probs[sample_idx][pred]

            # Prepare image for display
            img_display = img.transpose(1, 2, 0)
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

            # Row 0: Original image
            axes[0, run_idx].imshow(img_display)

            is_correct = pred == true_label
            status = "CORRECT" if is_correct else "INCORRECT"
            title_color = "green" if is_correct else "red"

            axes[0, run_idx].set_title(
                f"{run.metadata.run_name[:20]}\n"
                f"Pred: {class_names[pred]} ({conf:.1%})\n"
                f"[{status}]",
                fontsize=10,
                color=title_color,
            )
            axes[0, run_idx].axis("off")

            # Row 1: Grad-CAM overlay
            # Try to compute Grad-CAM if model is available
            model, target_layer = _get_model_and_layer(run)

            if model is not None and target_layer is not None:
                # Compute Grad-CAM
                img_tensor = torch.from_numpy(img).float()
                cam = _compute_gradcam(model, target_layer, img_tensor, target_class=pred)

                if cam is not None:
                    any_gradcam = True
                    # Resize CAM to image size
                    h, w = img_display.shape[:2]
                    cam_resized = _resize_cam(cam, h, w)

                    # Create heatmap overlay
                    heatmap = _cam_to_heatmap(cam_resized)

                    # Blend with original image
                    overlay = 0.5 * img_display + 0.5 * heatmap
                    overlay = np.clip(overlay, 0, 1)

                    axes[1, run_idx].imshow(overlay)
                    axes[1, run_idx].set_title("Grad-CAM Attention", fontsize=10)
                else:
                    axes[1, run_idx].imshow(img_display)
                    axes[1, run_idx].set_title("Grad-CAM failed", fontsize=10, color="orange")
            else:
                # No model available - show message
                axes[1, run_idx].imshow(img_display, alpha=0.5)
                axes[1, run_idx].text(
                    0.5, 0.5, "Model not saved\nwith this run",
                    ha="center", va="center",
                    transform=axes[1, run_idx].transAxes,
                    fontsize=12, color="red",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )
                axes[1, run_idx].set_title("No Grad-CAM available", fontsize=10, color="orange")

            axes[1, run_idx].axis("off")

        # Hide empty columns
        for run_idx in range(n_runs, n_cols):
            axes[0, run_idx].axis("off")
            axes[1, run_idx].axis("off")

        fig.suptitle(
            f"Sample {sample_idx + 1}/{n_samples} | True: {class_names[true_label]}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        if not any_gradcam:
            print("\nNote: To enable Grad-CAM visualization, save training runs with save_model=True")

        # Show probability comparison
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        x = np.arange(len(class_names))
        width = 0.8 / len(runs)

        for i, run in enumerate(runs):
            offset = (i - len(runs)/2 + 0.5) * width
            ax2.bar(x + offset, run.probs[sample_idx], width,
                   label=run.metadata.run_name[:20], alpha=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha="right")
        ax2.set_ylabel("Probability")
        ax2.set_title("Prediction Probabilities Comparison")
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def _show_gradcam_averages(runs):
        """Show average attention comparison across runs."""
        class_names = runs[0].metadata.class_names
        n_classes = len(class_names)

        # Per-class accuracy comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Accuracy per class
        n_runs = len(runs)
        x = np.arange(n_classes)
        width = 0.8 / n_runs

        for i, run in enumerate(runs):
            acc_per_class = []
            for c in range(n_classes):
                mask = run.labels == c
                if mask.sum() > 0:
                    acc_per_class.append((run.preds[mask] == c).mean())
                else:
                    acc_per_class.append(0)

            offset = (i - n_runs/2 + 0.5) * width
            axes[0, 0].bar(x + offset, acc_per_class, width,
                          label=run.metadata.run_name[:20], alpha=0.8)

        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45, ha="right")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Per-Class Accuracy", fontweight="bold")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_ylim(0, 1.1)

        # Average confidence
        for i, run in enumerate(runs):
            conf_per_class = []
            for c in range(n_classes):
                mask = run.labels == c
                if mask.sum() > 0:
                    conf_per_class.append(run.probs[mask, c].mean())
                else:
                    conf_per_class.append(0)

            offset = (i - n_runs/2 + 0.5) * width
            axes[0, 1].bar(x + offset, conf_per_class, width,
                          label=run.metadata.run_name[:20], alpha=0.8)

        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(class_names, rotation=45, ha="right")
        axes[0, 1].set_ylabel("Avg Confidence")
        axes[0, 1].set_title("Average Confidence for True Class", fontweight="bold")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].set_ylim(0, 1.1)

        # Agreement analysis (for 2 runs)
        if n_runs >= 2:
            run1, run2 = runs[0], runs[1]
            agree_mask = run1.preds == run2.preds
            both_correct = ((run1.preds == run1.labels) & agree_mask).sum()
            both_wrong = ((run1.preds != run1.labels) & (run2.preds != run2.labels)).sum()
            only_r1 = ((run1.preds == run1.labels) & (run2.preds != run2.labels)).sum()
            only_r2 = ((run2.preds == run2.labels) & (run1.preds != run1.labels)).sum()

            categories = ["Both\nCorrect", "Both\nWrong", f"Only\n{run1.metadata.run_name[:10]}",
                         f"Only\n{run2.metadata.run_name[:10]}"]
            counts = [both_correct, both_wrong, only_r1, only_r2]
            colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

            axes[1, 0].bar(categories, counts, color=colors)
            axes[1, 0].set_ylabel("Number of Samples")
            axes[1, 0].set_title("Agreement Analysis", fontweight="bold")
            for i, (cat, count) in enumerate(zip(categories, counts)):
                axes[1, 0].text(i, count + 1, str(count), ha="center", fontsize=10)
        else:
            axes[1, 0].text(0.5, 0.5, "Select 2+ runs\nfor agreement analysis",
                           ha="center", va="center", fontsize=12)
            axes[1, 0].axis("off")

        # Summary
        summary_lines = ["COMPARISON SUMMARY", "=" * 40]
        for run in runs:
            acc = (run.preds == run.labels).mean()
            dataset = _get_dataset_info(run)
            summary_lines.append(f"\n{run.metadata.run_name[:25]}")
            summary_lines.append(f"  Model: {run.metadata.model_name}")
            summary_lines.append(f"  Accuracy: {acc:.2%}")
            summary_lines.append(f"  Dataset: {dataset}")
            summary_lines.append(f"  Samples: {len(run.labels)}")

        axes[1, 1].text(0.05, 0.95, "\n".join(summary_lines),
                       transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment="top", fontfamily="monospace",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        axes[1, 1].axis("off")

        fig.suptitle("Model Comparison Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def _show_comparison():
        """Show detailed comparison view."""
        selected = list(run_selector.value)
        if len(selected) < 2:
            print("Select at least 2 runs for comparison.")
            return

        runs = [_load_run(rid) for rid in selected]
        summaries = compare_runs_summary(runs)

        # Find best values
        best_acc = max(s["accuracy"] for s in summaries)
        best_f1 = max(s["f1_macro"] for s in summaries)

        # Comparison chart
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        run_names = [s["run_name"][:15] for s in summaries]
        accuracies = [s["accuracy"] for s in summaries]
        f1_scores = [s["f1_macro"] for s in summaries]

        x = np.arange(len(run_names))

        # Accuracy comparison
        colors = ["#27ae60" if acc == best_acc else "#3498db" for acc in accuracies]
        axes[0].bar(x, accuracies, color=colors)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(run_names, rotation=45, ha="right")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy Comparison", fontweight="bold")
        axes[0].set_ylim(0, 1.1)
        for i, acc in enumerate(accuracies):
            axes[0].text(i, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=9)

        # F1 comparison
        colors = ["#27ae60" if f1 == best_f1 else "#e74c3c" for f1 in f1_scores]
        axes[1].bar(x, f1_scores, color=colors)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(run_names, rotation=45, ha="right")
        axes[1].set_ylabel("F1 Score (macro)")
        axes[1].set_title("F1 Score Comparison", fontweight="bold")
        axes[1].set_ylim(0, 1.1)
        for i, f1 in enumerate(f1_scores):
            axes[1].text(i, f1 + 0.02, f"{f1:.3f}", ha="center", fontsize=9)

        # Loss comparison
        for run in runs:
            if run.history.get("val_loss"):
                axes[2].plot(run.history["val_loss"], label=run.metadata.run_name[:15], linewidth=2)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Validation Loss")
        axes[2].set_title("Validation Loss Over Time", fontweight="bold")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # Main update function
    # -------------------------------------------------------------------------
    def _update(change=None):
        state["selected_run_ids"] = list(run_selector.value)
        dashboard_obj.selected_runs = state["selected_run_ids"]

        _update_header()

        # Update class filter options
        if state["selected_run_ids"]:
            run = _load_run(state["selected_run_ids"][0])
            class_options = [("All Classes", "all")] + [
                (name, str(i)) for i, name in enumerate(run.metadata.class_names)
            ]
            class_filter.options = class_options

        with out_main:
            out_main.clear_output(wait=True)

            view = view_selector.value
            if view == "summary":
                _show_summary()
            elif view == "loss":
                _show_loss_curves()
            elif view == "cm":
                _show_confusion_matrices()
            elif view == "roc":
                _show_roc_curves()
            elif view == "confidence":
                _show_confidence_distribution()
            elif view == "gradcam":
                _show_gradcam()
            elif view == "compare":
                _show_comparison()

    def _on_select_all(btn):
        run_selector.value = [opt[1] for opt in run_options]

    def _on_clear(btn):
        run_selector.value = []

    def _on_prev(btn):
        if sample_slider.value > 0:
            sample_slider.value -= 1

    def _on_next(btn):
        if sample_slider.value < sample_slider.max:
            sample_slider.value += 1

    def _on_random(btn):
        if sample_slider.max > 0:
            sample_slider.value = np.random.randint(0, sample_slider.max + 1)

    # Wire up observers
    run_selector.observe(_update, names="value")
    view_selector.observe(_update, names="value")
    gradcam_mode.observe(_update, names="value")
    sample_slider.observe(_update, names="value")
    prediction_filter.observe(_update, names="value")
    class_filter.observe(_update, names="value")

    select_all_btn.on_click(_on_select_all)
    clear_btn.on_click(_on_clear)
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)
    random_btn.on_click(_on_random)

    # Initial render
    _update()

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    # Control panel
    control_header = widgets.HTML(
        """<div style="background: #34495e; color: white; padding: 12px;
           border-radius: 6px 6px 0 0; font-weight: bold;">
           Control Panel
        </div>"""
    )

    run_section = widgets.HTML(
        """<div style="background: #ecf0f1; padding: 8px 12px; font-weight: bold;
           font-size: 12px; color: #2c3e50;">Training Runs</div>"""
    )

    gradcam_controls = VBox([
        widgets.HTML("<div style='font-weight: bold; margin: 10px 0 5px 0;'>Grad-CAM Options</div>"),
        gradcam_mode,
        HBox([prev_btn, next_btn, random_btn]),
        sample_slider,
        prediction_filter,
    ])

    control_content = VBox([
        run_section,
        run_selector,
        HBox([select_all_btn, clear_btn]),
        widgets.HTML("<hr style='margin: 10px 0;'>"),
        gradcam_controls,
    ], layout=widgets.Layout(padding="10px", background="#f8f9fa"))

    control_panel = VBox([
        control_header,
        control_content,
    ], layout=widgets.Layout(
        width="300px",
        border="1px solid #bdc3c7",
        border_radius="6px",
        margin="0 15px 0 0",
    ))

    # Main content
    main_content = VBox([
        view_selector,
        out_main,
    ], layout=widgets.Layout(flex="1"))

    content_row = HBox([control_panel, main_content])

    dashboard_widget = VBox([
        summary_header,
        content_row,
    ])

    dashboard_obj.widget = dashboard_widget
    return dashboard_obj
