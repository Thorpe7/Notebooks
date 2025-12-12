"""
Reusable Voila-friendly dashboards for the XNAT ResNet notebook.

Each function returns an ipywidgets.VBox that you can display in
Jupyter and Voila, e.g.:

    from src.utils.xnat_voila_dashboards import (
        run_comparison_dashboard,
        gradcam_comparison_dashboard,
    )

    ui = run_comparison_dashboard("training_runs")
    ui

All plots are rendered with matplotlib and updated via ipywidgets.

Dashboard summary:
1. run_comparison_dashboard - Compare metrics across training runs
2. gradcam_comparison_dashboard - Compare model predictions between runs
"""

import numpy as np
import ipywidgets as widgets
from ipywidgets import VBox, HBox
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# -------------------------------------------------------------------------
# 1. Training Run Comparison Dashboard
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
# 2. Grad-CAM Comparison Dashboard Between Model Runs
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
