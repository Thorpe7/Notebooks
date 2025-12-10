"""
Training run persistence and management.

Provides utilities for saving, loading, and comparing training runs.
Each run captures all data needed to recreate evaluation dashboards.
"""

import json
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class TrainingRunMetadata:
    """Metadata about a training run."""
    run_id: str
    run_name: str
    created_at: str
    model_name: str
    num_epochs: int
    num_classes: int
    class_names: List[str]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class TrainingRunData:
    """
    Complete data from a training run for evaluation dashboards.

    Attributes
    ----------
    metadata : TrainingRunMetadata
        Run identification and configuration info
    history : dict
        Training history with keys: train_loss, val_loss, train_err, val_err
    labels : np.ndarray
        Ground truth labels for validation set
    preds : np.ndarray
        Model predictions for validation set
    probs : np.ndarray
        Prediction probabilities (n_samples, n_classes)
    images : np.ndarray, optional
        Validation images for Grad-CAM (can be large, optional to save)
    model_state_dict : dict, optional
        Model weights (optional, for reloading model)
    """
    metadata: TrainingRunMetadata
    history: Dict[str, List[float]]
    labels: np.ndarray
    preds: np.ndarray
    probs: np.ndarray
    images: Optional[np.ndarray] = None
    model_state_dict: Optional[Dict] = None


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_training_run(
    run_name: str,
    model_name: str,
    history: Dict[str, List[float]],
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    images: Optional[np.ndarray] = None,
    model_state_dict: Optional[Dict] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> TrainingRunData:
    """
    Create a TrainingRunData object from training results.

    Parameters
    ----------
    run_name : str
        Human-readable name for this run (e.g., "EfficientNet-B0 with augmentation")
    model_name : str
        Model architecture name (e.g., "efficientnet_b0")
    history : dict
        Training history dictionary
    labels : np.ndarray
        Ground truth validation labels
    preds : np.ndarray
        Model predictions
    probs : np.ndarray
        Prediction probabilities
    class_names : list of str
        Class names
    images : np.ndarray, optional
        Validation images (warning: can be large)
    model_state_dict : dict, optional
        Model state dict for reloading
    hyperparameters : dict, optional
        Hyperparameters used (lr, batch_size, etc.)
    notes : str, optional
        Additional notes about this run

    Returns
    -------
    TrainingRunData
    """
    run_id = generate_run_id()

    metadata = TrainingRunMetadata(
        run_id=run_id,
        run_name=run_name,
        created_at=datetime.now().isoformat(),
        model_name=model_name,
        num_epochs=len(history.get("train_loss", [])),
        num_classes=len(class_names),
        class_names=list(class_names),
        hyperparameters=hyperparameters or {},
        notes=notes,
    )

    return TrainingRunData(
        metadata=metadata,
        history=history,
        labels=np.asarray(labels),
        preds=np.asarray(preds),
        probs=np.asarray(probs),
        images=np.asarray(images) if images is not None else None,
        model_state_dict=model_state_dict,
    )


def save_training_run(
    run: TrainingRunData,
    save_dir: str = "training_runs",
    save_images: bool = False,
    save_model: bool = False,
) -> Path:
    """
    Save a training run to disk.

    Parameters
    ----------
    run : TrainingRunData
        The training run to save
    save_dir : str
        Directory to save runs in
    save_images : bool
        Whether to save validation images (can be large)
    save_model : bool
        Whether to save model state dict

    Returns
    -------
    Path
        Path to the saved run directory
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    run_dir = save_path / run.metadata.run_id
    run_dir.mkdir(exist_ok=True)

    # Save metadata as JSON (human-readable)
    metadata_dict = asdict(run.metadata)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=2)

    # Save history as JSON
    with open(run_dir / "history.json", "w") as f:
        json.dump(run.history, f, indent=2)

    # Save arrays as numpy files
    np.save(run_dir / "labels.npy", run.labels)
    np.save(run_dir / "preds.npy", run.preds)
    np.save(run_dir / "probs.npy", run.probs)

    # Optionally save images
    if save_images and run.images is not None:
        np.save(run_dir / "images.npy", run.images)

    # Optionally save model state
    if save_model and run.model_state_dict is not None:
        with open(run_dir / "model_state.pkl", "wb") as f:
            pickle.dump(run.model_state_dict, f)

    print(f"Saved training run '{run.metadata.run_name}' to {run_dir}")
    return run_dir


def load_training_run(run_path: str) -> TrainingRunData:
    """
    Load a training run from disk.

    Parameters
    ----------
    run_path : str
        Path to the run directory

    Returns
    -------
    TrainingRunData
    """
    run_dir = Path(run_path)

    # Load metadata
    with open(run_dir / "metadata.json", "r") as f:
        metadata_dict = json.load(f)
    metadata = TrainingRunMetadata(**metadata_dict)

    # Load history
    with open(run_dir / "history.json", "r") as f:
        history = json.load(f)

    # Load arrays
    labels = np.load(run_dir / "labels.npy")
    preds = np.load(run_dir / "preds.npy")
    probs = np.load(run_dir / "probs.npy")

    # Optionally load images
    images_path = run_dir / "images.npy"
    images = np.load(images_path) if images_path.exists() else None

    # Optionally load model state
    model_path = run_dir / "model_state.pkl"
    model_state_dict = None
    if model_path.exists():
        with open(model_path, "rb") as f:
            model_state_dict = pickle.load(f)

    return TrainingRunData(
        metadata=metadata,
        history=history,
        labels=labels,
        preds=preds,
        probs=probs,
        images=images,
        model_state_dict=model_state_dict,
    )


def list_training_runs(runs_dir: str = "training_runs") -> List[TrainingRunMetadata]:
    """
    List all saved training runs.

    Parameters
    ----------
    runs_dir : str
        Directory containing saved runs

    Returns
    -------
    List of TrainingRunMetadata
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []

    runs = []
    for run_dir in sorted(runs_path.iterdir()):
        if run_dir.is_dir():
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata_dict = json.load(f)
                runs.append(TrainingRunMetadata(**metadata_dict))

    return runs


def delete_training_run(run_id: str, runs_dir: str = "training_runs") -> bool:
    """
    Delete a saved training run.

    Parameters
    ----------
    run_id : str
        Run ID to delete
    runs_dir : str
        Directory containing saved runs

    Returns
    -------
    bool
        True if deleted, False if not found
    """
    import shutil
    run_path = Path(runs_dir) / run_id
    if run_path.exists():
        shutil.rmtree(run_path)
        print(f"Deleted run {run_id}")
        return True
    return False


def compare_runs_summary(runs: List[TrainingRunData]) -> Dict[str, Any]:
    """
    Generate a comparison summary of multiple runs.

    Parameters
    ----------
    runs : list of TrainingRunData
        Runs to compare

    Returns
    -------
    dict
        Summary statistics for each run
    """
    from sklearn.metrics import accuracy_score, f1_score

    summaries = []
    for run in runs:
        accuracy = accuracy_score(run.labels, run.preds)
        f1_macro = f1_score(run.labels, run.preds, average="macro", zero_division=0)

        final_train_loss = run.history["train_loss"][-1] if run.history.get("train_loss") else None
        final_val_loss = run.history["val_loss"][-1] if run.history.get("val_loss") else None
        best_val_loss = min(run.history["val_loss"]) if run.history.get("val_loss") else None

        summaries.append({
            "run_id": run.metadata.run_id,
            "run_name": run.metadata.run_name,
            "model_name": run.metadata.model_name,
            "num_epochs": run.metadata.num_epochs,
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "created_at": run.metadata.created_at,
        })

    return summaries
