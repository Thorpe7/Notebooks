""" Pytorch utility script to load data from XNAT data hierarchy into pytorch dataset """
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


def find_dicom_paths_from_dataframe(
    df: pd.DataFrame,
    base_dir: str = "/data",
    label_column: str = "ground_truth",
) -> tuple[list[str], list[int]]:
    """
    Extract DICOM file paths and labels from a metadata DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: project_id, experiment_label, and label_column
        The label_column (default: 'ground_truth') contains the class labels.
    base_dir : str
        Base directory for XNAT data mount (default: '/data')
    label_column : str
        Name of the column containing ground truth labels (default: 'ground_truth')

    Returns
    -------
    tuple[list[str], list[int]]
        List of DICOM file paths and corresponding labels
    """
    paths, labels = [], []

    # Check if label column exists
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    for _, row in df.iterrows():
        project_id = row.get('project_id')
        experiment_label = row.get('experiment_label')
        label = row.get(label_column)

        if pd.isna(project_id) or pd.isna(experiment_label):
            continue

        # Skip rows without a valid label
        if pd.isna(label):
            continue

        # Convert label to int
        try:
            label = int(label)
        except (ValueError, TypeError):
            continue

        # Build path to experiment directory and find DICOM files
        exp_path = Path(base_dir) / "projects" / project_id / "experiments" / experiment_label / "SCANS"

        if not exp_path.exists():
            continue

        # Find all DICOM files in secondary folders
        for dcm in exp_path.glob("*/secondary/*.dcm"):
            paths.append(str(dcm))
            labels.append(label)

    if not paths:
        raise RuntimeError(
            f"No DICOMs found from DataFrame entries in {base_dir}. "
            f"Check that '{label_column}' column has valid labels and paths exist."
        )

    return paths, labels


def find_dicom_paths_with_labels(
    proj_id: str,
    base_dir: str = "/data",
    ground_truth_filename: str = "ground_truth.csv",
) -> tuple[list[str], list[int]]:
    """
    Find DICOM paths and labels for a project using filesystem and ground truth CSV.

    Parameters
    ----------
    proj_id : str
        XNAT project ID
    base_dir : str
        Base directory for XNAT data mount (default: '/data')
    ground_truth_filename : str
        Name of the ground truth CSV file in project resources

    Returns
    -------
    tuple[list[str], list[int]]
        List of DICOM file paths and corresponding labels
    """
    project_path = Path(base_dir) / "projects" / proj_id
    experiments_path = project_path / "experiments"

    # Look for ground truth CSV in project resources
    gt_csv_path = None
    resources_path = project_path / "resources"
    if resources_path.exists():
        for csv_file in resources_path.glob(f"**/{ground_truth_filename}"):
            gt_csv_path = csv_file
            break

    if gt_csv_path is None:
        raise FileNotFoundError(
            f"Ground truth CSV '{ground_truth_filename}' not found in project resources at {resources_path}. "
            "Please ensure the CSV is uploaded to the project-level resources in XNAT."
        )

    # Load ground truth CSV
    gt_df = pd.read_csv(gt_csv_path)

    # Build lookup dict: (subject, experiment, scan_name) -> ground_truth
    # The CSV has columns: project, subject, experiment, scan_name, ground_truth
    label_lookup = {}
    for _, row in gt_df.iterrows():
        key = (
            str(row.get("subject", "")),
            str(row.get("experiment", "")),
            str(row.get("scan_name", "")),
        )
        label = row.get("ground_truth")
        if pd.notna(label):
            try:
                label_lookup[key] = int(label)
            except (ValueError, TypeError):
                pass

    # Find DICOM files and match to labels
    paths, labels = [], []
    for dcm in experiments_path.glob("*/SCANS/*/secondary/*.dcm"):
        # Extract experiment label from path
        experiment_label = dcm.parents[3].name

        # Extract subject label (may be part of experiment label or need lookup)
        # For now, try to match on experiment label
        scan_dir = dcm.parents[1].name
        scan_type = scan_dir.split("_", 1)[0] if "_" in scan_dir else scan_dir

        # Try different key combinations to find a match
        matched_label = None
        for (subj, exp, scan_name), label in label_lookup.items():
            if exp == experiment_label:
                matched_label = label
                break

        if matched_label is not None:
            paths.append(str(dcm))
            labels.append(matched_label)

    if not paths:
        raise RuntimeError(
            f"No DICOMs found with matching labels under {experiments_path}. "
            f"Ground truth CSV has {len(label_lookup)} entries."
        )

    return paths, labels

def dicom_to_numpy(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = arr.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    p1, p99 = np.percentile(arr, (1, 99))
    if p99 > p1:
        arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
    else:
        mn, mx = float(arr.min()), float(arr.max())
        arr = (arr - mn) / (mx - mn + 1e-8)
    return arr

class DICOMDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None,
                 classes: list[str] | None = None, label2id: Dict[int,int] | None = None, id2label: Dict[int,int] | None = None):
        self.paths = paths
        self.labels = labels  # MUST be contiguous 0..K-1 for CE loss
        self.transform = transform
        self.classes = classes if classes is not None else [f"CLASS_{i}" for i in sorted(set(labels))]
        self.label2id = label2id or {}
        self.id2label = id2label or {}

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        x = dicom_to_numpy(self.paths[idx])      # HxW in [0,1]
        x = np.expand_dims(x, 0)                 # 1xHxW
        x = np.repeat(x, 3, axis=0)              # 3xHxW (optional if using 1-ch models)
        x = torch.from_numpy(x)                  # float32 tensor
        y = int(self.labels[idx])                # 0..K-1
        if self.transform:
            x = self.transform(x)
        return x, y

def _make_class_names(unique_labels: list[int], prefix: str = "CLASS") -> list[str]:
    """Generate human-readable class names from unique label values."""
    return [f"{prefix}_{label}" for label in unique_labels]

def load_dicom_data_with_logging(
    data_source: Union[str, pd.DataFrame],  # project id, CSV path, or DataFrame
    batch_size: int = 32,
    num_workers: int = 2,
    val_split: float = 0.2,
    image_size: int = 224,   # target image size (height and width)
    base_dir: str = "/data",  # base directory for XNAT data mount
    label_column: str = "ground_truth",  # column containing labels
    class_prefix: str = "CLASS",  # prefix for class names (e.g., "BIRADS" -> "BIRADS_1")
) -> Dict[str, DataLoader]:
    """
    Load DICOM data into PyTorch DataLoaders with stratified train/val split.

    Parameters
    ----------
    data_source : str or pd.DataFrame
        One of:
        - Project ID string (e.g., "InBreastProject") - scans mounted directory
        - Path to CSV file containing filtered metadata
        - DataFrame with columns: project_id, experiment_label, and label_column
    batch_size : int
        Batch size for DataLoaders
    num_workers : int
        Number of worker processes for data loading
    val_split : float
        Fraction of data to use for validation
    image_size : int
        Target image size (height and width)
    base_dir : str
        Base directory for XNAT data mount (default: '/data')
    label_column : str
        Name of the column containing ground truth labels (default: 'ground_truth')
    class_prefix : str
        Prefix for human-readable class names (default: 'CLASS').
        E.g., 'BIRADS' produces class names like 'BIRADS_1', 'BIRADS_2', etc.

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary with 'train' and 'val' DataLoaders
    """
    log_dir = Path.cwd() / "logs" / "data_loading"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "data_loading.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 1) Discover files + labels based on data_source type
    if isinstance(data_source, pd.DataFrame):
        # Use DataFrame directly
        logging.info(f"Loading DICOM paths from DataFrame with {len(data_source)} rows")
        paths, targets = find_dicom_paths_from_dataframe(
            data_source, base_dir=base_dir, label_column=label_column
        )
    elif isinstance(data_source, str) and data_source.endswith('.csv'):
        # Load from CSV file
        logging.info(f"Loading DICOM paths from CSV: {data_source}")
        df = pd.read_csv(data_source)
        paths, targets = find_dicom_paths_from_dataframe(
            df, base_dir=base_dir, label_column=label_column
        )
    else:
        # Assume it's a project ID (original behavior)
        logging.info(f"Loading DICOM paths from project: {data_source}")
        paths, targets = find_dicom_paths_with_labels(data_source, base_dir=base_dir)

    indices = np.arange(len(paths))

    # 2) Build mapping labels -> contiguous ids (0..K-1)
    unique_labels_sorted = sorted(set(targets))   # e.g., [1,2,3,4,5,6]
    label2id = {lbl: i for i, lbl in enumerate(unique_labels_sorted)}  # 1->0, 2->1, ...
    id2label = {i: lbl for lbl, i in label2id.items()}
    targets_remap = [label2id[y] for y in targets]  # now 0..K-1

    # 3) Transforms with data augmentation for training
    # Augmentations help prevent overfitting on small medical imaging datasets
    # Resize slightly larger than target for random crop
    resize_size = int(image_size * 1.15)
    train_tf = T.Compose([
        T.Resize((resize_size, resize_size), antialias=True),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 4) Stratified split (use original labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(X=indices, y=targets))

    # 5) Human-readable class names
    class_names = _make_class_names(unique_labels_sorted, prefix=class_prefix)

    # 6) Datasets with REMAPPED labels
    train_ds = DICOMDataset([paths[i] for i in train_idx],
                            [targets_remap[i] for i in train_idx],
                            transform=train_tf,
                            classes=class_names,
                            label2id=label2id, id2label=id2label)
    val_ds   = DICOMDataset([paths[i] for i in val_idx],
                            [targets_remap[i] for i in val_idx],
                            transform=val_tf,
                            classes=class_names,
                            label2id=label2id, id2label=id2label)

    logging.info(f"Stratified split complete. train={len(train_ds)}, val={len(val_ds)}")
    logging.info(f"Classes: {unique_labels_sorted} (prefix: {class_prefix})")
    logging.info(f"Mapping label2id: {label2id}")

    # 7) Weighted sampler (use remapped labels)
    from collections import Counter
    cnt = Counter(train_ds.labels)                   # keys are 0..K-1
    class_weights = {k: 1.0/v for k, v in cnt.items()}
    train_weights = np.array([class_weights[y] for y in train_ds.labels], dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_weights),
        num_samples=len(train_weights),
        replacement=True
    )

    # 8) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return {"train": train_loader, "val": val_loader}

if __name__ == "__main__":
    loaders = load_dicom_data_with_logging("00001")  # project id
    images, labels = next(iter(loaders["train"]))
    print("Images shape:", images.shape)
    print("Labels (0..K-1):", labels)
    print("Classes (names):", loaders["train"].dataset.classes)
    ds = loaders["train"].dataset
    print("label2id:", ds.label2id)
    print("id2label:", ds.id2label)
