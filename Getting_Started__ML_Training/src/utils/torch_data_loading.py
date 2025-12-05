""" Pytorch utility script to load data from XNAT data hierarchy into pytorch dataset """
import os, re, logging
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

_BIRADS_RE = re.compile(r'BIRADS[_-]?(\d+)', re.IGNORECASE)

def find_dicom_paths_with_labels(proj_id: str, base_dir: str = "/data") -> tuple[list[str], list[int]]:
    base = Path(base_dir) / "projects" / proj_id / "experiments"
    paths, labels = [], []
    for dcm in base.glob("*/SCANS/*/secondary/*.dcm"):
        subj = dcm.parents[3].name  # experiments/<subject_BIRADS_x_UID>
        m = _BIRADS_RE.search(subj)
        if not m:
            continue
        label = int(m.group(1))  # e.g., 1..6
        paths.append(str(dcm))
        labels.append(label)
    if not paths:
        raise RuntimeError(f"No DICOMs found under {base}")
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

def _make_class_names_from_birads(unique_birads: list[int]) -> list[str]:
    return [f"BIRADS_{l}" for l in unique_birads]

def load_dicom_data_with_logging(
    data_root: str,          # project id like "00001" (since find_* builds the full path)
    batch_size: int = 32,
    num_workers: int = 2,
    val_split: float = 0.2,
    image_size: int = 224,   # target image size (height and width)
) -> Dict[str, DataLoader]:
    log_dir = Path.cwd() / "logs" / "data_loading"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "data_loading.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 1) Discover files + BI-RADS labels (e.g., [1,4,5,6])
    paths, targets_birads = find_dicom_paths_with_labels(data_root)
    indices = np.arange(len(paths))

    # 2) Build mapping BI-RADS -> contiguous ids (0..K-1)
    unique_birads_sorted = sorted(set(targets_birads))   # e.g., [1,2,3,4,5,6]
    label2id = {lbl: i for i, lbl in enumerate(unique_birads_sorted)}  # 1->0, 2->1, ...
    id2label = {i: lbl for lbl, i in label2id.items()}
    targets_remap = [label2id[y] for y in targets_birads]  # now 0..K-1

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

    # 4) Stratified split (use original BI-RADS or remapped—either is fine)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(X=indices, y=targets_birads))

    # 5) Human-readable class names (keep BI-RADS wording)
    class_names = _make_class_names_from_birads(unique_birads_sorted)

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
    logging.info(f"Classes (BI-RADS): {unique_birads_sorted}")
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
