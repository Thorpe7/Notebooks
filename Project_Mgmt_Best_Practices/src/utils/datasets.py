"""Dataset and DataLoader utilities for model-ready tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class XnatImageDataset(Dataset):
    """Generic image dataset built from a manifest DataFrame.

    Expected columns:
    - 'filepath': path to image file on disk
    - 'label': integer class index
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

        if "filepath" not in self.manifest or "label" not in self.manifest:
            raise ValueError("Manifest must contain 'filepath' and 'label' columns.")

        if "class_name" in self.manifest:
            self.classes = sorted(self.manifest["class_name"].unique().tolist())
        else:
            self.classes = sorted(self.manifest["label"].unique().tolist())

        self.labels = self.manifest["label"].tolist()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        img_path = Path(row["filepath"])
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def build_dataloaders(
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
):
    """Construct PyTorch dataloaders for train/val splits."""
    train_ds = XnatImageDataset(train_manifest, transform=train_transform)
    val_ds = XnatImageDataset(val_manifest, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"train": train_loader, "val": val_loader}
