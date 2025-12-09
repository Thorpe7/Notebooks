"""Training utilities with basic logging."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    logger,
) -> Dict[str, list]:
    """Generic training loop."""
    history = {"train_loss": [], "val_loss": [], "train_err": [], "val_err": []}

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb in dataloaders["train"]:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += yb.size(0)

        epoch_train_loss = running_loss / running_total
        epoch_train_err = 1.0 - (running_correct / running_total)

        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_running_total = 0

        with torch.no_grad():
            for xb, yb in dataloaders["val"]:
                xb = xb.to(device)
                yb = yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)

                val_running_loss += loss.item() * xb.size(0)
                preds = outputs.argmax(dim=1)
                val_running_correct += (preds == yb).sum().item()
                val_running_total += yb.size(0)

        epoch_val_loss = val_running_loss / val_running_total
        epoch_val_err = 1.0 - (val_running_correct / val_running_total)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_err"].append(epoch_train_err)
        history["val_err"].append(epoch_val_err)

        if scheduler is not None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}, "
            f"train_err={epoch_train_err:.4f}, val_err={epoch_val_err:.4f}"
        )

    return history
