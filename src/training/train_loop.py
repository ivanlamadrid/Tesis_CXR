"""Simple PyTorch train and validation loops."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.metrics import compute_multilabel_metrics, sigmoid_predictions


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, targets, _metadata in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader produced no samples.")
    return total_loss / total_samples


def evaluate_one_epoch(model, dataloader, criterion, device) -> dict[str, Any]:
    """Evaluate the model and return loss, arrays and multi-label metrics."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError("evaluate_one_epoch requires torch.") from exc

    model.eval()
    total_loss = 0.0
    total_samples = 0
    true_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets, _metadata in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            true_batches.append(targets.detach().cpu().numpy())
            prob_batches.append(sigmoid_predictions(logits))

    if total_samples == 0:
        raise ValueError("Validation dataloader produced no samples.")

    y_true = np.concatenate(true_batches, axis=0)
    y_prob = np.concatenate(prob_batches, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_prob)

    return {
        "val_loss": total_loss / total_samples,
        "y_true": y_true,
        "y_prob": y_prob,
        "metrics": metrics,
    }
