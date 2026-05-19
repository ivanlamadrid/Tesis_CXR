"""Loss factories for multi-label training."""

from __future__ import annotations


def build_multilabel_loss(pos_weight=None):
    """Return BCEWithLogitsLoss for multi-label classification."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ImportError(
            "build_multilabel_loss requires torch. Install requirements-train.txt "
            "or run in Kaggle with PyTorch enabled."
        ) from exc

    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
