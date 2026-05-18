"""Reproducibility helpers."""

from __future__ import annotations

import random


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy and PyTorch when available."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

