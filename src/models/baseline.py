"""DenseNet121 baseline for NIH ChestX-ray14 multi-label classification."""

from __future__ import annotations


def build_densenet121_baseline(num_labels: int = 14, pretrained: bool = True):
    """Build a DenseNet121 model that returns raw logits for 14 labels."""
    try:
        from torch import nn
        from torchvision.models import densenet121
    except ModuleNotFoundError as exc:
        raise ImportError(
            "build_densenet121_baseline requires torch and torchvision. "
            "Install requirements-train.txt or run in Kaggle with PyTorch enabled."
        ) from exc

    try:
        from torchvision.models import DenseNet121_Weights

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
    except (ImportError, TypeError):
        model = densenet121(pretrained=pretrained)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
    return model
