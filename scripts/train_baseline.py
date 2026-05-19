"""Train a DenseNet121 baseline for NIH ChestX-ray14 multi-label classification."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NO_FINDING = "No Finding"


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenseNet121 baseline.")
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--val-csv", default="data/splits/val.csv")
    parser.add_argument("--labels-json", default="artifacts/labels.json")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_labels(labels_json: Path) -> list[str]:
    with labels_json.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"{labels_json} must contain a JSON list of labels.")
    if len(labels) != 14:
        raise ValueError(f"Expected 14 labels, found {len(labels)}.")
    if NO_FINDING in labels:
        raise ValueError('"No Finding" must not be present in labels.json.')
    return labels


def require_training_dependencies():
    try:
        import numpy as np
        import pandas as pd
        import torch
        import sklearn.metrics  # noqa: F401
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ImportError(
            "train_baseline.py requires torch, torchvision, pandas, numpy and "
            "scikit-learn. Install requirements-train.txt or run in Kaggle."
        ) from exc
    return np, pd, torch, DataLoader, transforms


def set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str, torch_module):
    if device_arg == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(device_arg)


def build_transforms(image_size: int, transforms_module):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = transforms_module.Compose(
        [
            transforms_module.Resize((image_size, image_size)),
            transforms_module.RandomHorizontalFlip(),
            transforms_module.ToTensor(),
            transforms_module.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    val_transform = transforms_module.Compose(
        [
            transforms_module.Resize((image_size, image_size)),
            transforms_module.ToTensor(),
            transforms_module.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, val_transform


def metric_or_none(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def is_better_epoch(
    current_metrics: dict[str, Any],
    current_val_loss: float,
    best_macro_auc: float | None,
    best_val_loss: float | None,
) -> bool:
    current_macro_auc = metric_or_none(current_metrics, "macro_auc")
    if current_macro_auc is not None:
        return best_macro_auc is None or current_macro_auc > best_macro_auc
    return best_macro_auc is None and (
        best_val_loss is None or current_val_loss < best_val_loss
    )


def format_metric(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def save_history(history: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "macro_auc", "micro_auc", "f1_macro"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    np, _pd, torch, DataLoader, transforms = require_training_dependencies()

    from src.datasets.nih_dataset import NIHMultilabelDataset
    from src.models.baseline import build_densenet121_baseline
    from src.training.losses import build_multilabel_loss
    from src.training.train_loop import evaluate_one_epoch, train_one_epoch

    labels = load_labels(Path(args.labels_json))
    set_seed(args.seed, torch)
    device = resolve_device(args.device, torch)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform, val_transform = build_transforms(args.image_size, transforms)
    train_dataset = NIHMultilabelDataset(args.train_csv, labels=labels, transform=train_transform)
    val_dataset = NIHMultilabelDataset(args.val_csv, labels=labels, transform=val_transform)

    if NO_FINDING in train_dataset.data.columns or NO_FINDING in val_dataset.data.columns:
        raise ValueError('"No Finding" must not be present as a label column.')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_densenet121_baseline(num_labels=len(labels), pretrained=args.pretrained)
    model.to(device)
    criterion = build_multilabel_loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_auc: float | None = None
    best_val_loss: float | None = None
    best_metrics: dict[str, Any] | None = None
    best_model_path = output_dir / "baseline_densenet121_best.pt"

    print("DenseNet121 baseline training")
    print(f"Train rows: {len(train_dataset)}")
    print(f"Val rows: {len(val_dataset)}")
    print(f"Labels: {len(labels)}")
    print(f"Device: {device}")
    print(f"Pretrained ImageNet: {args.pretrained}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_result = evaluate_one_epoch(model, val_loader, criterion, device)
        val_loss = float(eval_result["val_loss"])
        metrics = eval_result["metrics"]

        macro_auc = metric_or_none(metrics, "macro_auc")
        micro_auc = metric_or_none(metrics, "micro_auc")
        f1_macro = metric_or_none(metrics, "f1_macro")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "macro_auc": macro_auc,
            "micro_auc": micro_auc,
            "f1_macro": f1_macro,
        }
        history.append(row)

        print(
            "epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            "macro_auc={macro_auc} micro_auc={micro_auc} f1_macro={f1_macro}".format(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                macro_auc=format_metric(macro_auc),
                micro_auc=format_metric(micro_auc),
                f1_macro=format_metric(f1_macro),
            )
        )

        if is_better_epoch(metrics, val_loss, best_macro_auc, best_val_loss):
            best_epoch = epoch
            best_macro_auc = macro_auc
            best_val_loss = val_loss
            best_metrics = metrics
            torch.save(
                {
                    "model_name": "baseline_densenet121",
                    "model_state_dict": model.state_dict(),
                    "labels": labels,
                    "num_labels": len(labels),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "metrics": metrics,
                    "args": vars(args),
                },
                best_model_path,
            )

    history_path = output_dir / "baseline_training_history.csv"
    save_history(history, history_path)

    metrics_payload = {
        "model_name": "baseline_densenet121",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_macro_auc": best_macro_auc,
        "labels": labels,
        "history": history,
        "best_metrics": best_metrics,
        "output_files": {
            "best_model": str(best_model_path),
            "history_csv": str(history_path),
        },
    }
    metrics_path = output_dir / "baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    print(f"Saved best model: {best_model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
