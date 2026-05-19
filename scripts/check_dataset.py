"""Validate NIHMultilabelDataset and DataLoader output shapes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.nih_dataset import NIHMultilabelDataset

NO_FINDING = "No Finding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that a prepared NIH split CSV feeds a PyTorch DataLoader."
    )
    parser.add_argument(
        "--csv-path",
        default="data/splits/train.csv",
        help="Path to train.csv, val.csv or test.csv.",
    )
    parser.add_argument(
        "--labels-json",
        default="artifacts/labels.json",
        help="Path to artifacts/labels.json.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
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


def require_training_dependencies() -> tuple[object, object]:
    try:
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ImportError(
            "check_dataset.py requires torch and torchvision. Install "
            "requirements-train.txt or run this in Kaggle with PyTorch enabled."
        ) from exc
    return DataLoader, transforms


def print_positive_counts(dataset: NIHMultilabelDataset, labels: list[str]) -> None:
    print("Positive count by label:")
    for label in labels:
        print(f"  {label}: {int(dataset.data[label].sum())}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    labels_json = Path(args.labels_json)

    labels = load_labels(labels_json)
    DataLoader, transforms = require_training_dependencies()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = NIHMultilabelDataset(
        csv_path=csv_path,
        labels=labels,
        transform=transform,
    )
    if len(dataset) == 0:
        raise ValueError(f"{csv_path} contains no rows.")
    if NO_FINDING in dataset.data.columns:
        raise ValueError('"No Finding" must not be present as a label column.')

    image, target, metadata = dataset[0]
    if list(target.shape) != [14]:
        raise ValueError(f"Expected sample target shape [14], found {list(target.shape)}.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch_images, batch_targets, batch_metadata = next(iter(loader))
    expected_batch = min(args.batch_size, len(dataset))

    if list(batch_images.shape) != [expected_batch, 3, 224, 224]:
        raise ValueError(
            "Expected batch images shape "
            f"[{expected_batch}, 3, 224, 224], found {list(batch_images.shape)}."
        )
    if list(batch_targets.shape) != [expected_batch, 14]:
        raise ValueError(
            f"Expected batch targets shape [{expected_batch}, 14], "
            f"found {list(batch_targets.shape)}."
        )

    print("NIH PyTorch dataset check")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Sample image shape: {list(image.shape)}")
    print(f"Sample target shape: {list(target.shape)}")
    print(f"Sample metadata: {metadata}")
    print(f"Batch images shape: {list(batch_images.shape)}")
    print(f"Batch targets shape: {list(batch_targets.shape)}")
    print(f"Batch metadata keys: {list(batch_metadata.keys())}")
    print_positive_counts(dataset, labels)
    print('OK: "No Finding" is absent from labels and CSV columns.')


if __name__ == "__main__":
    main()
