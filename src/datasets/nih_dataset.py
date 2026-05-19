"""PyTorch dataset for prepared NIH ChestX-ray14 multilabel CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only without training deps
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback base so the module can be imported without torch installed."""

        pass

NO_FINDING = "No Finding"
REQUIRED_METADATA_COLUMNS = ["Image Index", "Patient ID", "image_path"]


class NIHMultilabelDataset(Dataset):
    """Load NIH ChestX-ray14 images and 14-label targets from a prepared CSV."""

    def __init__(
        self,
        csv_path: str | Path,
        labels: Sequence[str],
        transform: Any | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.labels = list(labels)
        self.transform = transform

        if torch is None:
            raise ImportError(
                "NIHMultilabelDataset requires torch. Install requirements-train.txt "
                "or use a Kaggle notebook with PyTorch enabled."
            )

        self.data = pd.read_csv(self.csv_path)

        self._validate()

    def _validate(self) -> None:
        if len(self.labels) != 14:
            raise ValueError(f"Expected 14 labels, found {len(self.labels)}.")
        if NO_FINDING in self.labels:
            raise ValueError('"No Finding" must not be used as a model label.')
        if NO_FINDING in self.data.columns:
            raise ValueError('"No Finding" must not be present as a label column.')

        required_columns = [*REQUIRED_METADATA_COLUMNS, *self.labels]
        missing = [column for column in required_columns if column not in self.data.columns]
        if missing:
            raise ValueError(f"{self.csv_path} is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor, dict[str, Any]]:
        row = self.data.iloc[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found for row {index}: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target_values = row[self.labels].to_numpy(dtype="float32")
        target = torch.tensor(target_values, dtype=torch.float32)

        metadata = {
            "image_path": str(image_path),
            "image_index": row.get("Image Index"),
            "patient_id": row.get("Patient ID"),
        }
        if "Finding Labels" in row.index:
            metadata["finding_labels"] = row["Finding Labels"]
        return image, target, metadata
