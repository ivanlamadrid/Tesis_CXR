"""Create patient-level train/validation/test splits for NIH ChestX-ray14."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test CSV files without patient leakage."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/nih_multilabel.csv",
        help="Path to the prepared NIH multilabel CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Directory where train.csv, val.csv and test.csv will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_inputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if "Patient ID" not in df.columns:
        raise ValueError('Input CSV must contain "Patient ID".')

    missing_labels = [label for label in LABELS if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Input CSV is missing label columns: {missing_labels}")

    if "No Finding" in df.columns:
        raise ValueError('"No Finding" must not be used as a model label column.')

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0.")


def split_patient_ids(
    patient_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(patient_ids, dtype=str)
    rng.shuffle(shuffled)

    n_patients = len(shuffled)
    n_train = int(round(n_patients * train_ratio))
    n_val = int(round(n_patients * val_ratio))

    if n_train + n_val > n_patients:
        n_val = max(0, n_patients - n_train)

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    test_ids = set(shuffled[n_train + n_val :])
    return train_ids, val_ids, test_ids


def subset_by_patient(df: pd.DataFrame, patient_ids: set[str]) -> pd.DataFrame:
    patient_series = df["Patient ID"].astype(str)
    return df[patient_series.isin(patient_ids)].copy()


def print_split_summary(name: str, split_df: pd.DataFrame) -> None:
    print(f"{name}:")
    print(f"  images: {len(split_df)}")
    print(f"  patients: {split_df['Patient ID'].astype(str).nunique()}")
    print("  positive count by label:")
    for label in LABELS:
        print(f"    {label}: {int(split_df[label].sum())}")


def print_leakage_summary(
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> None:
    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids

    print("Patient ID intersection check:")
    print(f"  train vs val: {len(train_val)}")
    print(f"  train vs test: {len(train_test)}")
    print(f"  val vs test: {len(val_test)}")

    if train_val or train_test or val_test:
        raise ValueError("Patient leakage detected between splits.")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(input_csv)
    validate_inputs(df, args)

    patient_ids = df["Patient ID"].astype(str).drop_duplicates().to_numpy()
    train_ids, val_ids, test_ids = split_patient_ids(
        patient_ids=patient_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    splits = {
        "train": subset_by_patient(df, train_ids),
        "val": subset_by_patient(df, val_ids),
        "test": subset_by_patient(df, test_ids),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)

    print("NIH patient-level split summary")
    for split_name, split_df in splits.items():
        print_split_summary(split_name, split_df)
    print_leakage_summary(train_ids, val_ids, test_ids)
    print(f"Saved splits in: {output_dir}")


if __name__ == "__main__":
    main()
