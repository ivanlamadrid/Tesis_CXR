"""Prepare NIH ChestX-ray14 metadata for multi-label training.

The script does not download data or copy images. It only reads the NIH metadata
CSV, expands the 14 pathology labels into binary columns and records where each
image should be found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.labels import NO_FINDING, normalize_finding_label

REQUIRED_COLUMNS = ["Image Index", "Finding Labels", "Patient ID"]
DEFAULT_IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a multilabel CSV for NIH ChestX-ray14."
    )
    parser.add_argument(
        "--metadata-csv",
        default="data/raw/nih/Data_Entry_2017.csv",
        help="Path to NIH Data_Entry_2017.csv.",
    )
    parser.add_argument(
        "--images-root",
        default="data/raw/nih/images",
        help="Directory containing NIH images or subdirectories with images.",
    )
    parser.add_argument(
        "--labels-json",
        default="artifacts/labels.json",
        help="Path to the project label list.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/nih_multilabel.csv",
        help="Output path for the processed multilabel CSV.",
    )
    parser.add_argument(
        "--recursive-image-search",
        action="store_true",
        help="Search images recursively below --images-root.",
    )
    return parser.parse_args()


def load_labels(labels_json: Path) -> list[str]:
    with labels_json.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError(f"{labels_json} must contain a JSON list of label names.")
    if len(labels) != 14:
        raise ValueError(f"Expected 14 labels, found {len(labels)}.")
    if NO_FINDING in labels:
        raise ValueError('"No Finding" must not be present in labels.json.')
    return labels


def split_findings(value: object) -> list[str]:
    if pd.isna(value):
        return []
    findings: list[str] = []
    for part in str(value).split("|"):
        normalized = normalize_finding_label(part)
        if normalized:
            findings.append(normalized)
    return findings


def validate_metadata(df: pd.DataFrame, metadata_csv: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{metadata_csv} is missing required columns: {missing}")


def image_suffixes_from_names(image_names: set[str]) -> list[str]:
    suffixes = sorted({Path(name).suffix.lower() for name in image_names if Path(name).suffix})
    return suffixes or DEFAULT_IMAGE_SUFFIXES


def register_image_path(
    image_path: Path,
    image_names: set[str],
    image_index: dict[str, Path],
    duplicates: list[str],
) -> None:
    image_name = image_path.name
    if image_name not in image_names:
        return
    if image_name in image_index:
        if image_index[image_name] != image_path:
            duplicates.append(image_name)
        return
    image_index[image_name] = image_path


def index_from_patterns(
    images_root: Path,
    image_names: set[str],
    image_index: dict[str, Path],
    duplicates: list[str],
    patterns: list[str],
) -> None:
    expected_count = len(image_names)
    for pattern in patterns:
        for image_path in images_root.glob(pattern):
            register_image_path(image_path, image_names, image_index, duplicates)
            if len(image_index) == expected_count:
                return


def print_image_index_progress(stage: str, expected_count: int, indexed_count: int) -> None:
    missing_count = expected_count - indexed_count
    print(
        f"[image-index] {stage}: expected={expected_count} "
        f"indexed={indexed_count} missing={missing_count}"
    )


def build_recursive_image_index(images_root: Path, image_names: set[str]) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    duplicates: list[str] = []
    expected_count = len(image_names)
    suffixes = image_suffixes_from_names(image_names)

    print_image_index_progress("start", expected_count, len(image_index))

    direct_patterns = [f"*{suffix}" for suffix in suffixes]
    index_from_patterns(images_root, image_names, image_index, duplicates, direct_patterns)
    print_image_index_progress("after direct folder scan", expected_count, len(image_index))

    if len(image_index) < expected_count:
        kaggle_patterns = [f"images_*/images/*{suffix}" for suffix in suffixes]
        index_from_patterns(images_root, image_names, image_index, duplicates, kaggle_patterns)
        print_image_index_progress("after images_*/images scan", expected_count, len(image_index))

    if len(image_index) < expected_count:
        recursive_patterns = [f"**/*{suffix}" for suffix in suffixes]
        index_from_patterns(images_root, image_names, image_index, duplicates, recursive_patterns)
        print_image_index_progress("after filtered recursive fallback", expected_count, len(image_index))

    if duplicates:
        duplicate_preview = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"Duplicate image filenames found during recursive search: {duplicate_preview}")

    print_image_index_progress("final", expected_count, len(image_index))
    return image_index


def attach_image_paths(
    df: pd.DataFrame,
    images_root: Path,
    recursive_image_search: bool,
) -> pd.DataFrame:
    image_names = set(df["Image Index"].astype(str))

    if recursive_image_search:
        image_index = build_recursive_image_index(images_root, image_names)
        paths = [image_index.get(image_name) for image_name in df["Image Index"].astype(str)]
        df["image_path"] = [str(path) if path is not None else "" for path in paths]
        df["image_exists"] = [path is not None for path in paths]
    else:
        paths = [images_root / image_name for image_name in df["Image Index"].astype(str)]
        df["image_path"] = [str(path) for path in paths]
        df["image_exists"] = [path.exists() for path in paths]
    return df


def add_label_columns(df: pd.DataFrame, labels: list[str]) -> tuple[pd.DataFrame, int]:
    label_set = set(labels)
    unknown_counter: Counter[str] = Counter()
    no_finding_rows = 0
    binary_rows: list[dict[str, int]] = []

    for raw_findings in df["Finding Labels"]:
        findings = split_findings(raw_findings)
        finding_set = set(findings)

        if finding_set == {NO_FINDING}:
            no_finding_rows += 1
            binary_rows.append({label: 0 for label in labels})
            continue

        for finding in finding_set:
            if finding != NO_FINDING and finding not in label_set:
                unknown_counter[finding] += 1

        binary_rows.append({label: int(label in finding_set) for label in labels})

    if unknown_counter:
        unknown_preview = ", ".join(
            f"{label} ({count})" for label, count in unknown_counter.most_common(10)
        )
        raise ValueError(f"Unknown finding labels found in metadata: {unknown_preview}")

    label_df = pd.DataFrame(binary_rows, index=df.index)
    return pd.concat([df, label_df], axis=1), no_finding_rows


def validate_output_label_columns(df: pd.DataFrame, labels: list[str]) -> None:
    missing = [label for label in labels if label not in df.columns]
    if missing:
        raise ValueError(f"Output CSV is missing canonical label columns: {missing}")

    forbidden_columns = ["Pleural_Thickening", NO_FINDING]
    present_forbidden = [column for column in forbidden_columns if column in df.columns]
    if present_forbidden:
        raise ValueError(f"Output CSV must not contain non-canonical label columns: {present_forbidden}")

    if "Pleural Thickening" not in df.columns:
        raise ValueError('Output CSV must contain canonical column "Pleural Thickening".')


def print_summary(df: pd.DataFrame, labels: list[str], no_finding_rows: int) -> None:
    total_rows = len(df)
    unique_patients = df["Patient ID"].nunique()
    existing_images = int(df["image_exists"].sum())
    missing_images = total_rows - existing_images

    print("NIH multilabel preparation summary")
    print(f"Total rows: {total_rows}")
    print(f"Unique patients: {unique_patients}")
    print(f"Existing images: {existing_images}")
    print(f"Missing images: {missing_images}")
    print(f"No Finding rows: {no_finding_rows}")
    print("Positive count by label:")
    for label in labels:
        print(f"  {label}: {int(df[label].sum())}")


def main() -> None:
    args = parse_args()
    metadata_csv = Path(args.metadata_csv)
    images_root = Path(args.images_root)
    labels_json = Path(args.labels_json)
    output_csv = Path(args.output_csv)

    labels = load_labels(labels_json)
    df = pd.read_csv(metadata_csv)
    validate_metadata(df, metadata_csv)

    df, no_finding_rows = add_label_columns(df, labels)
    df = attach_image_paths(df, images_root, args.recursive_image_search)
    validate_output_label_columns(df, labels)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print_summary(df, labels, no_finding_rows)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
