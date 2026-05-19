"""Utilities for NIH ChestX-ray14 label normalization."""

from __future__ import annotations

import re

NO_FINDING = "No Finding"

CANONICAL_NIH_LABELS = [
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

_SPACE_PATTERN = re.compile(r"\s+")
_CANONICAL_BY_NORMALIZED = {
    _SPACE_PATTERN.sub(" ", label.replace("_", " ").strip()).casefold(): label
    for label in [*CANONICAL_NIH_LABELS, NO_FINDING]
}


def normalize_finding_label(label: str) -> str:
    """Normalize a raw NIH finding label to the project canonical spelling."""
    normalized = _SPACE_PATTERN.sub(" ", str(label).replace("_", " ").strip())
    return _CANONICAL_BY_NORMALIZED.get(normalized.casefold(), normalized)
