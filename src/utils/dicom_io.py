"""DICOM loading helpers.

This module contains a simple Phase 1 implementation and will be hardened later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def load_dicom_as_pil(path: str | Path) -> Image.Image:
    """Load a DICOM file as a normalized 8-bit PIL image."""
    dataset = pydicom.dcmread(str(path))
    pixels = dataset.pixel_array.astype(np.float32)
    pixels -= float(pixels.min())

    max_value = float(pixels.max())
    if max_value > 0:
        pixels /= max_value

    pixels = (pixels * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")

