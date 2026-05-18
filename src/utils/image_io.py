"""Image loading helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image_as_rgb(path: str | Path) -> Image.Image:
    """Load a standard image file and return it as RGB."""
    return Image.open(path).convert("RGB")

