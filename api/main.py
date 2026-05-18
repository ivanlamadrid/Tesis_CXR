"""Minimal FastAPI service for the Tesis_CXR academic demo."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI

from routers.predict import router as predict_router

DISCLAIMER = "Resultado preliminar de uso académico. No constituye diagnóstico clínico."
VERSION = "0.1.0"


def _artifact_path(filename: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "artifacts" / filename,
        base_dir.parent / "artifacts" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_labels() -> list[str]:
    labels_path = _artifact_path("labels.json")
    with labels_path.open("r", encoding="utf-8") as file:
        return json.load(file)


app = FastAPI(
    title="Tesis_CXR API",
    description="API académica preliminar para clasificación multi-etiqueta de radiografías de tórax.",
    version=VERSION,
)
app.include_router(predict_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health and mandatory academic disclaimer."""
    return {
        "status": "ok",
        "service": "cxr-api",
        "version": VERSION,
        "disclaimer": DISCLAIMER,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Return placeholder model metadata for Phase 1."""
    labels = _load_labels()
    return {
        "model_loaded": False,
        "model_name": "cnn_vit_multilabel",
        "num_labels": len(labels),
        "status": "placeholder",
        "message": "El modelo real se cargará en fases posteriores.",
        "disclaimer": DISCLAIMER,
    }

