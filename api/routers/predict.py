"""Prediction router with a Phase 1 placeholder response."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

DISCLAIMER = "Resultado preliminar de uso académico. No constituye diagnóstico clínico."

router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    gradcam_target: str | None = Form(default=None),
) -> JSONResponse:
    """Return the expected future schema without performing inference."""
    return JSONResponse(
        status_code=501,
        content={
            "status": "not_implemented",
            "message": "La inferencia real se implementará en fases posteriores.",
            "filename": file.filename,
            "probabilities": {},
            "positive_findings": [],
            "thresholds": {},
            "gradcam_target": gradcam_target,
            "gradcam_image_base64": None,
            "processing_time_ms": None,
            "disclaimer": DISCLAIMER,
        },
    )

