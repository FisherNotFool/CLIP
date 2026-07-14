"""API router — endpoint definitions."""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from app.api.deps import get_classifier, get_settings
from app.config import Settings
from app.errors.handlers import ImageLoadError, ImageNotFoundError, InferenceError
from app.schemas.request import ClassifyRequest
from app.schemas.response import ClassifyResponse, ImageClassification
from app.services.classifier import ClassificationResult, ClipClassifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/clip", tags=["classification"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "classifier"),
    }


# ---------------------------------------------------------------------------
# Classification endpoint
# ---------------------------------------------------------------------------


@router.post("/classify", response_model=ClassifyResponse)
async def classify_images(
    body: ClassifyRequest,
    classifier: ClipClassifier = Depends(get_classifier),
    settings: Settings = Depends(get_settings),
    debug: bool = Query(
        default=False,
        description="Populate all_scores for every image",
    ),
):
    """Classify one or more images from a Ceramics document.

    Returns ``image_type`` and ``confidence`` for each image.
    Pass ``?debug=true`` to receive the full softmax distribution per image.
    """
    started_at = time.perf_counter()
    image_inputs = body.image_inputs
    logger.info(
        "classify request document_id=%s images=%d captions=%d detector=%s debug=%s",
        body.document_id, len(image_inputs),
        sum(image.caption is not None for image in image_inputs),
        "gate", debug,
    )

    # Resolve paths against the configured image base directory
    resolved_paths: list[Path] = []
    for image in image_inputs:
        # Normalise: strip leading slash so join works correctly
        clean = image.image_path.lstrip("/").lstrip("\\")
        resolved_paths.append(settings.image_base_path / clean)

    # Delegate to the classifier service
    try:
        results: list[ClassificationResult] = classifier.classify_batch(
            resolved_paths,
            captions=[image.caption for image in image_inputs],
        )
    except Exception as exc:
        logger.exception("classify failed document_id=%s images=%d", body.document_id, len(image_inputs))
        raise InferenceError(str(exc)) from exc

    labels = Counter(result.label for result in results)
    errors = sum(result.label == "error" for result in results)
    logger.info(
        "classify completed document_id=%s elapsed_ms=%.1f labels=%s errors=%d",
        body.document_id, (time.perf_counter() - started_at) * 1000,
        dict(sorted(labels.items())), errors,
    )
    for result in results:
        logger.debug(
            "classify result document_id=%s path=%s label=%s confidence=%.4f",
            body.document_id, result.image_path, result.label, result.confidence,
        )

    # Map internal results → Pydantic response models
    classifications: list[ImageClassification] = []
    for r in results:
        classifications.append(
            ImageClassification(
                image_path=r.image_path,
                image_type=r.label,
                confidence=r.confidence,
                all_scores=r.all_scores if debug else None,
                error=None if r.label != "error" else f"Classification failed for {r.image_path}",
            )
        )

    return ClassifyResponse(
        document_id=body.document_id,
        classifications=classifications,
        model_name=classifier.model_name,
        model_device=classifier.device,
    )
