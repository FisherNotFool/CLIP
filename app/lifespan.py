"""FastAPI lifespan — model loading / unloading."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.services.classifier import ClipClassifier

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CLIP model on startup, clean up on shutdown."""
    settings = app.state.settings

    logger.info(
        "Starting CLIP classifier (model=%s, device=%s, offline=%s) ...",
        settings.clip_model_name,
        settings.device,
        settings.transformers_offline,
    )

    app.state.classifier = ClipClassifier(
        model_name=settings.clip_model_name,
        cache_dir=str(settings.model_cache_dir),
        device=settings.device,
        offline=settings.transformers_offline,
        max_image_size=settings.max_image_size,
    )

    logger.info("CLIP classifier loaded — ready to serve.")

    yield  # <-- application runs here

    # --- Shutdown ---
    if hasattr(app.state, "classifier"):
        del app.state.classifier

    if _cuda_available():
        import torch

        torch.cuda.empty_cache()
        logger.info("CUDA memory released.")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
