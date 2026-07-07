"""FastAPI dependency callables."""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.services.classifier import ClipClassifier


def get_settings(request: Request) -> Settings:
    """Return the application settings singleton."""
    return request.app.state.settings


def get_classifier(request: Request) -> ClipClassifier:
    """Return the CLIP classifier singleton (loaded at startup)."""
    return request.app.state.classifier
