"""FastAPI application entry point.

Usage::

    uvicorn app.main:app --host 0.0.0.0 --port 8011
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router
from app.config import Settings
from app.errors.handlers import (
    ImageLoadError,
    ImageNotFoundError,
    InferenceError,
    image_load_error_handler,
    image_not_found_handler,
    inference_error_handler,
)
from app.lifespan import lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a FastAPI application instance.

    A factory function (rather than a module-level ``app``) so that tests can
    create isolated instances with different *settings*.
    """
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="CLIP Image Classifier",
        description="Image classification for materials-science paper figures.",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Attach settings to app state so lifespan & deps can access them
    app.state.settings = settings

    # CORS — allow Ceramics frontend on :5173 and any other origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(ImageNotFoundError, image_not_found_handler)
    app.add_exception_handler(ImageLoadError, image_load_error_handler)
    app.add_exception_handler(InferenceError, inference_error_handler)

    # Routes
    app.include_router(router)

    return app


# Module-level instance for uvicorn (``uvicorn app.main:app``)
app = create_app()
