"""Custom exceptions and FastAPI exception handlers."""

from fastapi import Request
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class ImageNotFoundError(Exception):
    """Image file does not exist on disk."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Image not found: {path}")


class ImageLoadError(Exception):
    """File exists but cannot be opened as a valid image."""

    def __init__(self, path: str, detail: str = ""):
        self.path = path
        self.detail = detail
        super().__init__(f"Cannot load image '{path}': {detail}")


class InferenceError(Exception):
    """Model inference failed (OOM, device error, etc.)."""

    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(f"Inference failed: {detail}")


# ---------------------------------------------------------------------------
# FastAPI exception handlers
# ---------------------------------------------------------------------------


async def image_not_found_handler(request: Request, exc: ImageNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": f"Image not found: {exc.path}"},
    )


async def image_load_error_handler(request: Request, exc: ImageLoadError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": f"Cannot load image: {exc.path}", "reason": exc.detail},
    )


async def inference_error_handler(request: Request, exc: InferenceError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal inference error"},
    )
