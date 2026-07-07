"""Response schemas for the CLIP classify endpoint."""

from pydantic import BaseModel, Field


class ImageClassification(BaseModel):
    """Classification result for a single image."""

    image_path: str
    image_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: dict[str, float] | None = Field(
        default=None,
        description="Full softmax distribution across all classes (populated when ?debug=true)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if this image could not be classified",
    )


class ClassifyResponse(BaseModel):
    """Response for a batch classification request."""

    document_id: str
    classifications: list[ImageClassification]
    model_name: str
    model_device: str
