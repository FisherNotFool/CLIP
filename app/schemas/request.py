"""Request schemas for the CLIP classify endpoint."""

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """A batch classification request for images from one document."""

    document_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Ceramics document ID",
        examples=["doc_xxx"],
    )
    image_paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of image file paths (relative to IMAGE_BASE_PATH)",
        examples=[["/outputs/doc_xxx/images/1f2a3b4c.jpg"]],
    )
