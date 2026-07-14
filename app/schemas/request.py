"""Request schemas for the CLIP classification endpoint."""

from pydantic import BaseModel, Field, model_validator


class ImageInput(BaseModel):
    """One extracted document image and its optional upstream caption."""

    image_path: str = Field(..., min_length=1, description="Path relative to IMAGE_BASE_PATH")
    caption: str | None = Field(default=None, max_length=4000)


class ClassifyRequest(BaseModel):
    """A batch classification request for images from one document.

    ``images`` is the current contract. ``image_paths`` remains temporarily
    supported for callers that have not yet been upgraded.
    """

    document_id: str = Field(..., min_length=1, max_length=256, description="Ceramics document ID")
    images: list[ImageInput] | None = Field(default=None, max_length=100)
    image_paths: list[str] | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def require_one_image_representation(self) -> "ClassifyRequest":
        if self.images and self.image_paths:
            raise ValueError("Provide either images or image_paths, not both")
        if not self.images and not self.image_paths:
            raise ValueError("At least one image is required")
        if self.images is not None and len(self.images) == 0:
            raise ValueError("images must not be empty")
        if self.image_paths is not None and len(self.image_paths) == 0:
            raise ValueError("image_paths must not be empty")
        return self

    @property
    def image_inputs(self) -> list[ImageInput]:
        if self.images is not None:
            return self.images
        return [ImageInput(image_path=path) for path in self.image_paths or []]
