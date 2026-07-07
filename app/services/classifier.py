"""CLIP zero-shot image classifier with prompt ensembling.

Pure Python — zero HTTP dependencies. Can be used from scripts, notebooks, or
the FastAPI application equally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates — the core of zero-shot classification quality
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: dict[str, list[str]] = {
    "bar_chart": [
        "a bar chart with rectangular bars showing material property comparisons",
        "a grouped or stacked bar chart comparing different material compositions or properties",
        "a scientific bar chart showing quantitative comparisons with labeled axes",
    ],
    "line_chart": [
        "a line chart with smooth continuous curves showing trends over temperature, concentration, or time",
        "a multi-line trend chart plotting experimental data points connected by lines",
        "a scientific line graph with data series plotted against a continuous variable",
    ],
    "sem": [
        "a scanning electron microscope (SEM) micrograph showing material microstructure in grayscale",
        "an SEM image revealing particle morphology, grain boundaries, or surface texture at high magnification",
        "a grayscale electron microscopy image showing material surface features, pores, or fracture surface",
    ],
    "xrd": [
        "an X-ray diffraction (XRD) pattern with multiple sharp characteristic peaks at specific angles on a flat baseline",
        "an XRD spectrum displaying peak intensity versus 2-theta diffraction angle with narrow crystalline peaks",
        "a powder X-ray diffraction pattern showing intensity counts versus scattering angle with distinct bragg peaks",
    ],
    "other": [
        "a photograph of laboratory equipment, samples, or experimental setup",
        "a schematic diagram, flowchart, or process illustration showing experimental procedure",
        "a chemical structure diagram, molecular model, or crystal structure representation",
        "an optical microscope photograph, TEM image, or other characterization result",
    ],
}


# ---------------------------------------------------------------------------
# Internal result type (not Pydantic — the API layer maps to response models)
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    image_path: str
    label: str
    confidence: float
    all_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ClipClassifier:
    """Zero-shot image classifier backed by a CLIP model.

    On initialisation the model and processor are loaded and *text embeddings*
    for every prompt template are pre-computed (ensembled per class).  Inference
    then only needs to encode images and compute cosine similarity against the
    cached text embeddings.

    Parameters
    ----------
    model_name:
        HuggingFace model id, e.g. ``"openai/clip-vit-base-patch32"``.
    cache_dir:
        Local directory for cached model files.  Set together with
        ``offline=True`` for air-gapped deployments.
    device:
        ``"cpu"`` or ``"cuda"`` (or ``"cuda:0"``, ``"mps"``).
    offline:
        When ``True``, force ``local_files_only=True`` so that *no* network
        requests are made to HuggingFace Hub.
    max_image_size:
        Images with either dimension exceeding this are down-scaled
        (aspect-ratio preserved) before being fed to CLIP.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: str | None = None,
        device: str = "cpu",
        offline: bool = False,
        max_image_size: int = 1920,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_image_size = max_image_size

        # ------------------------------------------------------------------
        # Load model & processor
        # ------------------------------------------------------------------
        logger.info("Loading CLIP model '%s' on device '%s' ...", model_name, device)
        self.model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        self.processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        self.model = self.model.to(device)
        self.model.eval()

        # ------------------------------------------------------------------
        # Pre-compute ensembled text embeddings
        # ------------------------------------------------------------------
        self._labels: list[str] = []
        self._text_embeddings: torch.Tensor | None = None  # [num_classes, dim]
        self._precompute_text_embeddings()

        logger.info("CLIP classifier ready — %d classes loaded.", len(self._labels))

    # ------------------------------------------------------------------
    # Text embedding pre-computation
    # ------------------------------------------------------------------

    def _precompute_text_embeddings(self) -> None:
        """Encode every prompt template and average per class.

        Result is stored in ``self._text_embeddings`` as a [C, D] tensor
        where C = number of classes and D = embedding dimension.
        """
        all_embeddings: list[torch.Tensor] = []
        self._labels = []

        for label, templates in PROMPT_TEMPLATES.items():
            if not templates:
                raise ValueError(f"No prompt templates defined for class '{label}'")

            template_embeddings: list[torch.Tensor] = []
            for template in templates:
                inputs = self.processor(
                    text=[template],
                    return_tensors="pt",
                    padding=True,
                )
                # Move to device (only the text tensors — no pixel_values)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    text_features = self.model.get_text_features(**inputs)
                    text_features = F.normalize(text_features, dim=-1)

                template_embeddings.append(text_features)

            # Average across templates for this class, then re-normalize
            ensembled = torch.stack(template_embeddings).mean(dim=0, keepdim=True)
            ensembled = F.normalize(ensembled, dim=-1)

            all_embeddings.append(ensembled)
            self._labels.append(label)

        self._text_embeddings = torch.cat(all_embeddings, dim=0)  # [C, D]
        logger.debug(
            "Pre-computed text embeddings: shape=%s",
            tuple(self._text_embeddings.shape),
        )

    # ------------------------------------------------------------------
    # Single-image classification
    # ------------------------------------------------------------------

    def classify_single(
        self,
        image_path: str | Path,
        *,
        confidence_threshold: float | None = None,
    ) -> ClassificationResult:
        """Classify a single image on disk.

        Returns a ``ClassificationResult`` with the top-1 label and full
        softmax scores across all classes.

        Raises
        ------
        FileNotFoundError
            If *image_path* does not exist.
        ValueError
            If the image cannot be opened / decoded.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot load image '{image_path}': {exc}") from exc

        # Down-scale large images while preserving aspect ratio
        if max(image.size) > self.max_image_size:
            image.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)

        return self._classify_image(image, str(image_path), confidence_threshold)

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        image_paths: list[str | Path],
        *,
        confidence_threshold: float | None = None,
    ) -> list[ClassificationResult]:
        """Classify multiple images, processing in sub-batches.

        Images that fail to load are returned as "error" results rather than
        aborting the entire batch.
        """
        from app.config import settings

        batch_size = settings.batch_size
        results: list[ClassificationResult] = []

        for chunk_start in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[chunk_start : chunk_start + batch_size]
            images: list[Image.Image] = []
            valid_indices: list[int] = []

            for i, p in enumerate(chunk_paths):
                path = Path(p)
                if not path.exists():
                    results.append(
                        ClassificationResult(
                            image_path=str(p),
                            label="error",
                            confidence=0.0,
                            all_scores={},
                        )
                    )
                    continue

                try:
                    image = Image.open(path).convert("RGB")
                except Exception:
                    results.append(
                        ClassificationResult(
                            image_path=str(p),
                            label="error",
                            confidence=0.0,
                            all_scores={},
                        )
                    )
                    continue

                if max(image.size) > self.max_image_size:
                    image.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)

                images.append(image)
                valid_indices.append(chunk_start + i)

            if not images:
                continue

            # Batch inference on valid images
            chunk_results = self._classify_images(images, [str(image_paths[i]) for i in valid_indices], confidence_threshold)

            # Interleave results back into correct positions
            result_idx = 0
            for global_idx in range(chunk_start, chunk_start + len(chunk_paths)):
                if global_idx in valid_indices:
                    results.append(chunk_results[result_idx])
                    result_idx += 1
                # else: already appended error result above

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _classify_images(
        self,
        images: list[Image.Image],
        paths: list[str],
        confidence_threshold: float | None,
    ) -> list[ClassificationResult]:
        """Run a single forward pass for a batch of images."""
        assert self._text_embeddings is not None

        inputs = self.processor(
            text=[""],  # dummy — we already have text embeddings
            images=images,
            return_tensors="pt",
            padding=True,
        )
        # Only keep pixel values
        pixel_values = inputs["pixel_values"].to(self.device)

        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = F.normalize(image_features, dim=-1)  # [B, D]

        # Cosine similarity → logits → softmax
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ self._text_embeddings.T)  # [B, C]
        probs = logits.softmax(dim=-1)  # [B, C]

        threshold = confidence_threshold

        results: list[ClassificationResult] = []
        for i, path in enumerate(paths):
            scores = {
                label: round(probs[i, j].item(), 4)
                for j, label in enumerate(self._labels)
            }
            best_idx = probs[i].argmax().item()
            best_label = self._labels[best_idx]
            best_conf = probs[i, best_idx].item()

            # Apply confidence threshold — force "other" on weak predictions
            if threshold is not None and best_conf < threshold:
                best_label = "other"
                best_conf = scores.get("other", best_conf)

            results.append(
                ClassificationResult(
                    image_path=path,
                    label=best_label,
                    confidence=round(best_conf, 4),
                    all_scores=scores,
                )
            )

        return results

    def _classify_image(
        self,
        image: Image.Image,
        path: str,
        confidence_threshold: float | None,
    ) -> ClassificationResult:
        """Classify a single PIL image (internal, no I/O)."""
        return self._classify_images([image], [path], confidence_threshold)[0]
