"""CLIP image classifier with a trained linear probe head.

The CLIP vision encoder is frozen; a single ``nn.Linear(512, 4)`` layer
(trained via ``scripts/train.py``) sits on top.  Images that lie too far
from all class centroids in cosine-distance space fall back to ``"other"``.

The "is it any known class?" (centroid distance) and "which class is it?"
(linear probe) decisions are independent — centroids handle outlier
detection; the probe handles fine-grained classification.

Pure Python — zero HTTP dependencies.  Can be used from scripts, notebooks, or
the FastAPI application equally.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tensor(output: torch.Tensor | object) -> torch.Tensor:
    """Extract a plain tensor from a HuggingFace model output.

    ``get_image_features()`` should return a tensor, but some ``transformers``
    versions return ``BaseModelOutputWithPooling`` instead.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output  # type: ignore[return-value]
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]  # type: ignore[return-value]
    raise TypeError(f"Cannot extract tensor from {type(output).__name__}")


# ---------------------------------------------------------------------------
# Internal result type
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
    """Image classifier backed by a frozen CLIP vision encoder + trained linear head.

    On initialisation the CLIP model is loaded and set to eval mode.
    The linear probe weights are loaded from a ``.pt`` file produced by
    ``scripts/train.py``.

    Parameters
    ----------
    model_name:
        HuggingFace model id, e.g. ``"openai/clip-vit-base-patch32"``.
    cache_dir:
        Local directory for cached model files.
    device:
        ``"cpu"`` or ``"cuda"``.
    offline:
        When ``True``, force ``local_files_only=True``.
    max_image_size:
        Images exceeding this are down-scaled before inference.
    linear_probe_path:
        Path to the trained ``LinearProbe`` state dict.
    label_map_path:
        Path to a JSON file with ``{"label_names": [...]}``.
    centroids_path:
        Path to ``model_cache/centroids.pt`` — class centroids for outlier
        detection (cosine-distance based "other" fallback).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: str | None = None,
        device: str = "cpu",
        offline: bool = False,
        max_image_size: int = 1920,
        linear_probe_path: str | Path | None = None,
        label_map_path: str | Path | None = None,
        centroids_path: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_image_size = max_image_size

        # ------------------------------------------------------------------
        # Load frozen CLIP vision encoder
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
        # Load trained linear probe
        # ------------------------------------------------------------------
        probe_path = Path(
            linear_probe_path
            or Path(cache_dir or "model_cache") / "linear_probe.pt"
        )
        map_path = Path(
            label_map_path
            or Path(cache_dir or "model_cache") / "label_map.json"
        )

        if not probe_path.exists():
            raise FileNotFoundError(
                f"Linear probe weights not found: {probe_path}\n"
                f"Run: python scripts/train.py"
            )
        if not map_path.exists():
            raise FileNotFoundError(
                f"Label map not found: {map_path}\n"
                f"Run: python scripts/train.py"
            )

        with open(map_path, encoding="utf-8") as f:
            label_data = json.load(f)
        self._label_names: list[str] = label_data["label_names"]
        self._label_to_idx: dict[str, int] = {n: i for i, n in enumerate(self._label_names)}
        num_classes = len(self._label_names)

        self._linear_head = nn.Linear(512, num_classes, bias=True)
        state_dict = torch.load(probe_path, map_location=device, weights_only=True)
        self._linear_head.load_state_dict(state_dict)
        self._linear_head.to(device)
        self._linear_head.eval()

        # ------------------------------------------------------------------
        # Load class centroids (outlier / "other" detection)
        # ------------------------------------------------------------------
        c_path = Path(
            centroids_path
            or Path(cache_dir or "model_cache") / "centroids.pt"
        )

        if not c_path.exists():
            raise FileNotFoundError(
                f"Centroids file not found: {c_path}\n"
                f"Run: python scripts/train.py"
            )

        centroids_data = torch.load(c_path, map_location=device, weights_only=True)
        self._centroids = centroids_data["centroids"].to(device)  # [C, 512]
        # Ensure centroids are L2-normalised (they should be from train.py, but be safe)
        self._centroids = F.normalize(self._centroids, p=2, dim=1)
        # Verify label order matches the probe
        centroids_labels = centroids_data["label_names"]
        if centroids_labels != self._label_names:
            logger.warning(
                "Centroid label order %s differs from probe %s — results may be wrong!",
                centroids_labels, self._label_names,
            )

        logger.info(
            "CLIP classifier ready — %d classes (linear probe + centroids): %s",
            num_classes,
            ", ".join(self._label_names),
        )

    # ------------------------------------------------------------------
    # Single-image classification
    # ------------------------------------------------------------------

    def classify_single(
        self,
        image_path: str | Path,
        *,
        distance_threshold: float | None = None,
    ) -> ClassificationResult:
        """Classify a single image on disk.

        Returns a ``ClassificationResult`` with the top-1 label and softmax
        scores across all training classes.  When *distance_threshold* is
        set and the image's minimum cosine distance to any class centroid
        exceeds it, the label is forced to ``"other"``.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot load image '{image_path}': {exc}") from exc

        if max(image.size) > self.max_image_size:
            image.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)

        return self._classify_image(image, str(image_path), distance_threshold)

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        image_paths: list[str | Path],
        *,
        distance_threshold: float | None = None,
    ) -> list[ClassificationResult]:
        """Classify multiple images, processing in sub-batches.

        Images that fail to load are returned as ``"error"`` results rather
        than aborting the entire batch.  Results are always in the same order
        as the input *image_paths*.
        """
        from app.config import settings

        batch_size = settings.batch_size
        results: list[ClassificationResult] = []

        for chunk_start in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[chunk_start : chunk_start + batch_size]
            images: list[Image.Image] = []
            chunk_map: dict[int, ClassificationResult | None] = {}

            # --- Load images, record errors by position ---
            for i, p in enumerate(chunk_paths):
                global_idx = chunk_start + i
                path = Path(p)
                if not path.exists():
                    chunk_map[global_idx] = ClassificationResult(
                        image_path=str(p), label="error",
                        confidence=0.0, all_scores={},
                    )
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                except Exception:
                    chunk_map[global_idx] = ClassificationResult(
                        image_path=str(p), label="error",
                        confidence=0.0, all_scores={},
                    )
                    continue

                if max(image.size) > self.max_image_size:
                    image.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)

                images.append(image)
                chunk_map[global_idx] = None  # placeholder — filled after inference

            pending = [idx for idx, r in chunk_map.items() if r is None]

            # --- Run inference on valid images ---
            if images:
                batch_results = self._classify_images(
                    images,
                    [str(image_paths[idx]) for idx in pending],
                    distance_threshold,
                )
                for idx, cr in zip(pending, batch_results):
                    chunk_map[idx] = cr

            # --- Collect in input order ---
            for global_idx in range(chunk_start, chunk_start + len(chunk_paths)):
                r = chunk_map[global_idx]
                assert r is not None, f"Missing result for index {global_idx}"
                results.append(r)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _classify_images(
        self,
        images: list[Image.Image],
        paths: list[str],
        distance_threshold: float | None,
    ) -> list[ClassificationResult]:
        """Run a single forward pass for a batch of images.

        1. Encode with frozen CLIP → 512-dim features
        2. L2-normalise features and compute cosine distance to each centroid
        3. If min distance > *distance_threshold* → ``"other"``
        4. Otherwise → linear probe softmax → best class
        """
        # Step 1 — CLIP encode
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device)

        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = _to_tensor(image_features)              # [B, 512]

        # Step 2 — cosine distance to centroids
        features_norm = F.normalize(image_features, p=2, dim=1)  # [B, 512]
        cos_sim = features_norm @ self._centroids.T               # [B, C]
        cos_dist = 1.0 - cos_sim                                  # [B, C]
        min_dist, nearest_centroid = cos_dist.min(dim=1)          # [B], [B]

        # Step 3 — linear probe classification (for images that pass the centroid check)
        logits = self._linear_head(image_features)                # [B, C]
        probs = logits.softmax(dim=-1)                            # [B, C]

        results: list[ClassificationResult] = []
        for i, path in enumerate(paths):
            scores: dict[str, float] = {}
            for j, label in enumerate(self._label_names):
                scores[label] = round(probs[i, j].item(), 4)

            # --- Centroid-distance check ---
            if distance_threshold is not None and min_dist[i].item() > distance_threshold:
                results.append(ClassificationResult(
                    image_path=path,
                    label="other",
                    confidence=round(1.0 - min_dist[i].item(), 4),
                    all_scores=scores,
                ))
                continue

            # --- Normal classification via linear probe ---
            best_idx = probs[i].argmax().item()
            best_label = self._label_names[best_idx]
            best_conf = probs[i, best_idx].item()

            results.append(ClassificationResult(
                image_path=path,
                label=best_label,
                confidence=round(best_conf, 4),
                all_scores=scores,
            ))

        return results

    def _classify_image(
        self,
        image: Image.Image,
        path: str,
        distance_threshold: float | None,
    ) -> ClassificationResult:
        return self._classify_images([image], [path], distance_threshold)[0]
