"""Gate-first CLIP image classifier for the production API service."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.services.caption_rules import caption_override

logger = logging.getLogger(__name__)


def _to_tensor(output: torch.Tensor | object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output  # type: ignore[return-value]
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]  # type: ignore[return-value]
    raise TypeError(f"Cannot extract tensor from {type(output).__name__}")


@dataclass
class ClassificationResult:
    image_path: str
    label: str
    confidence: float
    all_scores: dict[str, float] = field(default_factory=dict)


class ClipClassifier:
    """Frozen CLIP encoder, explicit other gate, and five-class linear probe."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: str | None = None,
        device: str = "cpu",
        offline: bool = False,
        max_image_size: int = 1920,
        linear_probe_path: str | Path | None = None,
        label_map_path: str | Path | None = None,
        other_gate_path: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_image_size = max_image_size
        cache_path = Path(cache_dir or "model_cache")
        logger.info("Loading CLIP model '%s' on device '%s'", model_name, device)
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=offline).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=offline)

        probe_path = Path(linear_probe_path or cache_path / "linear_probe.pt")
        map_path = Path(label_map_path or cache_path / "label_map.json")
        gate_path = Path(other_gate_path or cache_path / "other_gate.pt")
        for artifact, name in ((probe_path, "Linear probe"), (map_path, "Label map"), (gate_path, "Other gate")):
            if not artifact.exists():
                raise FileNotFoundError(f"{name} artifact not found: {artifact}")

        with map_path.open(encoding="utf-8") as handle:
            self._label_names: list[str] = json.load(handle)["label_names"]
        self._linear_head = nn.Linear(512, len(self._label_names), bias=True).to(device)
        self._linear_head.load_state_dict(torch.load(probe_path, map_location=device, weights_only=True))
        self._linear_head.eval()
        gate_data = torch.load(gate_path, map_location=device, weights_only=True)
        self._other_gate = nn.Linear(512, 2, bias=True).to(device)
        self._other_gate.load_state_dict(gate_data["state_dict"])
        self._other_gate.eval()
        self._other_gate_threshold = float(gate_data["threshold"])
        logger.info("CLIP classifier ready labels=%s gate_threshold=%.4f", ", ".join(self._label_names), self._other_gate_threshold)

    def classify_single(self, image_path: str | Path, *, caption: str | None = None) -> ClassificationResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot load image '{image_path}': {exc}") from exc
        if max(image.size) > self.max_image_size:
            image.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)
        return self._classify_images([image], [str(image_path)], [caption])[0]

    def classify_batch(self, image_paths: list[str | Path], *, captions: list[str | None] | None = None) -> list[ClassificationResult]:
        from app.config import settings

        if captions is not None and len(captions) != len(image_paths):
            raise ValueError("captions must have the same length as image_paths")
        captions = captions or [None] * len(image_paths)
        started_at = time.perf_counter()
        results: list[ClassificationResult] = []
        for start in range(0, len(image_paths), settings.batch_size):
            paths = image_paths[start : start + settings.batch_size]
            chunk_captions = captions[start : start + settings.batch_size]
            images: list[Image.Image] = []
            pending_paths: list[str] = []
            pending_captions: list[str | None] = []
            chunk_results: list[ClassificationResult | None] = [None] * len(paths)
            for index, path_value in enumerate(paths):
                path = Path(path_value)
                if not path.exists():
                    logger.warning("classifier image missing path=%s", path_value)
                    chunk_results[index] = ClassificationResult(str(path_value), "error", 0.0)
                    continue
                try:
                    image = Image.open(path).convert("RGB")
                except Exception:
                    logger.warning("classifier image unreadable path=%s", path_value, exc_info=True)
                    chunk_results[index] = ClassificationResult(str(path_value), "error", 0.0)
                    continue
                if max(image.size) > self.max_image_size:
                    image.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)
                images.append(image)
                pending_paths.append(str(path_value))
                pending_captions.append(chunk_captions[index])
            inferred = iter(self._classify_images(images, pending_paths, pending_captions)) if images else iter(())
            for index, result in enumerate(chunk_results):
                if result is None:
                    chunk_results[index] = next(inferred)
            results.extend(result for result in chunk_results if result is not None)
        logger.info("classifier batch completed images=%d elapsed_ms=%.1f labels=%s", len(results), (time.perf_counter() - started_at) * 1000, dict(sorted(Counter(result.label for result in results).items())))
        return results

    @torch.no_grad()
    def _classify_images(self, images: list[Image.Image], paths: list[str], captions: list[str | None]) -> list[ClassificationResult]:
        pixel_values = self.processor(images=images, return_tensors="pt", padding=True)["pixel_values"].to(self.device)
        features = _to_tensor(self.model.get_image_features(pixel_values=pixel_values))
        probabilities = self._linear_head(features).softmax(dim=-1)
        other_probabilities = self._other_gate(features).softmax(dim=-1)[:, 1]
        results: list[ClassificationResult] = []
        for index, path in enumerate(paths):
            scores = {label: round(probabilities[index, j].item(), 4) for j, label in enumerate(self._label_names)}
            other_confidence = other_probabilities[index].item()
            if other_confidence >= self._other_gate_threshold:
                results.append(ClassificationResult(path, "other", round(other_confidence, 4), scores))
                continue
            best_index = probabilities[index].argmax().item()
            best_label = self._label_names[best_index]
            confidence = probabilities[index, best_index].item()
            caption_label = caption_override(captions[index], confidence)
            if caption_label is not None:
                logger.debug("caption override path=%s visual_label=%s visual_confidence=%.4f caption_label=%s", path, best_label, confidence, caption_label)
            results.append(ClassificationResult(path, caption_label or best_label, round(confidence, 4), scores))
        return results
