"""Integration tests — real model weights + sample images.

These tests require:

1. The CLIP model downloaded to ``model_cache/``
   (run ``python scripts/download_model.py`` first).
2. Sample images placed in ``samples/<class_name>/``.

Marked with ``@pytest.mark.integration`` and skipped by default.
Run with::

    pytest tests/test_integration.py -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLES_DIR = Path("samples")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_sample_images() -> dict[str, list[Path]]:
    """Scan samples/ for class-labelled directories and their images."""
    if not SAMPLES_DIR.exists():
        return {}

    samples: dict[str, list[Path]] = {}
    for class_dir in SAMPLES_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*"))  # match jpg, png, tiff, etc.
        if images:
            samples[class_dir.name] = images
    return samples


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealModelLoad:
    def test_classifier_initializes(self, real_classifier):
        """Sanity check — the real model loads without errors."""
        assert real_classifier is not None
        assert len(real_classifier._labels) >= 5
        assert real_classifier._text_embeddings is not None


@pytest.mark.integration
class TestRealModelInference:
    def test_classify_single_returns_result(self, real_classifier):
        """Run inference on the first available sample image."""
        all_samples = _collect_sample_images()
        if not all_samples:
            pytest.skip("No sample images found in samples/")

        # Pick any image
        first_class = next(iter(all_samples.values()))
        image_path = first_class[0]

        result = real_classifier.classify_single(image_path)
        assert result.label in real_classifier._labels
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_batch(self, real_classifier):
        """Batch inference on all available samples."""
        all_samples = _collect_sample_images()
        all_paths = [p for paths in all_samples.values() for p in paths]
        if len(all_paths) < 2:
            pytest.skip("Need at least 2 sample images for batch test")

        results = real_classifier.classify_batch(all_paths[:5])
        assert len(results) == min(5, len(all_paths))
        for r in results:
            assert r.label != "error"


@pytest.mark.integration
class TestKnownSamples:
    def test_known_samples_classified_correctly(self, real_classifier):
        """Each sample in samples/<class>/ should get classified as that class."""
        all_samples = _collect_sample_images()
        if not all_samples:
            pytest.skip("No sample images found in samples/")

        correct = 0
        total = 0
        failures: list[str] = []

        for expected_label, image_paths in all_samples.items():
            for image_path in image_paths:
                result = real_classifier.classify_single(image_path)
                total += 1
                if result.label == expected_label:
                    correct += 1
                else:
                    failures.append(
                        f"{image_path.name}: expected={expected_label}, "
                        f"got={result.label} (conf={result.confidence:.3f})"
                    )

        accuracy = correct / total if total > 0 else 0.0
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")

        if failures:
            print("Misclassifications:")
            for f in failures:
                print(f"  {f}")

        # At minimum, accuracy should be better than random (20% for 5 classes)
        assert accuracy >= 0.20, f"Accuracy {accuracy:.1%} worse than random chance"

    def test_confidence_threshold_tuning(self, real_classifier):
        """Smoke-test different confidence thresholds."""
        all_samples = _collect_sample_images()
        if not all_samples:
            pytest.skip("No sample images found in samples/")

        first_path = next(iter(next(iter(all_samples.values()))))

        # Without threshold — should get a real label
        r1 = real_classifier.classify_single(first_path, confidence_threshold=None)
        assert r1.label != "error"

        # With very high threshold — should force "other"
        r2 = real_classifier.classify_single(first_path, confidence_threshold=0.999)
        assert r2.label == "other"
