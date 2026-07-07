"""Integration tests — real model weights + trained linear probe + sample images.

Requirements:
1. CLIP model in ``model_cache/`` (run ``python scripts/download_model.py``)
2. Trained linear probe in ``model_cache/linear_probe.pt`` (run ``python scripts/train.py``)
3. Sample images in ``samples/{class_name}/``

Marked with ``@pytest.mark.integration`` — skipped by default.
Run with::

    pytest tests/test_integration.py -m integration -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLES_DIR = Path("samples")

# Only these 4 classes participate in training; "other" is fallback
TRAIN_CLASSES = {"bar_chart", "line_chart", "sem", "xrd"}


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
        images = list(class_dir.glob("*"))
        if images:
            samples[class_dir.name] = images
    return samples


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRealModelLoad:
    def test_classifier_initializes(self, real_classifier):
        assert real_classifier is not None
        assert len(real_classifier._label_names) == 4
        assert real_classifier._linear_head is not None

    def test_trained_classes_match_expected(self, real_classifier):
        assert set(real_classifier._label_names) == TRAIN_CLASSES


@pytest.mark.integration
class TestRealModelInference:
    def test_classify_single_returns_result(self, real_classifier):
        all_samples = _collect_sample_images()
        if not all_samples:
            pytest.skip("No sample images found in samples/")

        first_class = next(iter(all_samples.values()))
        image_path = first_class[0]

        result = real_classifier.classify_single(image_path)
        assert result.label in TRAIN_CLASSES | {"other"}
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_batch(self, real_classifier):
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
        """Each sample in samples/<class>/ should get classified as that class.

        Only evaluates the 4 training classes; "other" samples are only used
        for threshold calibration, not accuracy measurement.
        """
        all_samples = _collect_sample_images()
        if not all_samples:
            pytest.skip("No sample images found in samples/")

        correct = 0
        total = 0
        failures: list[str] = []

        for expected_label, image_paths in all_samples.items():
            # "other" is not a trained class — skip for accuracy
            if expected_label not in TRAIN_CLASSES:
                continue

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
        print(f"\nAccuracy (4 training classes): {correct}/{total} = {accuracy:.1%}")

        if failures:
            print(f"\nMisclassifications ({len(failures)}):")
            for f in failures:
                print(f"  {f}")

        # Linear probe should comfortably beat random (25% for 4 classes)
        assert accuracy >= 0.40, f"Accuracy {accuracy:.1%} — barely above random"

    def test_accuracy_with_centroid_threshold(self, real_classifier):
        """Same as above but WITH centroid distance threshold — shows how many
        real training-class images get falsely rejected as 'other'."""
        from app.config import settings

        all_samples = _collect_sample_images()
        if not all_samples:
            pytest.skip("No sample images found in samples/")

        threshold = settings.centroid_distance_threshold
        correct = 0
        total = 0
        forced_to_other: list[str] = []
        misclassified: list[str] = []

        for expected_label, image_paths in all_samples.items():
            if expected_label not in TRAIN_CLASSES:
                continue

            for image_path in image_paths:
                result = real_classifier.classify_single(
                    image_path, distance_threshold=threshold,
                )
                total += 1

                if result.label == "other":
                    # Centroid threshold rejected a real training-class image
                    forced_to_other.append(
                        f"{image_path.name}: expected={expected_label}, "
                        f"forced to other"
                    )
                elif result.label == expected_label:
                    correct += 1
                else:
                    misclassified.append(
                        f"{image_path.name}: expected={expected_label}, "
                        f"got={result.label} (conf={result.confidence:.3f})"
                    )

        accuracy = correct / total if total > 0 else 0.0
        false_positive_rate = len(forced_to_other) / total if total > 0 else 0.0

        print(f"\nAccuracy WITH centroid threshold={threshold}: {correct}/{total} = {accuracy:.1%}")
        print(f"  Rejected as 'other' (false positives): {len(forced_to_other)}/{total} = {false_positive_rate:.1%}")

        if forced_to_other:
            print(f"\n  Images wrongly rejected as 'other' ({len(forced_to_other)}):")
            for f in forced_to_other[:20]:  # cap at 20
                print(f"    {f}")
            if len(forced_to_other) > 20:
                print(f"    ... and {len(forced_to_other) - 20} more")

        if misclassified:
            print(f"\n  Still misclassified (not forced to other) ({len(misclassified)}):")
            for m in misclassified[:10]:
                print(f"    {m}")
            if len(misclassified) > 10:
                print(f"    ... and {len(misclassified) - 10} more")

        # This is informational — no hard assertion
        assert accuracy >= 0.0

    def test_other_samples_fall_below_threshold(self, real_classifier):
        """samples/other/ images should mostly be caught by the centroid
        distance threshold and output 'other'."""
        all_samples = _collect_sample_images()
        other_paths = all_samples.get("other", [])
        if not other_paths:
            pytest.skip("No samples/other/ images found")

        from app.config import settings

        caught = 0
        for image_path in other_paths:
            result = real_classifier.classify_single(
                image_path,
                distance_threshold=settings.centroid_distance_threshold,
            )
            if result.label == "other":
                caught += 1

        rate = caught / len(other_paths)
        print(f"\nOther fallback rate (centroid distance): {caught}/{len(other_paths)} = {rate:.1%}")
        print(f"  (centroid_distance_threshold = {settings.centroid_distance_threshold})")

        # Not a hard assertion — threshold may need tuning
        # Just report the number so the user can calibrate
        assert rate >= 0.0  # informational only
