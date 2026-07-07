"""Unit tests for ClipClassifier — mocked model, no network."""

from __future__ import annotations

import pytest

from app.services.classifier import ClipClassifier, ClassificationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier(mock_clip_model) -> ClipClassifier:
    return ClipClassifier(
        model_name="openai/clip-vit-base-patch32",
        device="cpu",
        offline=True,
        linear_probe_path=mock_clip_model["probe_path"],
        label_map_path=mock_clip_model["label_map_path"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_linear_head_loaded(self, classifier):
        assert classifier._linear_head is not None
        assert classifier._linear_head.out_features == 4
        assert classifier._linear_head.in_features == 512

    def test_label_names_are_4_training_classes(self, classifier):
        assert classifier._label_names == ["bar_chart", "line_chart", "sem", "xrd"]
        assert len(classifier._label_names) == 4

    def test_labels_match_config(self, classifier):
        from app.config import settings

        trained = set(classifier._label_names)
        config_labels = set(settings.class_labels.keys())
        # "other" is in config but not in trained labels
        assert trained.issubset(config_labels)
        assert "other" not in trained


class TestClassifySingle:
    def test_returns_classification_result(self, classifier, sample_image_rgb):
        result = classifier.classify_single(sample_image_rgb)
        assert isinstance(result, ClassificationResult)
        assert result.label in {"bar_chart", "line_chart", "sem", "xrd", "other"}
        assert 0.0 <= result.confidence <= 1.0

    def test_all_scores_cover_trained_classes(self, classifier, sample_image_rgb):
        result = classifier.classify_single(sample_image_rgb)
        assert set(result.all_scores.keys()) == set(classifier._label_names)

    def test_all_scores_sum_to_one(self, classifier, sample_image_rgb):
        result = classifier.classify_single(sample_image_rgb)
        total = sum(result.all_scores.values())
        assert abs(total - 1.0) < 0.001

    def test_file_not_found_raises(self, classifier, tmp_path):
        nonexistent = tmp_path / "does_not_exist.jpg"
        with pytest.raises(FileNotFoundError):
            classifier.classify_single(nonexistent)

    def test_corrupt_image_raises(self, classifier, corrupt_image):
        with pytest.raises(ValueError, match="Cannot load image"):
            classifier.classify_single(corrupt_image)

    def test_grayscale_image_converted(self, classifier, sample_image_gray):
        result = classifier.classify_single(sample_image_gray)
        assert result.label in {"bar_chart", "line_chart", "sem", "xrd", "other"}

    def test_rgba_image_converted(self, classifier, sample_image_rgba):
        result = classifier.classify_single(sample_image_rgba)
        assert result.label in {"bar_chart", "line_chart", "sem", "xrd", "other"}

    def test_threshold_forces_other(self, classifier, sample_image_rgb):
        """With threshold=0.99 the best label should be forced to 'other'."""
        result = classifier.classify_single(sample_image_rgb, confidence_threshold=0.99)
        assert result.label == "other"


class TestClassifyBatch:
    @pytest.fixture
    def three_images(self, sample_image_rgb, sample_image_gray, sample_image_rgba):
        return [sample_image_rgb, sample_image_gray, sample_image_rgba]

    def test_returns_all_results(self, classifier, three_images):
        results = classifier.classify_batch(three_images)
        assert len(results) == len(three_images)

    def test_order_preserved(self, classifier, three_images):
        results = classifier.classify_batch(three_images)
        for i, r in enumerate(results):
            assert str(three_images[i]) in r.image_path

    def test_each_result_has_valid_label(self, classifier, three_images):
        results = classifier.classify_batch(three_images)
        for r in results:
            assert r.label in {"bar_chart", "line_chart", "sem", "xrd", "other", "error"}

    def test_mixed_valid_and_missing(self, classifier, sample_image_rgb, tmp_path):
        nonexistent = tmp_path / "ghost.jpg"
        paths = [sample_image_rgb, nonexistent, sample_image_rgb]
        results = classifier.classify_batch(paths)
        assert len(results) == 3
        assert results[1].label == "error"
        assert results[1].confidence == 0.0
        assert results[0].label != "error"
        assert results[2].label != "error"


class TestClassificationResult:
    def test_dataclass_fields(self):
        r = ClassificationResult(
            image_path="test.jpg",
            label="sem",
            confidence=0.95,
            all_scores={"bar_chart": 0.01, "line_chart": 0.01, "sem": 0.95, "xrd": 0.03},
        )
        assert r.image_path == "test.jpg"
        assert r.label == "sem"
        assert r.confidence == 0.95
