"""Shared fixtures for the CLIP classifier test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image_rgb(tmp_path):
    """Create a small colourful RGB image (simulates a chart)."""
    img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    path = tmp_path / "chart.jpg"
    img.save(path)
    return path


@pytest.fixture
def sample_image_gray(tmp_path):
    """Create a small grayscale image (simulates SEM micrograph)."""
    img = Image.new("L", (224, 224), color=128)
    rgb = img.convert("RGB")
    path = tmp_path / "sem_micrograph.jpg"
    rgb.save(path)
    return path


@pytest.fixture
def sample_image_rgba(tmp_path):
    """Create a small RGBA image (tests channel conversion)."""
    img = Image.new("RGBA", (224, 224), color=(100, 150, 200, 255))
    path = tmp_path / "rgba_image.png"
    img.save(path)
    return path


@pytest.fixture
def corrupt_image(tmp_path):
    """Create a file with a .jpg extension that is NOT a valid image."""
    path = tmp_path / "corrupt.jpg"
    path.write_text("this is not an image")
    return path


# ---------------------------------------------------------------------------
# Mocked CLIP model
# ---------------------------------------------------------------------------


class _MockVisionModel:
    """Returns a distinct image embedding based on the dominant colour channel."""

    def __call__(self, pixel_values):
        import torch

        # Batch size
        b = pixel_values.shape[0]
        # Simple heuristic: if mean red > mean blue → "chart-like" (bar/line/xrd)
        # else → "micrograph-like" (sem/other)
        mean_r = pixel_values[:, 0, :, :].mean(dim=[1, 2])  # [B]
        mean_b = pixel_values[:, 2, :, :].mean(dim=[1, 2])  # [B]

        # Create distinct embeddings per class
        dim = 512
        emb = torch.zeros(b, dim)

        # Colourful → bar_chart (index 0 in PROMPT_TEMPLATES keys)
        # Grayscale → sem (index 2)
        colorful = (mean_r - mean_b) > 0.05

        emb[colorful, 0] = 1.0  # aligns with bar_chart text embedding
        emb[~colorful, 2] = 1.0  # aligns with sem text embedding

        # Normalize
        norm = emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return emb / norm


class _MockTextModel:
    """Returns text embeddings that align with the mock image embeddings."""

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        import torch

        num_texts = input_ids.shape[0]
        dim = 512
        emb = torch.zeros(num_texts, dim)

        # We know the prompt order from PROMPT_TEMPLATES:
        # bar_chart (3 templates) → index 0, line_chart (3) → 1,
        # sem (3) → 2, xrd (3) → 3, other (4) → 4
        template_idx = 0
        for class_idx, count in enumerate([3, 3, 3, 3, 4]):
            for _ in range(count):
                if template_idx < num_texts:
                    emb[template_idx, class_idx] = 1.0
                template_idx += 1

        norm = emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return emb / norm


@pytest.fixture
def mock_clip_model(mocker):
    """Mock the CLIPModel so tests run without downloading real weights."""
    import torch
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.config = MagicMock()
    mock.device = torch.device("cpu")
    mock.logit_scale = torch.nn.Parameter(torch.tensor(2.6592))  # ln(1/0.07)

    mock.get_image_features = _MockVisionModel()
    mock.get_text_features = _MockTextModel()
    mock.eval = MagicMock()
    mock.to = MagicMock(return_value=mock)

    # Patch CLIPModel.from_pretrained
    mocker.patch(
        "transformers.CLIPModel.from_pretrained",
        return_value=mock,
    )

    # Patch CLIPProcessor.from_pretrained — return a real processor
    # (the real one is lightweight, no network needed for init)
    mocker.patch(
        "transformers.CLIPProcessor.from_pretrained",
        return_value=_dummy_processor(),
    )

    return mock


def _dummy_processor():
    """A minimal processor-like object that tensor-ises images."""
    from unittest.mock import MagicMock
    import torch
    from PIL import Image

    proc = MagicMock()

    def _fake_process(text=None, images=None, return_tensors="pt", padding=True):
        result = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            result["input_ids"] = torch.ones(len(text), 77, dtype=torch.long)
            result["attention_mask"] = torch.ones(len(text), 77, dtype=torch.long)

        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            # Produce a predictable dummy pixel tensor
            batch = []
            for img in images:
                arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
                arr = arr.reshape(img.size[1], img.size[0], -1).permute(2, 0, 1)
                if arr.shape[0] == 1:  # grayscale → expand to 3
                    arr = arr.expand(3, -1, -1)
                elif arr.shape[0] == 4:  # RGBA → drop alpha
                    arr = arr[:3]
                batch.append(arr)
            result["pixel_values"] = torch.stack(batch)

        return result

    proc.side_effect = _fake_process
    proc.__call__ = _fake_process
    return proc


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def test_app(mock_clip_model):
    """Return a TestClient wired to an isolated FastAPI app."""
    from app.main import create_app
    from app.config import Settings

    # Override settings for testing
    import app.config as cfg
    original = cfg.settings
    cfg.settings = Settings(
        image_base_path=".",
        model_cache_dir=".",
        device="cpu",
        transformers_offline=True,
    )

    app = create_app()
    client = TestClient(app)

    yield client

    # Restore original
    cfg.settings = original


# ---------------------------------------------------------------------------
# Real classifier (integration tests only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_classifier():
    """Create a real ClipClassifier using actual model weights.

    Requires the model to be downloaded to model_cache/ first.
    Module-scoped so we only load once.
    """
    from app.services.classifier import ClipClassifier
    from app.config import settings

    return ClipClassifier(
        model_name=settings.clip_model_name,
        cache_dir=str(settings.model_cache_dir),
        device="cpu",
        offline=True,
    )
