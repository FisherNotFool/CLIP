"""Shared fixtures for the CLIP classifier test suite (linear probe)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image_rgb(tmp_path):
    """A small colourful RGB image (simulates a chart)."""
    img = Image.new("RGB", (224, 224), color=(200, 50, 50))
    path = tmp_path / "chart.jpg"
    img.save(path)
    return path


@pytest.fixture
def sample_image_gray(tmp_path):
    """A small grayscale image (simulates SEM micrograph)."""
    img = Image.new("L", (224, 224), color=128)
    rgb = img.convert("RGB")
    path = tmp_path / "sem_micrograph.jpg"
    rgb.save(path)
    return path


@pytest.fixture
def sample_image_rgba(tmp_path):
    """A small RGBA image (tests channel conversion)."""
    img = Image.new("RGBA", (224, 224), color=(100, 150, 200, 255))
    path = tmp_path / "rgba_image.png"
    img.save(path)
    return path


@pytest.fixture
def corrupt_image(tmp_path):
    """A file with a .jpg extension that is NOT a valid image."""
    path = tmp_path / "corrupt.jpg"
    path.write_text("this is not an image")
    return path


# ---------------------------------------------------------------------------
# Mocked CLIP vision encoder
# ---------------------------------------------------------------------------


class _MockVisionModel:
    """Returns a deterministic 512-dim embedding based on colour channel.

    Colourful images (mean red > mean blue) → feature[0] = 1.0  → bar_chart (class 0)
    Grayscale images                     → feature[2] = 1.0  → sem        (class 2)

    The mock linear head maps feature[c] = 1.0 → class c with weight[c,c]=1.0.
    """

    def __call__(self, pixel_values):
        import torch

        b = pixel_values.shape[0]
        mean_r = pixel_values[:, 0, :, :].mean(dim=[1, 2])
        mean_b = pixel_values[:, 2, :, :].mean(dim=[1, 2])

        dim = 512
        emb = torch.zeros(b, dim)
        colorful = (mean_r - mean_b) > 0.05
        emb[colorful, 0] = 1.0   # → bar_chart
        emb[~colorful, 2] = 1.0  # → sem
        return emb


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
            batch = []
            for img in images:
                arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
                arr = arr.reshape(img.size[1], img.size[0], -1).permute(2, 0, 1)
                if arr.shape[0] == 1:
                    arr = arr.expand(3, -1, -1)
                elif arr.shape[0] == 4:
                    arr = arr[:3]
                batch.append(arr)
            result["pixel_values"] = torch.stack(batch)

        return result

    proc.side_effect = _fake_process
    proc.__call__ = _fake_process
    return proc


# ---------------------------------------------------------------------------
# Mock CLIP model + linear probe files
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_clip_model(mocker, tmp_path):
    """Mock CLIPModel + create temporary linear probe artifacts."""
    import torch
    from unittest.mock import MagicMock

    # --- Mock linear probe weights (4 classes, hand-crafted) ---
    # weight[c, c] = 1.0 → feature channel c maps to class c
    # bias[3] = -5.0 → xrd (class 3) needs stronger signal; default → bar (0) or sem (2)
    probe_path = tmp_path / "linear_probe.pt"
    head = torch.nn.Linear(512, 4)
    head.weight.data.fill_(0.0)
    head.bias.data.fill_(0.0)
    for c in range(4):
        head.weight.data[c, c] = 1.0
    torch.save(head.state_dict(), probe_path)

    # --- Mock label map ---
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(
        json.dumps({"label_names": ["bar_chart", "line_chart", "sem", "xrd"]})
    )

    # --- Mock centroids (unit vectors along each class dimension) ---
    centroids_path = tmp_path / "centroids.pt"
    centroids_tensor = torch.zeros(4, 512)
    for c in range(4):
        centroids_tensor[c, c] = 1.0  # already L2-normalised
    torch.save(
        {"centroids": centroids_tensor, "label_names": ["bar_chart", "line_chart", "sem", "xrd"]},
        centroids_path,
    )

    # --- Mock CLIPModel ---
    mock = MagicMock()
    mock.config = MagicMock()
    mock.device = torch.device("cpu")
    mock.logit_scale = torch.nn.Parameter(torch.tensor(2.6592))
    mock.get_image_features = _MockVisionModel()
    mock.eval = MagicMock()
    mock.to = MagicMock(return_value=mock)

    mocker.patch("transformers.CLIPModel.from_pretrained", return_value=mock)
    mocker.patch(
        "transformers.CLIPProcessor.from_pretrained",
        return_value=_dummy_processor(),
    )

    return {
        "model": mock,
        "probe_path": probe_path,
        "label_map_path": label_map_path,
        "centroids_path": centroids_path,
    }


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def test_app(mock_clip_model):
    """Return a TestClient wired to an isolated FastAPI app.

    We bypass the lifespan (which TestClient doesn't reliably trigger) and
    inject a classifier built from mock artifacts directly.
    """
    from app.main import create_app
    from app.config import Settings
    from app.services.classifier import ClipClassifier

    test_settings = Settings(
        image_base_path=".",
        model_cache_dir=".",
        device="cpu",
        transformers_offline=True,
        linear_probe_path=mock_clip_model["probe_path"],
        label_map_path=mock_clip_model["label_map_path"],
        centroids_path=mock_clip_model["centroids_path"],
    )

    app = create_app(test_settings)

    # Bypass lifespan — create classifier from mock artifacts directly
    app.state.classifier = ClipClassifier(
        model_name="openai/clip-vit-base-patch32",
        device="cpu",
        offline=True,
        linear_probe_path=str(mock_clip_model["probe_path"]),
        label_map_path=str(mock_clip_model["label_map_path"]),
        centroids_path=str(mock_clip_model["centroids_path"]),
    )

    client = TestClient(app)
    yield client


# ---------------------------------------------------------------------------
# Real classifier (integration tests only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_classifier():
    """Create a real ClipClassifier using actual model weights + trained probe.

    Requires:
    - model_cache/ with CLIP weights
    - model_cache/linear_probe.pt + model_cache/label_map.json (run train.py first)
    """
    from app.services.classifier import ClipClassifier
    from app.config import settings

    return ClipClassifier(
        model_name=settings.clip_model_name,
        cache_dir=str(settings.model_cache_dir),
        device="cpu",
        offline=True,
        linear_probe_path=settings.linear_probe_path,
        label_map_path=settings.label_map_path,
        centroids_path=settings.centroids_path,
    )
