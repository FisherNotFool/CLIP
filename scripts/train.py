#!/usr/bin/env python3
"""Train a linear probe on frozen CLIP ViT features for 4-class classification.

Only trains on bar_chart / line_chart / sem / xrd.  The "other" category is
handled at inference time via a confidence threshold — images that don't
confidently match any of the 4 classes fall back to "other".

Usage::

    python scripts/train.py                     # full train
    python scripts/train.py --extract-only      # just cache CLIP features
    python scripts/train.py --epochs 200 --lr 0.005
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Allow importing from app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLES_DIR = Path("samples")
FEATURES_CACHE = Path("scripts/features.pt")
DEFAULT_OUTPUT = Path("model_cache/linear_probe.pt")
DEFAULT_LABEL_MAP = Path("model_cache/label_map.json")
EMBEDDING_DIM = 512

# Only these 4 classes participate in training (other is fallback)
TRAIN_CLASSES = ["bar_chart", "line_chart", "sem", "xrd"]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(device: str = "cpu") -> dict:
    """Walk samples/{class}/ for each of the 4 training classes.

    Encodes every image with the frozen CLIP vision encoder and returns a dict
    with stacked features, labels, and metadata.
    """
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = (
        CLIPModel.from_pretrained(
            settings.clip_model_name,
            cache_dir=str(settings.model_cache_dir),
            local_files_only=True,
        )
        .to(device)
        .eval()
    )

    processor = CLIPProcessor.from_pretrained(
        settings.clip_model_name,
        cache_dir=str(settings.model_cache_dir),
        local_files_only=True,
    )

    X_list: list[torch.Tensor] = []
    y_list: list[int] = []
    path_list: list[str] = []

    for class_idx, class_name in enumerate(TRAIN_CLASSES):
        class_dir = SAMPLES_DIR / class_name
        if not class_dir.is_dir():
            logger.warning("Class directory not found: %s", class_dir)
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items() if k == "pixel_values"}
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                    features = _to_tensor(features)
                X_list.append(features.cpu().squeeze(0))
                y_list.append(class_idx)
                path_list.append(str(img_path))
            except Exception:
                logger.exception("Failed to process %s", img_path)

    X = torch.stack(X_list)  # [N, 512]
    y = torch.tensor(y_list)  # [N]

    logger.info("Extracted %d feature vectors:", len(X))
    for i, name in enumerate(TRAIN_CLASSES):
        count = (y == i).sum().item()
        if count < 5:
            logger.warning("  %s: %d  <-- VERY FEW samples!", name, count)
        else:
            logger.info("  %s: %d", name, count)

    return {"X": X, "y": y, "label_names": TRAIN_CLASSES, "image_paths": path_list}


def _to_tensor(output) -> torch.Tensor:
    """Extract a plain tensor from a HuggingFace model output."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Cannot extract tensor from {type(output).__name__}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Single linear layer on top of frozen CLIP features."""

    def __init__(self, in_dim: int = EMBEDDING_DIM, num_classes: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_probe(
    features: dict,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> tuple[LinearProbe, dict]:
    """Train the linear probe. Returns (model, metrics_dict)."""
    X = features["X"]
    y = features["y"]
    label_names = features["label_names"]

    # Stratified 80/20 split
    indices = list(range(len(y)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y.numpy(), random_state=42
    )
    X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
    X_test, y_test = X[test_idx].to(device), y[test_idx].to(device)

    logger.info("Train: %d  Test: %d", len(train_idx), len(test_idx))

    # Compute class weights to counter imbalance (e.g. rare xrd gets higher weight)
    class_counts = torch.bincount(y_train, minlength=len(label_names))
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    class_weights = class_weights / class_weights.sum() * len(label_names)
    class_weights = class_weights.to(device)
    logger.info("Class weights: %s", {n: round(w.item(), 3) for n, w in zip(label_names, class_weights)})

    model = LinearProbe(in_dim=EMBEDDING_DIM, num_classes=len(label_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(1) == y_train).float().mean().item()
                test_logits = model(X_test)
                test_acc = (test_logits.argmax(1) == y_test).float().mean().item()
            logger.info(
                "Epoch %3d: loss=%.4f  train_acc=%.3f  test_acc=%.3f",
                epoch + 1, loss.item(), train_acc, test_acc,
            )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        y_pred = test_logits.argmax(1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    print("\n" + "=" * 60)
    print("Classification Report (Test Set — 4 training classes)")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    return model, {"classification_report": report, "confusion_matrix": cm.tolist()}


# ---------------------------------------------------------------------------
# Other-class evaluation (threshold calibration)
# ---------------------------------------------------------------------------


def evaluate_other(
    model: LinearProbe,
    features_cache: dict,
    device: str = "cpu",
) -> list[dict]:
    """Run the 4-class probe against samples/other/ images.

    Returns per-image probabilities to help calibrate the confidence threshold.
    """
    other_dir = SAMPLES_DIR / "other"
    if not other_dir.is_dir():
        logger.info("No samples/other/ directory — skipping threshold calibration.")
        return []

    # Load the full CLIP model to encode other samples
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    clip = (
        CLIPModel.from_pretrained(
            settings.clip_model_name,
            cache_dir=str(settings.model_cache_dir),
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    processor = CLIPProcessor.from_pretrained(
        settings.clip_model_name,
        cache_dir=str(settings.model_cache_dir),
        local_files_only=True,
    )

    model = model.to(device).eval()
    results: list[dict] = []

    for img_path in sorted(other_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if k == "pixel_values"}
            with torch.no_grad():
                features = clip.get_image_features(**inputs)
                features = _to_tensor(features)
                logits = model(features)
                probs = logits.softmax(dim=-1).squeeze(0)

            best_idx = probs.argmax().item()
            results.append({
                "path": str(img_path),
                "predicted": TRAIN_CLASSES[best_idx],
                "confidence": round(probs[best_idx].item(), 4),
                "scores": {TRAIN_CLASSES[i]: round(probs[i].item(), 4) for i in range(len(TRAIN_CLASSES))},
            })
        except Exception:
            logger.exception("Failed to process %s", img_path)

    # Summary
    above_threshold = sum(
        1 for r in results if r["confidence"] >= settings.confidence_threshold
    )
    print("\n" + "=" * 60)
    print(f"Other-Class Evaluation ({len(results)} images)")
    print("=" * 60)
    print(f"  Threshold: {settings.confidence_threshold}")
    print(f"  Above threshold (would be misclassified): {above_threshold}/{len(results)}")
    print(f"  Below threshold (correctly → other):   {len(results) - above_threshold}/{len(results)}")

    if results:
        print("\n  Per-image predictions:")
        for r in results:
            flag = "MISCLASSIFIED" if r["confidence"] >= settings.confidence_threshold else "ok"
            print(f"  [{flag}] {Path(r['path']).name}: → {r['predicted']} ({r['confidence']:.3f})")

    return results


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_probe(
    model: LinearProbe,
    label_names: list[str],
    output_path: Path,
    label_map_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save only the inner nn.Linear state_dict so classifier.py can load it
    # without needing the LinearProbe wrapper class.
    torch.save(model.linear.state_dict(), output_path)
    with open(label_map_path, "w") as f:
        json.dump({"label_names": label_names}, f, indent=2)
    logger.info("Saved linear probe to %s", output_path)
    logger.info("Saved label map to %s", label_map_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear probe on CLIP features")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract and cache features, skip training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract features even if cache exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # --- Extract (or load cached) features ---
    if FEATURES_CACHE.exists() and not args.force_extract:
        logger.info("Loading cached features from %s", FEATURES_CACHE)
        features = torch.load(FEATURES_CACHE, weights_only=False)
    else:
        logger.info("Extracting features from %s ...", SAMPLES_DIR)
        features = extract_features(args.device)
        FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, FEATURES_CACHE)
        logger.info("Cached features to %s", FEATURES_CACHE)

    if args.extract_only:
        logger.info("Extraction complete. Exiting.")
        return

    # --- Train ---
    probe, metrics = train_probe(
        features,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    # --- Save ---
    save_probe(probe, features["label_names"], args.output, args.label_map)

    # --- Threshold calibration on "other" ---
    evaluate_other(probe, features, device=args.device)

    print("\nDone. Next steps:")
    print("  1. Check the classification report above.")
    print("  2. Adjust CONFIDENCE_THRESHOLD in .env based on 'other' evaluation.")
    print("  3. Run: pytest tests/test_classifier.py tests/test_api.py -v")
    print("  4. Start service: uvicorn app.main:app --port 8011")


if __name__ == "__main__":
    main()
