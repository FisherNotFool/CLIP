#!/usr/bin/env python3
"""Train a linear probe on frozen CLIP ViT features for 4-class classification.

Only trains on bar_chart / line_chart / sem / xrd.  The "other" category is
handled at inference time via **cosine distance from class centroids**:

1. Compute the 512-dim centroid (mean feature) for each training class.
2. At inference time, if a new image's minimum cosine distance to *any*
   centroid exceeds a threshold, it is classified as "other".
3. Otherwise, the linear probe decides which of the 4 classes it belongs to.

This separates "is it any known class at all?" (centroid distance) from
"which class is it?" (linear probe).

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
DEFAULT_CENTROIDS = Path("model_cache/centroids.pt")
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
# Centroids — class-conditional means for outlier detection
# ---------------------------------------------------------------------------


def compute_centroids(
    features: dict,
    label_names: list[str],
    device: str = "cpu",
) -> dict:
    """Compute the 512-dim mean feature vector for each training class.

    Uses all extracted features (not just the train split) since centroids
    are pure averages — no gradient-based learning is involved.

    Returns a dict with L2-normalised centroids ready for cosine-distance checks.
    """
    import torch.nn.functional as F

    X = features["X"]
    y = features["y"]

    centroids = []
    for class_idx in range(len(label_names)):
        mask = y == class_idx
        if mask.sum() == 0:
            raise ValueError(f"No features found for class '{label_names[class_idx]}'")
        centroid = X[mask].mean(dim=0)  # [512]
        # L2-normalise so cosine similarity = dot product
        centroid = F.normalize(centroid, p=2, dim=0)
        centroids.append(centroid)

    centroids_tensor = torch.stack(centroids)  # [C, 512]

    logger.info("Computed %d class centroids (L2-normalised):", len(label_names))
    for i, name in enumerate(label_names):
        logger.info("  %s: norm=%.4f", name, centroids_tensor[i].norm().item())

    return {"centroids": centroids_tensor, "label_names": label_names}


def save_centroids(
    centroids_data: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(centroids_data, output_path)
    logger.info("Saved centroids to %s", output_path)


# ---------------------------------------------------------------------------
# Other-class evaluation (centroid-distance based threshold calibration)
# ---------------------------------------------------------------------------


def evaluate_other(
    centroids_data: dict,
    device: str = "cpu",
) -> list[dict]:
    """Run centroid-distance check against samples/other/ images.

    Computes cosine distance to the nearest class centroid for every image
    in samples/other/ and prints a distribution to help pick a
    ``centroid_distance_threshold``.
    """
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    other_dir = SAMPLES_DIR / "other"
    if not other_dir.is_dir():
        logger.info("No samples/other/ directory — skipping threshold calibration.")
        return []

    centroids = centroids_data["centroids"].to(device)  # [C, 512], already L2-normalised
    label_names = centroids_data["label_names"]

    # Load CLIP vision encoder
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
                features = _to_tensor(features).squeeze(0)  # [512]
                features = F.normalize(features, p=2, dim=0)  # L2-normalise

                # Cosine distance = 1 - cosine_similarity
                cos_sim = features @ centroids.T              # [C]
                cos_dist = 1.0 - cos_sim                      # [C]
                min_dist, best_idx = cos_dist.min(dim=0)

            results.append({
                "path": str(img_path),
                "nearest_class": label_names[best_idx.item()],
                "min_cosine_dist": round(min_dist.item(), 4),
                "all_distances": {
                    label_names[i]: round(cos_dist[i].item(), 4)
                    for i in range(len(label_names))
                },
            })
        except Exception:
            logger.exception("Failed to process %s", img_path)

    # --- Distribution summary ---
    distances = sorted([r["min_cosine_dist"] for r in results])
    n = len(distances)
    if n == 0:
        return results

    def _percentile(sorted_vals: list[float], p: float) -> float:
        """Linear-interpolation percentile (like numpy)."""
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        k = (p / 100) * (n - 1)
        f = int(k)
        c = k - f
        if f + 1 >= n:
            return sorted_vals[-1]
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])

    pcts = [0, 25, 50, 75, 85, 90, 95, 100]
    pct_values = {p: _percentile(distances, p) for p in pcts}

    print("\n" + "=" * 60)
    print(f"Other-Class Evaluation ({n} images) — Cosine Distance to Nearest Centroid")
    print("=" * 60)
    print(f"  Distance range: [{distances[0]:.4f}, {distances[-1]:.4f}]")
    print(f"  Percentiles:")
    for p in pcts:
        print(f"    {p:3d}th: {pct_values[p]:.4f}")
    print()
    print(f"  Suggested threshold: {pct_values[85]:.4f}  (85th percentile — catches ~85% of 'other')")
    print(f"  If you pick {pct_values[75]:.4f} (75th), fewer 'other' images leak through")
    print(f"  but more real charts may also be flagged as 'other'.")

    # Show per-image details
    print(f"\n  Per-image predictions:")
    for r in results:
        dist = r["min_cosine_dist"]
        nearest = r["nearest_class"]
        print(f"  [{dist:.4f}] {Path(r['path']).name}: nearest={nearest}")

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

    # --- Save probe + label map ---
    save_probe(probe, features["label_names"], args.output, args.label_map)

    # --- Compute & save centroids (for outlier-based "other" detection) ---
    centroids_data = compute_centroids(features, features["label_names"], args.device)
    save_centroids(centroids_data, DEFAULT_CENTROIDS)

    # --- Threshold calibration on "other" (cosine distance) ---
    evaluate_other(centroids_data, device=args.device)

    print("\nDone. Next steps:")
    print("  1. Check the classification report above.")
    print("  2. Check the cosine-distance percentiles above and set")
    print("     CENTROID_DISTANCE_THRESHOLD in .env (e.g. 0.30).")
    print("  3. Run: pytest tests/test_classifier.py tests/test_api.py -v")
    print("  4. Start service: uvicorn app.main:app --port 8011")


if __name__ == "__main__":
    main()
