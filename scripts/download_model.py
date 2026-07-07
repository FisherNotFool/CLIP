#!/usr/bin/env python3
"""Download CLIP model to a local cache directory for offline deployment.

Run this ONCE on an internet-connected machine.  The resulting cache directory
can be transferred to the air-gapped deployment target.

Usage::

    python scripts/download_model.py
    python scripts/download_model.py --model openai/clip-vit-base-patch32
    python scripts/download_model.py --cache-dir /opt/models/clip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download a CLIP model to a local cache for offline use.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"),
        help="HuggingFace model id (default: openai/clip-vit-base-patch32)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("MODEL_CACHE_DIR", "./model_cache"),
        help="Local directory to store cached model files (default: ./model_cache)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).resolve()
    model_name = args.model

    print(f"Model:     {model_name}")
    print(f"Cache dir: {cache_dir}")
    print()

    # 1. Download full repository snapshot via huggingface_hub
    print("[1/3] Downloading model repository (snapshot_download) ...")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_name, cache_dir=str(cache_dir))
    print("       Done.")

    # 2. Load via transformers to ensure processor configs are cached correctly
    print("[2/3] Loading model & processor (transformers) ...")
    os.environ["HF_HOME"] = str(cache_dir)

    from transformers import CLIPModel, CLIPProcessor

    CLIPModel.from_pretrained(model_name, cache_dir=str(cache_dir))
    CLIPProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
    print("       Done.")

    # 3. Verify offline loadability
    print("[3/3] Verifying offline load ...")
    CLIPModel.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    CLIPProcessor.from_pretrained(model_name, cache_dir=str(cache_dir), local_files_only=True)
    print("       OK — model verified for offline use.")

    # Report
    total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
    print()
    print(f"Cache directory : {cache_dir}")
    print(f"Total size      : {total_size / (1024 ** 3):.2f} GiB")
    print()
    print("Ready for offline deployment.  Copy this directory to the target machine.")
    print(f"Set MODEL_CACHE_DIR={cache_dir} and TRANSFORMERS_OFFLINE=1 in .env.")


if __name__ == "__main__":
    main()
