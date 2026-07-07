"""Central configuration via pydantic-settings.

All tunable values live here. Environment variables or .env file override defaults.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- Model ---
    clip_model_name: str = "openai/clip-vit-base-patch32"
    model_cache_dir: Path = Path("./model_cache")
    device: str = "cpu"  # "cpu", "cuda", "cuda:0"
    transformers_offline: bool = False

    # --- Service ---
    host: str = "0.0.0.0"
    port: int = 8011

    # --- Image processing ---
    image_base_path: Path = Path("./outputs")
    confidence_threshold: float = 0.30
    max_image_size: int = 1920
    batch_size: int = 8

    # --- Labels — single source of truth for available classes ---
    class_labels: dict[str, str] = {
        "bar_chart": "Bar Chart",
        "line_chart": "Line Chart",
        "sem": "SEM Micrograph",
        "xrd": "XRD Pattern",
        "other": "Other / Schematic / Photo",
    }


settings = Settings()
